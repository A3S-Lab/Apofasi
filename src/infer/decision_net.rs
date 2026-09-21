//! Decision network: ModernBERT encoder + typed head + scorer + act head.

use candle_core::{Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::modernbert::{Config as EncoderConfig, ModernBert};
#[cfg(feature = "ort")]
use super::ort_encoder::OrtEncoder;
use crate::error::{Error, Result};

/// Outputs of one forward pass.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Option logits `[K]` (invalid slots already masked to a large negative).
    pub logits: Vec<f32>,
    /// Action head logits `[n_act]` (typically act / escalate). Empty when skipped.
    pub act_logits: Vec<f32>,
}

/// Optional act/escalate probabilities derived from [`ForwardOutput::act_logits`].
#[derive(Debug, Clone, Copy)]
pub struct ActOutput {
    /// Probability of acting autonomously.
    pub act: f32,
    /// Probability of escalating to a host / human.
    pub escalate: f32,
}

enum EncoderImpl {
    Candle(ModernBert),
    #[cfg(feature = "ort")]
    Ort(OrtEncoder),
}

impl EncoderImpl {
    fn forward(&self, ids: &Tensor, attn: &Tensor, all_ones: Option<bool>) -> CandleResult<Tensor> {
        match self {
            Self::Candle(enc) => enc.forward_with_pad_hint(ids, attn, all_ones),
            #[cfg(feature = "ort")]
            Self::Ort(enc) => enc.forward(ids, attn),
        }
    }

    #[cfg(feature = "ort")]
    fn forward_packed_ort(
        &self,
        ids: &[u32],
        mask: &[u32],
        b: usize,
        l: usize,
    ) -> CandleResult<Option<Tensor>> {
        match self {
            Self::Ort(enc) => Ok(Some(enc.forward_packed(ids, mask, b, l)?)),
            Self::Candle(_) => Ok(None),
        }
    }
}

/// Full decision model: encoder → type emb → transformer head → scorer / act.
pub struct DecisionNet {
    encoder: EncoderImpl,
    type_emb: Embedding,
    head: Vec<EncoderLayer>,
    scorer_norm: LayerNorm,
    scorer_fc1: Linear,
    scorer_fc2: Linear,
    act_fc1: Linear,
    act_fc2: Linear,
    device: Device,
}

impl DecisionNet {
    /// Load from a remapped safetensors [`VarBuilder`] and encoder config.
    pub fn load(
        vb: VarBuilder,
        enc_cfg: &EncoderConfig,
        head_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        let enc_vb = vb
            .clone()
            .rename_f(|name| match name.strip_prefix("model.") {
                Some(rest) => format!("encoder.{rest}"),
                None => name.to_string(),
            });
        let encoder = ModernBert::load(enc_vb, enc_cfg)
            .map_err(|e| Error::Checkpoint(format!("load encoder: {e}")))?;
        Self::load_heads(
            vb,
            enc_cfg.hidden_size,
            head_layers,
            device,
            EncoderImpl::Candle(encoder),
        )
    }

    /// CPU Scale-3 path: ORT ModernBERT encoder + Candle typed head.
    #[cfg(feature = "ort")]
    pub fn load_with_ort_encoder(
        vb: VarBuilder,
        enc_cfg: &EncoderConfig,
        head_layers: usize,
        device: &Device,
        onnx: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        if !device.is_cpu() {
            return Err(Error::Checkpoint(
                "ORT encoder is only supported on CPU".into(),
            ));
        }
        let encoder = OrtEncoder::load(onnx, enc_cfg.hidden_size)?;
        Self::load_heads(
            vb,
            enc_cfg.hidden_size,
            head_layers,
            device,
            EncoderImpl::Ort(encoder),
        )
    }

    fn load_heads(
        vb: VarBuilder,
        d: usize,
        head_layers: usize,
        device: &Device,
        encoder: EncoderImpl,
    ) -> Result<Self> {
        let type_emb = embedding(3, d, vb.pp("type_emb"))
            .map_err(|e| Error::Checkpoint(format!("load type_emb: {e}")))?;

        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            head.push(
                EncoderLayer::load(vb.pp(format!("head.layers.{i}")), d)
                    .map_err(|e| Error::Checkpoint(format!("load head.layers.{i}: {e}")))?,
            );
        }

        let scorer_norm = layer_norm(d, 1e-5, vb.pp("scorer.0"))
            .map_err(|e| Error::Checkpoint(format!("load scorer.0: {e}")))?;
        let scorer_fc1 = linear(d, d, vb.pp("scorer.1"))
            .map_err(|e| Error::Checkpoint(format!("load scorer.1: {e}")))?;
        let scorer_fc2 = linear(d, 1, vb.pp("scorer.3"))
            .map_err(|e| Error::Checkpoint(format!("load scorer.3: {e}")))?;

        let act_fc1 = linear(d + 4, 256, vb.pp("act_head.0"))
            .map_err(|e| Error::Checkpoint(format!("load act_head.0: {e}")))?;
        let act_fc2 = linear(256, 2, vb.pp("act_head.2"))
            .map_err(|e| Error::Checkpoint(format!("load act_head.2: {e}")))?;

        Ok(Self {
            encoder,
            type_emb,
            head,
            scorer_norm,
            scorer_fc1,
            scorer_fc2,
            act_fc1,
            act_fc2,
            device: device.clone(),
        })
    }
}

/// One packed question inside a [`DecisionNet::forward_batch`] call.
pub struct BatchItem<'a> {
    /// Token ids for this question (unpadded).
    pub input_ids: &'a [u32],
    /// Absolute `[MASK]` positions within `input_ids`.
    pub markers: &'a [usize],
    /// Dense question-type id.
    pub qtype: u8,
}

impl DecisionNet {
    /// Forward one packed question (`input_ids` length `L`, `markers` length `K`).
    ///
    /// Computes the act head (for hosts that want escalate logits).
    pub fn forward(
        &self,
        input_ids: &[u32],
        markers: &[usize],
        qtype: u8,
    ) -> Result<ForwardOutput> {
        self.forward_batch(
            &[BatchItem {
                input_ids,
                markers,
                qtype,
            }],
            0,
            true,
        )?
        .pop()
        .ok_or_else(|| Error::Infer("empty forward".into()))
    }

    /// Option logits only (skips act head) — hot path for [`crate::NeuralEngine`].
    pub fn forward_logits(
        &self,
        input_ids: &[u32],
        markers: &[usize],
        qtype: u8,
    ) -> Result<ForwardOutput> {
        self.forward_batch(
            &[BatchItem {
                input_ids,
                markers,
                qtype,
            }],
            0,
            false,
        )?
        .pop()
        .ok_or_else(|| Error::Infer("empty forward".into()))
    }

    /// One encoder forward for every question in the batch.
    ///
    /// Sequences are right-padded to the longest item. `pad_id` fills token
    /// slots; attention masks keep pads from changing real-token positions, so
    /// logits match a per-question [`Self::forward`].
    ///
    /// When `compute_act` is false, the act/escalate head is skipped (answers
    /// only need option logits). Hosts that gate on `confidence`/`noul` should
    /// leave this false on the hot path.
    pub fn forward_batch(
        &self,
        items: &[BatchItem<'_>],
        pad_id: u32,
        compute_act: bool,
    ) -> Result<Vec<ForwardOutput>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        self.forward_batch_inner(items, pad_id, compute_act)
            .map_err(|e| Error::Infer(format!("forward: {e}")))
    }

    fn forward_batch_inner(
        &self,
        items: &[BatchItem<'_>],
        pad_id: u32,
        compute_act: bool,
    ) -> CandleResult<Vec<ForwardOutput>> {
        let batch = items.len();
        let max_len = items
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            candle_core::bail!("empty packed sequence");
        }

        let mut ids_flat = vec![pad_id; batch * max_len];
        let mut attn_flat = vec![0u32; batch * max_len];
        let mut pad_flat = vec![1u8; batch * max_len];
        for (row, item) in items.iter().enumerate() {
            let n = item.input_ids.len();
            if n == 0 {
                candle_core::bail!("empty packed sequence");
            }
            let base = row * max_len;
            ids_flat[base..base + n].copy_from_slice(item.input_ids);
            for slot in &mut attn_flat[base..base + n] {
                *slot = 1;
            }
            for slot in &mut pad_flat[base..base + n] {
                *slot = 0;
            }
        }

        let all_ones = attn_flat.iter().all(|&x| x == 1);
        #[cfg(feature = "ort")]
        let mut h = if let Some(hidden) = self
            .encoder
            .forward_packed_ort(&ids_flat, &attn_flat, batch, max_len)?
        {
            hidden
        } else {
            let ids = Tensor::from_vec(ids_flat, (batch, max_len), &self.device)?;
            let attn = Tensor::from_vec(attn_flat, (batch, max_len), &self.device)?;
            self.encoder.forward(&ids, &attn, Some(all_ones))?
        };
        #[cfg(not(feature = "ort"))]
        let mut h = {
            let ids = Tensor::from_vec(ids_flat, (batch, max_len), &self.device)?;
            let attn = Tensor::from_vec(attn_flat, (batch, max_len), &self.device)?;
            self.encoder.forward(&ids, &attn, Some(all_ones))?
        };

        let qtypes: Vec<u32> = items.iter().map(|item| u32::from(item.qtype)).collect();
        let q = Tensor::from_vec(qtypes, batch, &self.device)?;
        let te = self.type_emb.forward(&q)?.unsqueeze(1)?; // [B, 1, D]
        h = h.broadcast_add(&te)?;

        let pad = Tensor::from_vec(pad_flat, (batch, max_len), &self.device)?;
        for layer in &self.head {
            h = layer.forward(&h, &pad)?;
        }

        // Flatten markers across the batch and score once → one host sync.
        let mut gather_idx = Vec::new();
        let mut marker_counts = Vec::with_capacity(batch);
        for (row, item) in items.iter().enumerate() {
            let seq_len = item.input_ids.len();
            marker_counts.push(item.markers.len());
            for &marker in item.markers {
                if marker >= seq_len {
                    candle_core::bail!("marker {marker} out of range for seq_len {seq_len}");
                }
                gather_idx.push((row * max_len + marker) as u32);
            }
        }
        if gather_idx.is_empty() {
            candle_core::bail!("no markers in batch");
        }
        let flat_h = h.flatten(0, 1)?; // [B*L, D]
        let idx_t = Tensor::new(gather_idx.as_slice(), &self.device)?;
        let gathered = flat_h.index_select(&idx_t, 0)?; // [N, D]
        let all_logits = gathered
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_fc1)?
            .gelu()?
            .apply(&self.scorer_fc2)?
            .squeeze(1)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;

        let mut outputs = Vec::with_capacity(batch);
        let mut offset = 0usize;
        for (row, count) in marker_counts.into_iter().enumerate() {
            let logits = all_logits[offset..offset + count].to_vec();
            offset += count;
            let act_logits = if compute_act {
                let probs = softmax_cpu(&logits);
                let k = count.max(2) as f32;
                let ent = {
                    let mut e = 0.0f32;
                    for &pi in &probs {
                        let pi = pi.clamp(1e-9, 1.0);
                        e -= pi * pi.ln();
                    }
                    e / k.ln()
                };
                let (top1, margin) = top2_margin(&probs);
                let feats = Tensor::new(&[top1, margin, ent, k / 255.0], &self.device)?
                    .to_dtype(h.dtype())?;
                let pooled = h.i((row, 0))?;
                Tensor::cat(&[&pooled, &feats], 0)?
                    .unsqueeze(0)?
                    .apply(&self.act_fc1)?
                    .gelu()?
                    .apply(&self.act_fc2)?
                    .squeeze(0)?
                    .to_dtype(candle_core::DType::F32)?
                    .to_vec1::<f32>()?
            } else {
                Vec::new()
            };
            outputs.push(ForwardOutput { logits, act_logits });
        }
        Ok(outputs)
    }
}

fn softmax_cpu(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    for z in &mut exps {
        *z /= sum;
    }
    exps
}

fn top2_margin(p: &[f32]) -> (f32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in p {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    if !second.is_finite() {
        second = 0.0;
    }
    (best, best - second)
}

/// One PyTorch-compatible `TransformerEncoderLayer` (norm_first, batch_first).
struct EncoderLayer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    nhead: usize,
    head_dim: usize,
}

impl EncoderLayer {
    fn load(vb: VarBuilder, d: usize) -> CandleResult<Self> {
        let nhead = (d / 64).max(1);
        let head_dim = d / nhead;
        let in_w = vb.get((3 * d, d), "self_attn.in_proj_weight")?;
        let in_b = vb.get(3 * d, "self_attn.in_proj_bias")?;
        let in_proj = Linear::new(in_w, Some(in_b));
        let out_proj = linear(d, d, vb.pp("self_attn.out_proj"))?;
        let linear1 = linear(d, 4 * d, vb.pp("linear1"))?;
        let linear2 = linear(4 * d, d, vb.pp("linear2"))?;
        let norm1 = layer_norm(d, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(d, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            in_proj,
            out_proj,
            linear1,
            linear2,
            norm1,
            norm2,
            nhead,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, key_padding_mask: &Tensor) -> CandleResult<Tensor> {
        // norm_first: x = x + attn(norm1(x)); x = x + ff(norm2(x))
        let x2 = xs.apply(&self.norm1)?;
        let attn = self.mha(&x2, key_padding_mask)?;
        let xs = (xs + attn)?;
        let x2 = xs.apply(&self.norm2)?;
        let ff = x2.apply(&self.linear1)?.gelu()?.apply(&self.linear2)?;
        xs + ff
    }

    fn mha(&self, xs: &Tensor, key_padding_mask: &Tensor) -> CandleResult<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let qkv = xs.apply(&self.in_proj)?; // [B, L, 3D]
        let qkv = qkv.reshape((b, l, 3, self.nhead, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, B, H, L, Dh]
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let scale = (self.head_dim as f32).powf(-0.5);
        if xs.device().is_metal() && self.head_dim == 64 {
            // Build additive mask [B,H,L,L] from key padding (1 = pad).
            let pad = key_padding_mask
                .to_dtype(q.dtype())?
                .unsqueeze(1)?
                .unsqueeze(2)?; // [B,1,1,L]
            let mask = (&pad * (-1.0e4f64))?
                .broadcast_as((b, self.nhead, l, l))?
                .contiguous()?;
            let out = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?;
            let out = out.transpose(1, 2)?.reshape((b, l, d))?;
            return out.apply(&self.out_proj);
        }

        let q = (q * f64::from(scale))?;
        let mut att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // [B,H,L,L]
        let pad = key_padding_mask
            .to_dtype(att.dtype())?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        let mask = (&pad * (-1e4f64))?;
        att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // [B,H,L,Dh]
        let out = out.transpose(1, 2)?.reshape((b, l, d))?;
        out.apply(&self.out_proj)
    }
}
