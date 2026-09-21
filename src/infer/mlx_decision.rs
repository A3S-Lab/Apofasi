//! Apple MLX decision forward.
//!
//! Candle Metal SDPA/GEMM on this ModernBERT is slower than the PyTorch MPS
//! reference. The forward in `native/mlx_forward.cc` runs the same safetensors
//! checkpoint on MLX fused kernels. Weights are upcast to f32 so option logits
//! stay aligned with the Candle path.

use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;

use super::decision_net::{BatchItem, ForwardOutput};
use super::modernbert::Config as EncoderConfig;
use crate::error::{Error, Result};

#[repr(C)]
struct ApofasiMlxConfig {
    hidden: i32,
    n_layers: i32,
    n_heads: i32,
    global_every: i32,
    local_attention: i32,
    head_layers: i32,
    global_theta: f32,
    local_theta: f32,
    eps: f32,
}

enum MlxNet {}

extern "C" {
    fn apofasi_mlx_load(
        path: *const c_char,
        cfg: *const ApofasiMlxConfig,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut MlxNet;
    fn apofasi_mlx_free(net: *mut MlxNet);
    fn apofasi_mlx_forward(
        net: *mut MlxNet,
        ids: *const i32,
        attn: *const u8,
        batch: i32,
        seq: i32,
        markers: *const i32,
        n_markers: i32,
        qtypes: *const u8,
        out: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
}

/// Decision net executed on MLX.
pub struct MlxDecision {
    net: *mut MlxNet,
}

// The MLX net is only used from `decide` on one engine. The pointer is owned
// here and freed in `Drop`.
unsafe impl Send for MlxDecision {}

impl Drop for MlxDecision {
    fn drop(&mut self) {
        if !self.net.is_null() {
            unsafe { apofasi_mlx_free(self.net) };
        }
    }
}

impl MlxDecision {
    /// Load `model.safetensors` and keep f32 copies of every tensor this forward uses.
    pub fn load(weights: &Path, enc: &EncoderConfig, head_layers: usize) -> Result<Self> {
        if enc.num_attention_heads == 0 || enc.hidden_size % enc.num_attention_heads != 0 {
            return Err(Error::Checkpoint(format!(
                "hidden {} not divisible by heads {}",
                enc.hidden_size, enc.num_attention_heads
            )));
        }
        let path = CString::new(weights.to_string_lossy().as_bytes()).map_err(|_| {
            Error::Checkpoint(format!(
                "weight path is not a C string: {}",
                weights.display()
            ))
        })?;
        let cfg = ApofasiMlxConfig {
            hidden: i32::try_from(enc.hidden_size)
                .map_err(|_| Error::Checkpoint("hidden".into()))?,
            n_layers: i32::try_from(enc.num_hidden_layers)
                .map_err(|_| Error::Checkpoint("layers".into()))?,
            n_heads: i32::try_from(enc.num_attention_heads)
                .map_err(|_| Error::Checkpoint("heads".into()))?,
            global_every: i32::try_from(enc.global_attn_every_n_layers)
                .map_err(|_| Error::Checkpoint("global attention stride".into()))?,
            local_attention: i32::try_from(enc.local_attention)
                .map_err(|_| Error::Checkpoint("local attention".into()))?,
            head_layers: i32::try_from(head_layers)
                .map_err(|_| Error::Checkpoint("head layers".into()))?,
            global_theta: enc.global_rope_theta as f32,
            local_theta: enc.local_rope_theta as f32,
            eps: enc.layer_norm_eps as f32,
        };
        let mut err = [0u8; 1024];
        let net = unsafe {
            apofasi_mlx_load(
                path.as_ptr(),
                &cfg,
                err.as_mut_ptr().cast::<c_char>(),
                err.len(),
            )
        };
        if net.is_null() {
            return Err(Error::Checkpoint(format!(
                "mlx load {}: {}",
                weights.display(),
                c_message(&err)
            )));
        }
        Ok(Self { net })
    }

    /// One encoder forward for the packed questions. `pad_id` fills shorter rows.
    pub fn forward_batch(
        &self,
        items: &[BatchItem<'_>],
        pad_id: u32,
        _compute_act: bool,
    ) -> Result<Vec<ForwardOutput>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let batch = items.len();
        let max_len = items
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            return Err(Error::Infer("empty packed sequence".into()));
        }

        let mut ids = vec![i32::try_from(pad_id).unwrap_or(0); batch * max_len];
        let mut attn = vec![0u8; batch * max_len];
        for (row, item) in items.iter().enumerate() {
            if item.input_ids.is_empty() {
                return Err(Error::Infer("empty packed sequence".into()));
            }
            let base = row * max_len;
            for (i, id) in item.input_ids.iter().enumerate() {
                ids[base + i] = i32::try_from(*id).unwrap_or(0);
                attn[base + i] = 1;
            }
        }

        let mut markers = Vec::new();
        let mut counts = Vec::with_capacity(batch);
        let mut qtypes = Vec::with_capacity(batch);
        for (row, item) in items.iter().enumerate() {
            counts.push(item.markers.len());
            qtypes.push(item.qtype);
            for &marker in item.markers {
                if marker >= item.input_ids.len() {
                    return Err(Error::Infer(format!(
                        "marker {marker} out of range for seq_len {}",
                        item.input_ids.len()
                    )));
                }
                let flat = row * max_len + marker;
                markers.push(i32::try_from(flat).unwrap_or(0));
            }
        }
        if markers.is_empty() {
            return Err(Error::Infer("no markers in batch".into()));
        }

        let b = i32::try_from(batch).map_err(|_| Error::Infer("batch does not fit i32".into()))?;
        let seq =
            i32::try_from(max_len).map_err(|_| Error::Infer("length does not fit i32".into()))?;
        let n = i32::try_from(markers.len())
            .map_err(|_| Error::Infer("marker count does not fit i32".into()))?;
        let mut logits = vec![0f32; markers.len()];
        let mut err = [0u8; 1024];
        let rc = unsafe {
            apofasi_mlx_forward(
                self.net,
                ids.as_ptr(),
                attn.as_ptr(),
                b,
                seq,
                markers.as_ptr(),
                n,
                qtypes.as_ptr(),
                logits.as_mut_ptr(),
                err.as_mut_ptr().cast::<c_char>(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Infer(c_message(&err)));
        }

        let mut outputs = Vec::with_capacity(batch);
        let mut offset = 0usize;
        for count in counts {
            let end = offset + count;
            if end > logits.len() {
                return Err(Error::Infer(
                    "scorer returned fewer logits than markers".into(),
                ));
            }
            outputs.push(ForwardOutput {
                logits: logits[offset..end].to_vec(),
                act_logits: Vec::new(),
            });
            offset = end;
        }
        Ok(outputs)
    }
}

fn c_message(buf: &[u8]) -> String {
    let end = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
