//! Neural [`DecisionEngine`] backed by [`DecisionNet`].

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use indexmap::IndexMap;

use super::checkpoint::{load_agent_config, load_encoder_config, AgentConfig, CheckpointPaths};
use super::decision_net::{ActOutput, BatchItem, DecisionNet};
use super::device::{resolve_device, DeviceRequest};
#[cfg(feature = "mlx")]
use super::mlx_decision::MlxDecision;
use super::tokenizer::HfTokenizer;
use crate::decode::answer_from_logits;
use crate::engine::DecisionEngine;
use crate::error::{Error, Result};
use crate::schema::{SystemOneRequest, SystemOneResponse, TokenUsage};
use crate::sequence::pack_question;

enum EngineNet {
    Candle(DecisionNet),
    #[cfg(feature = "mlx")]
    Mlx(MlxDecision),
}

/// Candle neural System-1 engine.
pub struct NeuralEngine {
    net: EngineNet,
    tokenizer: HfTokenizer,
    config: AgentConfig,
    model_id: String,
    device: Device,
}

impl NeuralEngine {
    /// Load a checkpoint directory with automatic device selection.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(root, DeviceRequest::Auto)
    }

    /// Load a checkpoint with an explicit device preference.
    pub fn load_with(root: impl AsRef<Path>, device_request: DeviceRequest) -> Result<Self> {
        let paths = CheckpointPaths::resolve(root)?;
        let config = load_agent_config(&paths.agent_config)?;
        let enc_cfg = load_encoder_config(&paths.encoder_config)?;
        let tokenizer = HfTokenizer::from_file(&paths.tokenizer)?;
        let device = resolve_device(device_request)?;
        let model_id = format!(
            "apofasi-{}-{}",
            env!("CARGO_PKG_VERSION"),
            sanitize_id(&config.model_name)
        );

        #[cfg(feature = "mlx")]
        if device.is_metal() {
            let net = MlxDecision::load(&paths.weights, &enc_cfg, config.head_layers)?;
            return Ok(Self {
                net: EngineNet::Mlx(net),
                tokenizer,
                config,
                model_id,
                device,
            });
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[paths.weights.as_path()], DType::F32, &device)
        }
        .map_err(|e| Error::Checkpoint(format!("mmap safetensors: {e}")))?;

        let net = DecisionNet::load(vb, &enc_cfg, config.head_layers, &device)?;

        Ok(Self {
            net: EngineNet::Candle(net),
            tokenizer,
            config,
            model_id,
            device,
        })
    }

    /// Device currently hosting weights.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Softmax act / escalate probabilities from raw act logits.
    pub fn act_probs(act_logits: &[f32]) -> Option<ActOutput> {
        if act_logits.len() < 2 {
            return None;
        }
        let max = act_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e0 = (act_logits[0] - max).exp();
        let e1 = (act_logits[1] - max).exp();
        let sum = (e0 + e1).max(1e-12);
        Some(ActOutput {
            act: e0 / sum,
            escalate: e1 / sum,
        })
    }
}

impl DecisionEngine for NeuralEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let profile = std::env::var_os("APOFASI_PROFILE").is_some();
        let t_pack0 = std::time::Instant::now();
        let mut packed = Vec::with_capacity(request.questions.len());
        let mut input_tokens = 0u32;

        for (id, question) in &request.questions {
            question.validate().map_err(|err| match err {
                Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                    id: id.clone(),
                    reason,
                },
                other => other,
            })?;

            let one = pack_question(
                &self.tokenizer,
                self.tokenizer.specials,
                &self.tokenizer.mask_token,
                &request.state,
                id,
                question,
                self.config.sequence,
            )?;
            input_tokens = input_tokens.saturating_add(one.input_ids.len() as u32);
            packed.push(one);
        }
        let pack_ms = t_pack0.elapsed().as_secs_f64() * 1000.0;

        let items: Vec<BatchItem<'_>> = packed
            .iter()
            .map(|one| BatchItem {
                input_ids: &one.input_ids,
                markers: &one.markers,
                qtype: one.qtype,
            })
            .collect();
        let t_fwd0 = std::time::Instant::now();
        // Always one padded encoder launch (matches the reference System-1 path).
        // Metal SDPA makes the batched encoder cheaper than multiple sequential passes.
        let outputs = match &self.net {
            EngineNet::Candle(net) => {
                net.forward_batch(&items, self.tokenizer.specials.pad, false)?
            }
            #[cfg(feature = "mlx")]
            EngineNet::Mlx(net) => net.forward_batch(&items, self.tokenizer.specials.pad, false)?,
        };
        let fwd_ms = t_fwd0.elapsed().as_secs_f64() * 1000.0;
        if profile {
            let lens: Vec<usize> = packed.iter().map(|p| p.input_ids.len()).collect();
            eprintln!("apofasi.profile pack_ms={pack_ms:.2} fwd_ms={fwd_ms:.2} lens={lens:?}");
        }

        let mut answers = IndexMap::new();
        for ((id, question), (one, out)) in request
            .questions
            .iter()
            .zip(packed.iter().zip(outputs.iter()))
        {
            let answer = answer_from_logits(
                question,
                &one.option_labels,
                &out.logits,
                &self.config.temperatures,
            )
            .map_err(|err| match err {
                Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                    id: id.clone(),
                    reason,
                },
                other => other,
            })?;
            answers.insert(id.clone(), answer);
        }

        Ok(SystemOneResponse {
            model: self.model_id.clone(),
            answers,
            usage: TokenUsage {
                input_tokens,
                output_tokens: (request.questions.len() as u32).saturating_mul(8),
            },
        })
    }
}

fn sanitize_id(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if s.is_empty() {
        "neural".into()
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::decision_net::ForwardOutput;
    use crate::primitive::DecisionKind;
    use crate::schema::{Criteria, Question, State};
    use crate::sequence::pack_question;
    use serde_json::json;

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-3 + 1e-3 * x.abs().max(y.abs()))
    }

    #[test]
    fn batched_forward_matches_per_question() {
        let Some(root) = std::env::var_os("APOFASI_CHECKPOINT") else {
            eprintln!("skip: set APOFASI_CHECKPOINT");
            return;
        };
        let engine = NeuralEngine::load_with(root, DeviceRequest::Cpu).expect("load");
        let mut opts: IndexMap<String, Option<serde_json::Value>> = IndexMap::new();
        opts.insert("billing".into(), Some(json!("invoices payments refunds")));
        opts.insert("technical".into(), Some(json!("bugs outages errors")));
        let mut questions: IndexMap<String, Question> = IndexMap::new();
        questions.insert(
            "department".into(),
            Question::new(
                DecisionKind::Choice,
                json!("Which department should handle this request?"),
                Some(Criteria::Choice(opts)),
            )
            .unwrap(),
        );
        questions.insert(
            "refund_requested".into(),
            Question::new(
                DecisionKind::Noul,
                json!("Does the user explicitly request a refund?"),
                None,
            )
            .unwrap(),
        );
        let state = State::Text(
            "Hi, we were billed twice for March. Please refund the duplicate today.".into(),
        );
        let mut packed = Vec::new();
        for (id, question) in &questions {
            packed.push(
                pack_question(
                    &engine.tokenizer,
                    engine.tokenizer.specials,
                    &engine.tokenizer.mask_token,
                    &state,
                    id,
                    question,
                    engine.config.sequence,
                )
                .unwrap(),
            );
        }
        assert_ne!(
            packed[0].input_ids.len(),
            packed[1].input_ids.len(),
            "batch padding is only proven when sequence lengths differ"
        );
        let EngineNet::Candle(net) = &engine.net else {
            panic!("CPU load should use the Candle backend");
        };
        let singles: Vec<ForwardOutput> = packed
            .iter()
            .map(|one| {
                net.forward(&one.input_ids, &one.markers, one.qtype)
                    .unwrap()
            })
            .collect();
        let items: Vec<BatchItem<'_>> = packed
            .iter()
            .map(|one| BatchItem {
                input_ids: &one.input_ids,
                markers: &one.markers,
                qtype: one.qtype,
            })
            .collect();
        let batched = net
            .forward_batch(&items, engine.tokenizer.specials.pad, false)
            .unwrap();
        assert_eq!(singles.len(), batched.len());
        for (single, batch) in singles.iter().zip(batched.iter()) {
            assert!(
                close(&single.logits, &batch.logits),
                "logits single={:?} batch={:?}",
                single.logits,
                batch.logits
            );
        }
    }
}
