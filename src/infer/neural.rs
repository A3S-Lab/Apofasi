//! Neural [`DecisionEngine`] backed by [`DecisionNet`].

use std::path::Path;
#[cfg(any(test, feature = "ort"))]
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use indexmap::IndexMap;

use super::checkpoint::{load_agent_config, load_encoder_config, AgentConfig, CheckpointPaths};
use super::decision_net::{ActOutput, BatchItem, DecisionNet};
use super::device::{resolve_device, weight_dtype, DeviceRequest};
#[cfg(feature = "mlx")]
use super::mlx_decision::MlxDecision;
use super::tokenizer::HfTokenizer;
use crate::decode::{answer_from_logits, choice_from_probs, softmax};
use crate::engine::DecisionEngine;
use crate::error::{Error, Result};
use crate::primitive::DecisionKind;
use crate::schema::{Criteria, Question, State, SystemOneRequest, SystemOneResponse, TokenUsage};
use crate::sequence::{choice_option_costs, pack_question, PackedQuestion};
use crate::wide_choice::{compose_grouped_probs, plan_choice_groups};

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

        let dtype = weight_dtype(&device, &config.amp_dtype);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[paths.weights.as_path()], dtype, &device)
        }
        .map_err(|e| Error::Checkpoint(format!("mmap safetensors: {e}")))?;

        #[cfg(feature = "ort")]
        if device.is_cpu() {
            let onnx = select_ort_encoder(&paths.root);
            if let Some(onnx) = onnx {
                let net = DecisionNet::load_with_ort_encoder(
                    vb,
                    &enc_cfg,
                    config.head_layers,
                    &device,
                    &onnx,
                )?;
                return Ok(Self {
                    net: EngineNet::Candle(net),
                    tokenizer,
                    config,
                    model_id,
                    device,
                });
            }
        }

        let net = DecisionNet::load(vb, &enc_cfg, config.head_layers, &device)?;

        Ok(Self {
            net: EngineNet::Candle(net),
            tokenizer,
            config,
            model_id,
            device,
        })
    }

    /// Compute dtype used for Candle weights on this engine.
    pub fn weight_dtype(&self) -> DType {
        weight_dtype(&self.device, &self.config.amp_dtype)
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

    fn forward_packed(
        &self,
        packed: &[PackedQuestion],
    ) -> Result<Vec<super::decision_net::ForwardOutput>> {
        if packed.is_empty() {
            return Ok(Vec::new());
        }
        let items: Vec<BatchItem<'_>> = packed
            .iter()
            .map(|one| BatchItem {
                input_ids: &one.input_ids,
                markers: &one.markers,
                qtype: one.qtype,
            })
            .collect();
        match &self.net {
            EngineNet::Candle(net) => net.forward_batch(&items, self.tokenizer.specials.pad, false),
            #[cfg(feature = "mlx")]
            EngineNet::Mlx(net) => net.forward_batch(&items, self.tokenizer.specials.pad, false),
        }
    }

    fn question_is_wide(&self, question: &Question) -> Result<bool> {
        let Some(costs) = choice_option_costs(
            &self.tokenizer,
            &self.tokenizer.mask_token,
            question,
            self.config.sequence.option_cap,
        )?
        else {
            return Ok(false);
        };
        Ok(plan_choice_groups(&costs, self.config.sequence.head_max_len).len() > 1)
    }

    fn direct_distribution(
        &self,
        state: &State,
        question_id: &str,
        question: &Question,
    ) -> Result<(Vec<String>, Vec<f32>, u32)> {
        let packed = pack_question(
            &self.tokenizer,
            self.tokenizer.specials,
            &self.tokenizer.mask_token,
            state,
            question_id,
            question,
            self.config.sequence,
        )?;
        let tokens = packed.input_ids.len() as u32;
        let labels = packed.option_labels.clone();
        let profile = std::env::var_os("APOFASI_PROFILE").is_some();
        let t_fwd = std::time::Instant::now();
        let outputs = self.forward_packed(std::slice::from_ref(&packed))?;
        if profile {
            eprintln!(
                "apofasi.profile direct_fwd_ms={:.2} seq={} labels={}",
                t_fwd.elapsed().as_secs_f64() * 1000.0,
                packed.input_ids.len(),
                labels.len()
            );
        }
        let logits = &outputs
            .first()
            .ok_or_else(|| Error::Infer("empty forward".into()))?
            .logits;
        let temp = self
            .config
            .temperatures
            .resolve(question.type_, labels.len());
        Ok((labels, softmax(logits, temp), tokens))
    }

    fn choice_distribution(
        &self,
        state: &State,
        question_id: &str,
        question: &Question,
        depth: usize,
    ) -> Result<(Vec<String>, Vec<f32>, u32)> {
        let labels = match &question.criteria {
            Some(Criteria::Choice(opts)) => opts.keys().cloned().collect::<Vec<_>>(),
            _ => {
                return Err(Error::InvalidQuestion {
                    id: question_id.to_string(),
                    reason: "wide choice requires choice criteria".into(),
                });
            }
        };
        let costs = choice_option_costs(
            &self.tokenizer,
            &self.tokenizer.mask_token,
            question,
            self.config.sequence.option_cap,
        )?
        .ok_or_else(|| Error::InvalidQuestion {
            id: question_id.to_string(),
            reason: "wide choice requires choice criteria".into(),
        })?;
        let groups = plan_choice_groups(&costs, self.config.sequence.head_max_len);
        let splittable = groups.len() > 1 && groups.iter().any(|group| group.len() > 1);
        if depth > 8 || !splittable {
            return self.direct_distribution(state, question_id, question);
        }

        let mut jobs: Vec<Option<PackedQuestion>> = Vec::with_capacity(groups.len());
        let mut tokens = 0u32;
        for group in &groups {
            if group.len() < 2 {
                jobs.push(None);
                continue;
            }
            let subset = subset_choice(question, group)?;
            let packed = pack_question(
                &self.tokenizer,
                self.tokenizer.specials,
                &self.tokenizer.mask_token,
                state,
                question_id,
                &subset,
                self.config.sequence,
            )?;
            tokens = tokens.saturating_add(packed.input_ids.len() as u32);
            jobs.push(Some(packed));
        }
        let packed_jobs: Vec<PackedQuestion> = jobs.iter().filter_map(|job| job.clone()).collect();
        let profile = std::env::var_os("APOFASI_PROFILE").is_some();
        let t_fwd = std::time::Instant::now();
        let outputs = self.forward_packed(&packed_jobs)?;
        if profile {
            let seq = packed_jobs.first().map(|p| p.input_ids.len()).unwrap_or(0);
            eprintln!(
                "apofasi.profile wide_fwd_ms={:.2} batch={} seq={} depth={depth}",
                t_fwd.elapsed().as_secs_f64() * 1000.0,
                packed_jobs.len(),
                seq
            );
        }
        let mut output_at = 0usize;
        let mut chunk_probs = Vec::with_capacity(groups.len());
        let mut winner_keys = Vec::with_capacity(groups.len());
        for (group, job) in groups.iter().zip(&jobs) {
            let probs = if job.is_none() {
                vec![1.0]
            } else {
                let logits = &outputs[output_at].logits;
                output_at += 1;
                if logits.len() != group.len() {
                    return Err(Error::Infer(format!(
                        "question `{question_id}` group returned {} logits for {} options",
                        logits.len(),
                        group.len()
                    )));
                }
                let temp = self
                    .config
                    .temperatures
                    .resolve(DecisionKind::Choice, logits.len());
                softmax(logits, temp)
            };
            let win = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            winner_keys.push(labels[group[win]].clone());
            chunk_probs.push(probs);
        }

        let winner_probs = if winner_keys.len() == 1 {
            vec![1.0]
        } else {
            let winners = subset_choice_keys(question, &winner_keys)?;
            let (_labels, probs, winner_tokens) =
                self.choice_distribution(state, question_id, &winners, depth + 1)?;
            tokens = tokens.saturating_add(winner_tokens);
            probs
        };
        let composed = compose_grouped_probs(labels.len(), &groups, &chunk_probs, &winner_probs)
            .map_err(|err| match err {
                Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                    id: question_id.to_string(),
                    reason,
                },
                other => other,
            })?;
        Ok((labels, composed, tokens))
    }
}

impl DecisionEngine for NeuralEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        if request_has_wide(self, request)? {
            return self.decide_wide(request);
        }
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
                // System One does not generate text, so hosts must not bill a fake decode.
                output_tokens: 0,
            },
        })
    }
}

fn request_has_wide(engine: &NeuralEngine, request: &SystemOneRequest) -> Result<bool> {
    for (id, question) in &request.questions {
        question.validate().map_err(|err| match err {
            Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                id: id.clone(),
                reason,
            },
            other => other,
        })?;
        if engine.question_is_wide(question)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn subset_choice(question: &Question, indices: &[usize]) -> Result<Question> {
    let Some(Criteria::Choice(opts)) = &question.criteria else {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "wide choice requires choice criteria".into(),
        });
    };
    let mut sub = IndexMap::new();
    for &index in indices {
        let Some((key, value)) = opts.get_index(index) else {
            return Err(Error::InvalidQuestion {
                id: String::new(),
                reason: format!("choice option {index} is out of range"),
            });
        };
        sub.insert(key.clone(), value.clone());
    }
    Question::new(
        DecisionKind::Choice,
        question.instructions.clone(),
        Some(Criteria::Choice(sub)),
    )
}

fn subset_choice_keys(question: &Question, keys: &[String]) -> Result<Question> {
    let Some(Criteria::Choice(opts)) = &question.criteria else {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "wide choice requires choice criteria".into(),
        });
    };
    let mut sub = IndexMap::new();
    for key in keys {
        let Some(value) = opts.get(key) else {
            return Err(Error::InvalidQuestion {
                id: String::new(),
                reason: format!("choice option `{key}` is missing"),
            });
        };
        sub.insert(key.clone(), value.clone());
    }
    Question::new(
        DecisionKind::Choice,
        question.instructions.clone(),
        Some(Criteria::Choice(sub)),
    )
}

impl NeuralEngine {
    fn decide_wide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let mut answers = IndexMap::new();
        let mut input_tokens = 0u32;
        for (id, question) in &request.questions {
            if self.question_is_wide(question)? {
                let (labels, probs, tokens) =
                    self.choice_distribution(&request.state, id, question, 0)?;
                input_tokens = input_tokens.saturating_add(tokens);
                let answer = choice_from_probs(&labels, &probs).map_err(|err| match err {
                    Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                        id: id.clone(),
                        reason,
                    },
                    other => other,
                })?;
                answers.insert(id.clone(), answer);
            } else {
                let (answer, tokens) = self.fitted_one(id, &request.state, question)?;
                input_tokens = input_tokens.saturating_add(tokens);
                answers.insert(id.clone(), answer);
            }
        }
        Ok(SystemOneResponse {
            model: self.model_id.clone(),
            answers,
            usage: TokenUsage {
                input_tokens,
                // System One does not generate text, so hosts must not bill a fake decode.
                output_tokens: 0,
            },
        })
    }

    fn fitted_one(
        &self,
        id: &str,
        state: &State,
        question: &Question,
    ) -> Result<(crate::schema::Answer, u32)> {
        let packed = pack_question(
            &self.tokenizer,
            self.tokenizer.specials,
            &self.tokenizer.mask_token,
            state,
            id,
            question,
            self.config.sequence,
        )?;
        let tokens = packed.input_ids.len() as u32;
        let outputs = self.forward_packed(std::slice::from_ref(&packed))?;
        let logits = &outputs
            .first()
            .ok_or_else(|| Error::Infer("empty forward".into()))?
            .logits;
        let answer = answer_from_logits(
            question,
            &packed.option_labels,
            logits,
            &self.config.temperatures,
        )
        .map_err(|err| match err {
            Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                id: id.to_string(),
                reason,
            },
            other => other,
        })?;
        Ok((answer, tokens))
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

/// True only for an explicit opt-in (`1`, `true`, `yes`, `on`).
///
/// Unset, empty, `0`, and `false` stay off. INT8 must not turn on because the
/// variable happens to be present.
#[cfg(any(test, feature = "ort"))]
fn ort_quant_requested(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

/// Choose the ONNX encoder file beside a checkpoint.
///
/// Order when quantization is off: `encoder.opt.onnx`, then `encoder.onnx`.
/// `encoder.int8.onnx` is selected only when `want_quant` is true. A directory
/// that contains only the INT8 file falls back to the Candle encoder so host
/// gates keep FP32 confidence.
#[cfg(any(test, feature = "ort"))]
fn select_ort_encoder_path(root: &Path, want_quant: bool) -> Option<PathBuf> {
    let int8 = root.join("encoder.int8.onnx");
    let opt = root.join("encoder.opt.onnx");
    let fp32 = root.join("encoder.onnx");
    if want_quant && int8.is_file() {
        return Some(int8);
    }
    if opt.is_file() {
        return Some(opt);
    }
    if fp32.is_file() {
        return Some(fp32);
    }
    None
}

/// Prefer graph-optimized FP32 `encoder.opt.onnx`, then `encoder.onnx`.
/// Opt into INT8 with `APOFASI_ORT_QUANT=1` when `encoder.int8.onnx` is present.
#[cfg(feature = "ort")]
fn select_ort_encoder(root: &Path) -> Option<PathBuf> {
    let raw = std::env::var("APOFASI_ORT_QUANT").ok();
    select_ort_encoder_path(root, ort_quant_requested(raw.as_deref()))
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
    #[cfg_attr(not(feature = "mlx"), allow(clippy::infallible_destructuring_match))]
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
        let net = match &engine.net {
            EngineNet::Candle(net) => net,
            #[cfg(feature = "mlx")]
            EngineNet::Mlx(_) => panic!("CPU load should use the Candle backend"),
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

    #[test]
    fn ort_quant_flag_is_explicit() {
        assert!(!ort_quant_requested(None));
        assert!(!ort_quant_requested(Some("")));
        assert!(!ort_quant_requested(Some("0")));
        assert!(!ort_quant_requested(Some("false")));
        assert!(!ort_quant_requested(Some("no")));
        assert!(ort_quant_requested(Some("1")));
        assert!(ort_quant_requested(Some("true")));
        assert!(ort_quant_requested(Some(" yes ")));
        assert!(ort_quant_requested(Some("on")));
    }

    #[test]
    fn ort_encoder_selection_keeps_int8_opt_in() {
        let root = std::env::temp_dir().join(format!(
            "apofasi-ort-select-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let int8 = root.join("encoder.int8.onnx");
        std::fs::write(&int8, b"int8").unwrap();
        assert!(
            select_ort_encoder_path(&root, false).is_none(),
            "INT8 alone must not replace the Candle encoder"
        );
        assert_eq!(
            select_ort_encoder_path(&root, true).as_deref(),
            Some(int8.as_path())
        );

        let fp32 = root.join("encoder.onnx");
        std::fs::write(&fp32, b"fp32").unwrap();
        assert_eq!(
            select_ort_encoder_path(&root, false).as_deref(),
            Some(fp32.as_path())
        );

        let opt = root.join("encoder.opt.onnx");
        std::fs::write(&opt, b"opt").unwrap();
        assert_eq!(
            select_ort_encoder_path(&root, false).as_deref(),
            Some(opt.as_path())
        );
        assert_eq!(
            select_ort_encoder_path(&root, true).as_deref(),
            Some(int8.as_path())
        );

        std::fs::remove_file(&int8).unwrap();
        assert_eq!(
            select_ort_encoder_path(&root, true).as_deref(),
            Some(opt.as_path()),
            "quant opt-in with no INT8 file stays on the FP32 graph"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
