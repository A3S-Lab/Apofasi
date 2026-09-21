//! Checkpoint directory layout and agent config.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::confidence::TemperatureTable;
use crate::error::{Error, Result};
use crate::sequence::SequenceConfig;

/// On-disk paths for one checkpoint tree.
#[derive(Debug, Clone)]
pub struct CheckpointPaths {
    /// Root directory.
    pub root: PathBuf,
    /// `rl_agent_config.json`.
    pub agent_config: PathBuf,
    /// `model.safetensors`.
    pub weights: PathBuf,
    /// `encoder/config.json`.
    pub encoder_config: PathBuf,
    /// `tokenizer/tokenizer.json`.
    pub tokenizer: PathBuf,
}

impl CheckpointPaths {
    /// Resolve and validate the standard checkpoint layout under `root`.
    pub fn resolve(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let paths = Self {
            agent_config: root.join("rl_agent_config.json"),
            weights: root.join("model.safetensors"),
            encoder_config: root.join("encoder").join("config.json"),
            tokenizer: root.join("tokenizer").join("tokenizer.json"),
            root,
        };
        for (label, path) in [
            ("rl_agent_config.json", &paths.agent_config),
            ("model.safetensors", &paths.weights),
            ("encoder/config.json", &paths.encoder_config),
            ("tokenizer/tokenizer.json", &paths.tokenizer),
        ] {
            if !path.is_file() {
                return Err(Error::Checkpoint(format!(
                    "missing {label} at {}",
                    path.display()
                )));
            }
        }
        Ok(paths)
    }

    /// Resolve a named checkpoint inside a bundle directory.
    ///
    /// Bundle layout:
    /// - `english` at `bundle/english/` **or** the bundle root itself
    /// - `multilingual` at `bundle/multilingual/`
    /// - `typed-decisions` at `bundle/typed-decisions/`
    pub fn resolve_named(bundle: impl AsRef<Path>, id: crate::route::CheckpointId) -> Result<Self> {
        let bundle = bundle.as_ref();
        let dir = match id {
            crate::route::CheckpointId::English => {
                let nested = bundle.join("english");
                if nested.join("rl_agent_config.json").is_file() {
                    nested
                } else {
                    bundle.to_path_buf()
                }
            }
            crate::route::CheckpointId::Multilingual => bundle.join("multilingual"),
            crate::route::CheckpointId::TypedDecisions => bundle.join("typed-decisions"),
        };
        Self::resolve(dir)
    }
}

/// Runtime knobs stored next to the weights.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Encoder id string from training (informational).
    pub encoder: String,
    /// Decision-head transformer layers.
    pub head_layers: usize,
    /// Sequence packing budgets.
    pub sequence: SequenceConfig,
    /// Softmax temperatures from the checkpoint.
    pub temperatures: TemperatureTable,
    /// Reported model name.
    pub model_name: String,
}

#[derive(Debug, Deserialize)]
struct RawAgentConfig {
    encoder: String,
    head_layers: usize,
    max_len: usize,
    head_max_len: usize,
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_temps")]
    temperature: [f32; 3],
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f32>,
}

fn default_model_name() -> String {
    "apofasi".into()
}

fn default_temps() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// Load [`AgentConfig`] from `rl_agent_config.json`.
pub fn load_agent_config(path: impl AsRef<Path>) -> Result<AgentConfig> {
    let raw: RawAgentConfig = serde_json::from_str(
        &fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Checkpoint(format!("read agent config: {e}")))?,
    )
    .map_err(|e| Error::Checkpoint(format!("parse agent config: {e}")))?;

    let mut by_options: Vec<(String, f32)> = raw.temperature_by_options.into_iter().collect();
    by_options.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(AgentConfig {
        encoder: raw.encoder,
        head_layers: raw.head_layers,
        sequence: SequenceConfig {
            max_len: raw.max_len,
            head_max_len: raw.head_max_len,
            option_cap: 48,
        },
        temperatures: TemperatureTable {
            by_kind: raw.temperature,
            by_options,
        },
        model_name: raw.model_name,
    })
}

/// Parse ModernBERT geometry from `encoder/config.json` into Candle's config.
pub fn load_encoder_config(path: impl AsRef<Path>) -> Result<super::modernbert::Config> {
    #[derive(Deserialize)]
    struct RopeTheta {
        rope_theta: f64,
    }
    #[derive(Deserialize)]
    struct RopeParams {
        full_attention: Option<RopeTheta>,
        sliding_attention: Option<RopeTheta>,
    }
    #[derive(Deserialize)]
    struct HfConfig {
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        #[serde(default = "default_ln_eps")]
        layer_norm_eps: f64,
        #[serde(default = "default_ln_eps")]
        norm_eps: f64,
        pad_token_id: u32,
        global_attn_every_n_layers: usize,
        local_attention: usize,
        #[serde(default)]
        rope_parameters: Option<RopeParams>,
        #[serde(default)]
        global_rope_theta: Option<f64>,
        #[serde(default)]
        local_rope_theta: Option<f64>,
    }
    fn default_ln_eps() -> f64 {
        1e-5
    }

    let raw: HfConfig = serde_json::from_str(
        &fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Checkpoint(format!("read encoder config: {e}")))?,
    )
    .map_err(|e| Error::Checkpoint(format!("parse encoder config: {e}")))?;

    let global_rope_theta = raw
        .global_rope_theta
        .or_else(|| {
            raw.rope_parameters
                .as_ref()
                .and_then(|r| r.full_attention.as_ref())
                .map(|t| t.rope_theta)
        })
        .unwrap_or(160_000.0);
    let local_rope_theta = raw
        .local_rope_theta
        .or_else(|| {
            raw.rope_parameters
                .as_ref()
                .and_then(|r| r.sliding_attention.as_ref())
                .map(|t| t.rope_theta)
        })
        .unwrap_or(10_000.0);
    let layer_norm_eps = if raw.layer_norm_eps > 0.0 {
        raw.layer_norm_eps
    } else {
        raw.norm_eps
    };

    Ok(super::modernbert::Config {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        intermediate_size: raw.intermediate_size,
        max_position_embeddings: raw.max_position_embeddings,
        layer_norm_eps,
        pad_token_id: raw.pad_token_id,
        global_attn_every_n_layers: raw.global_attn_every_n_layers,
        global_rope_theta,
        local_attention: raw.local_attention,
        local_rope_theta,
        classifier_config: None,
    })
}
