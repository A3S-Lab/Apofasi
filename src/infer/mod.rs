//! Candle-backed neural System-1 inference (feature `infer`).
//!
//! Layout matches the published checkpoint contract:
//! `encoder.*` + `type_emb.*` + `head.*` + `scorer.*` + `act_head.*`.

mod checkpoint;
mod decision_net;
mod device;
#[cfg(feature = "mlx")]
mod mlx_decision;
mod modernbert;
mod neural;
#[cfg(feature = "ort")]
mod ort_encoder;
mod registry;
mod tokenizer;

pub use checkpoint::{load_agent_config, AgentConfig, CheckpointPaths};
pub use decision_net::{ActOutput, DecisionNet, ForwardOutput};
pub use device::{resolve_device, weight_dtype, DeviceRequest};
pub use neural::NeuralEngine;
pub use registry::CheckpointRegistry;
pub use tokenizer::HfTokenizer;
