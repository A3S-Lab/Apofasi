//! A3S Apofasi — typed System-1 decisions for A3S hosts.
//!
//! *Apófasi* (απόφαση) means **decision**. This crate owns fast, non-generative
//! typed decisions (`choice`, `score`, `noul`) that hosts can route, gate, and
//! specialize without parsing free-form model text.
//!
//! The public request / response types match the TypeSafe System One (Jev)
//! JSON contract: `state` + `questions` map in, `model` + `answers` map +
//! `usage` out.
//!
//! # Quick start
//!
//! ```
//! use a3s_apofasi::{Client, Criteria, DecisionKind, Question, State, SystemOneRequest};
//! use indexmap::IndexMap;
//! use serde_json::json;
//!
//! let mut opts = IndexMap::new();
//! opts.insert("billing".into(), Some(json!("refunds")));
//! opts.insert("other".into(), Some(json!("else")));
//! let mut questions = IndexMap::new();
//! questions.insert(
//!     "department".into(),
//!     Question::new(
//!         DecisionKind::Choice,
//!         json!("Which department?"),
//!         Some(Criteria::Choice(opts)),
//!     )
//!     .unwrap(),
//! );
//! let res = Client::default()
//!     .system_one(SystemOneRequest {
//!         model: None,
//!         state: State::Text("Please refund my invoice.".into()),
//!         questions,
//!     })
//!     .unwrap();
//! assert!(res.answers.contains_key("department"));
//! ```
//!
//! Default builds stay small: the lexical engine has no ML framework dependency.
//! Enable `infer` (and optionally `metal` / `cuda`) for Candle neural System-1.
//!
//! See [`ARCHITECTURE.md`](../ARCHITECTURE.md).

#![deny(missing_docs)]

pub mod client;
pub mod confidence;
pub mod decode;
pub mod detect;
pub mod engine;
pub mod error;
pub mod gate;
pub mod primitive;
pub mod reward;
pub mod route;
pub mod schema;
pub mod sequence;

#[cfg(feature = "router")]
pub mod router;

#[cfg(feature = "infer")]
pub mod infer;

#[cfg(feature = "train")]
pub mod train;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use client::Client;
pub use confidence::{confidence_from_probs, temp_bucket, TemperatureTable};
pub use decode::{answer_from_logits, softmax};
pub use detect::{analyse, Detection};
pub use engine::{DecisionEngine, LexicalEngine};
pub use error::{Error, Result};
pub use gate::{any_escalate, gate_answer, gate_response, GateAction, GatePolicy};
pub use primitive::DecisionKind;
pub use reward::{ece_score, proper_reward, RewardWeights};
pub use route::{CheckpointId, RouteDecision};
pub use schema::{
    criterion_text, instructions_text, Answer, Criteria, Instructions, Question, State,
    SystemOneRequest, SystemOneResponse, TokenUsage,
};
pub use sequence::{
    pack_question, ByteTokenizer, PackedQuestion, SequenceConfig, SpecialTokens, Tokenize,
};

#[cfg(feature = "router")]
pub use router::Router;

#[cfg(feature = "infer")]
pub use infer::{
    resolve_device, ActOutput, AgentConfig, CheckpointPaths, CheckpointRegistry, DeviceRequest,
    ForwardOutput, HfTokenizer, NeuralEngine,
};

#[cfg(feature = "train")]
pub use train::{
    evaluate_finetune_step, mean_batch_reward, FineTuneConfig, FineTuneStepStats, PublishLayout,
    RewardExample,
};
