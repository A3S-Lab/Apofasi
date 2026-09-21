//! Routing metadata types (always available; selection logic is feature-gated).

use serde::{Deserialize, Serialize};

use crate::detect::Detection;

/// Named checkpoint identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckpointId {
    /// English ModernBERT-large System-1.
    English,
    /// Multilingual mmBERT-base.
    Multilingual,
    /// Fine-tuned typed-decisions pack.
    TypedDecisions,
}

impl CheckpointId {
    /// Stable wire name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }

    /// Parse name or common alias.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "english" | "en" | "default" => Some(Self::English),
            "multilingual" | "multi" | "ml" => Some(Self::Multilingual),
            "typed-decisions" | "typed" | "typed_decisions" | "decisions" => {
                Some(Self::TypedDecisions)
            }
            _ => None,
        }
    }
}

/// Outcome of checkpoint selection (no model load).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    /// Selected checkpoint.
    pub model: CheckpointId,
    /// Human-readable reason.
    pub reason: String,
    /// Optional detection evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detection: Option<Detection>,
}
