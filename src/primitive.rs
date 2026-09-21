//! Decision primitive kinds.

use serde::{Deserialize, Serialize};

/// Typed decision primitive kinds supported by Apofasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionKind {
    /// Discrete label selection among named options.
    Choice,
    /// Ordinal score over an ordered rubric.
    Score,
    /// Calibrated probability that a proposition is true.
    Noul,
}

impl DecisionKind {
    /// Stable wire / config name for this primitive.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// Dense type id used by the decision head embedding.
    pub const fn type_id(self) -> u8 {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    /// Parse an Apofasi wire name.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_kind_names_are_stable() {
        assert_eq!(DecisionKind::Choice.as_str(), "choice");
        assert_eq!(DecisionKind::Score.as_str(), "score");
        assert_eq!(DecisionKind::Noul.as_str(), "noul");
        assert_eq!(DecisionKind::Choice.type_id(), 0);
        assert_eq!(DecisionKind::parse("NOUL"), Some(DecisionKind::Noul));
    }
}
