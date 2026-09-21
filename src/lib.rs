//! A3S Apofasi — typed System-1 decisions for A3S hosts.
//!
//! *Apófasi* (απόφαση) means **decision**. This crate owns fast, non-generative
//! typed decisions (`choice`, `score`, `noul`) that hosts can route, gate, and
//! specialize without parsing free-form model text.
//!
//! The public surface is intentionally small in `0.1.0`; inference backends and
//! host integrations land behind stable types rather than ad-hoc wrappers.

#![deny(missing_docs)]

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Typed decision primitive kinds supported by Apofasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_kind_names_are_stable() {
        assert_eq!(DecisionKind::Choice.as_str(), "choice");
        assert_eq!(DecisionKind::Score.as_str(), "score");
        assert_eq!(DecisionKind::Noul.as_str(), "noul");
    }
}
