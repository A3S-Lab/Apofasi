//! Host gating helpers: decide `auto` vs `escalate` from typed answers.
//!
//! Apofasi owns calibrated signals; the host owns when to require human
//! review. These helpers translate confidence / noul extremity into a
//! stable gate label without embedding host policy into the model path.

use std::collections::BTreeMap;

use crate::schema::{Answer, SystemOneResponse};

/// Recommended host action for one answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Proceed without human review.
    Auto,
    /// Escalate to a human / higher-cost path.
    Escalate,
}

impl GateAction {
    /// Stable wire label (`"auto"` / `"escalate"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Escalate => "escalate",
        }
    }
}

/// Threshold policy for [`gate_answer`] / [`gate_response`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatePolicy {
    /// Minimum confidence for Choice/Score to stay on [`GateAction::Auto`].
    pub min_confidence: f32,
    /// Minimum extremity `max(noul, 1 - noul)` for Noul to stay Auto.
    pub min_noul_extremity: f32,
}

impl Default for GatePolicy {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_noul_extremity: 0.7,
        }
    }
}

/// Gate one typed answer.
pub fn gate_answer(answer: &Answer, policy: &GatePolicy) -> GateAction {
    match answer {
        Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => {
            if *confidence >= policy.min_confidence {
                GateAction::Auto
            } else {
                GateAction::Escalate
            }
        }
        Answer::Noul { noul } => {
            let extremity = (*noul).max(1.0 - *noul);
            if extremity >= policy.min_noul_extremity {
                GateAction::Auto
            } else {
                GateAction::Escalate
            }
        }
    }
}

/// Gate every answer in a response (question id → action).
pub fn gate_response(
    response: &SystemOneResponse,
    policy: &GatePolicy,
) -> BTreeMap<String, GateAction> {
    response
        .answers
        .iter()
        .map(|(id, answer)| (id.clone(), gate_answer(answer, policy)))
        .collect()
}

/// True when any answer in the map is [`GateAction::Escalate`].
pub fn any_escalate(gates: &BTreeMap<String, GateAction>) -> bool {
    gates.values().any(|g| *g == GateAction::Escalate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Answer, SystemOneResponse, TokenUsage};
    use indexmap::IndexMap;

    #[test]
    fn peaked_choice_is_auto() {
        let answer = Answer::Choice {
            choice: "billing".into(),
            confidence: 0.85,
            probabilities: Default::default(),
        };
        assert_eq!(
            gate_answer(&answer, &GatePolicy::default()),
            GateAction::Auto
        );
    }

    #[test]
    fn flat_choice_escalates() {
        let answer = Answer::Choice {
            choice: "other".into(),
            confidence: 0.2,
            probabilities: Default::default(),
        };
        assert_eq!(
            gate_answer(&answer, &GatePolicy::default()),
            GateAction::Escalate
        );
    }

    #[test]
    fn extreme_noul_is_auto() {
        let answer = Answer::Noul { noul: 0.95 };
        assert_eq!(
            gate_answer(&answer, &GatePolicy::default()),
            GateAction::Auto
        );
        let answer = Answer::Noul { noul: 0.05 };
        assert_eq!(
            gate_answer(&answer, &GatePolicy::default()),
            GateAction::Auto
        );
    }

    #[test]
    fn mid_noul_escalates() {
        let answer = Answer::Noul { noul: 0.5 };
        assert_eq!(
            gate_answer(&answer, &GatePolicy::default()),
            GateAction::Escalate
        );
    }

    #[test]
    fn response_gate_map_covers_all_ids() {
        let mut answers = IndexMap::new();
        answers.insert(
            "department".into(),
            Answer::Choice {
                choice: "sales".into(),
                confidence: 0.4,
                probabilities: Default::default(),
            },
        );
        answers.insert("refund".into(), Answer::Noul { noul: 0.9 });
        let response = SystemOneResponse {
            model: "test".into(),
            answers,
            usage: TokenUsage::default(),
        };
        let gates = gate_response(&response, &GatePolicy::default());
        assert_eq!(gates["department"], GateAction::Escalate);
        assert_eq!(gates["refund"], GateAction::Auto);
        assert!(any_escalate(&gates));
        assert_eq!(GateAction::Auto.as_str(), "auto");
        assert_eq!(GateAction::Escalate.as_str(), "escalate");
    }
}
