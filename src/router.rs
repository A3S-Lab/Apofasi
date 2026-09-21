//! Checkpoint router — selection only, no model load.

use crate::detect::{analyse, Detection};
use crate::error::Result;
use crate::route::{CheckpointId, RouteDecision};
use crate::schema::SystemOneRequest;

/// Pure router over detection + explicit overrides.
#[derive(Debug, Clone)]
pub struct Router {
    /// Fallback when detection is inconclusive.
    pub default: CheckpointId,
    /// When true, exact typed-decisions workflow id sets may select that checkpoint.
    pub auto_task_detection: bool,
}

impl Default for Router {
    fn default() -> Self {
        Self {
            default: CheckpointId::English,
            auto_task_detection: false,
        }
    }
}

impl Router {
    /// Decide which checkpoint should answer `request`.
    ///
    /// `model` overrides [`SystemOneRequest::model`] when provided.
    pub fn route(
        &self,
        request: &SystemOneRequest,
        model: Option<&str>,
        lang: Option<&str>,
    ) -> Result<RouteDecision> {
        let explicit = model.or(request.model.as_deref());
        if let Some(name) = explicit {
            // Only treat as a checkpoint override when it parses as one.
            if let Some(id) = CheckpointId::parse(name) {
                return Ok(RouteDecision {
                    model: id,
                    reason: format!("explicit model={name:?}"),
                    detection: None,
                });
            }
            // Versioned runtime model names fall through to detection / default.
        }

        if self.auto_task_detection {
            let ids: Vec<&str> = request.questions.keys().map(|s| s.as_str()).collect();
            if let Some(wf) = match_typed_decisions_workflow(&ids) {
                return Ok(RouteDecision {
                    model: CheckpointId::TypedDecisions,
                    reason: format!("question ids match the {wf:?} typed-decisions workflow"),
                    detection: None,
                });
            }
        }

        if let Some(lang) = lang {
            let key = lang.to_ascii_lowercase();
            let en = matches!(key.as_str(), "en" | "eng" | "english")
                || key.split('-').next() == Some("en");
            let id = if en {
                CheckpointId::English
            } else {
                CheckpointId::Multilingual
            };
            return Ok(RouteDecision {
                model: id,
                reason: format!("explicit lang={lang:?}"),
                detection: None,
            });
        }

        let detection = analyse(&request.state);
        Ok(route_from_detection(detection, self.default))
    }
}

fn route_from_detection(detection: Detection, default: CheckpointId) -> RouteDecision {
    if detection.script == "unknown" {
        return RouteDecision {
            model: default,
            reason: format!(
                "no letters detected in state; using default ({})",
                default.as_str()
            ),
            detection: Some(detection),
        };
    }
    if detection.script != "latin" {
        return RouteDecision {
            model: CheckpointId::Multilingual,
            reason: format!(
                "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                detection.script,
                100.0 * detection.non_latin_fraction
            ),
            detection: Some(detection),
        };
    }
    if !detection.is_english {
        return RouteDecision {
            model: CheckpointId::Multilingual,
            reason: format!(
                "Latin script but language looks like {:?}, not English",
                detection.language
            ),
            detection: Some(detection),
        };
    }
    RouteDecision {
        model: CheckpointId::English,
        reason: "English Latin text".into(),
        detection: Some(detection),
    }
}

fn match_typed_decisions_workflow(ids: &[&str]) -> Option<&'static str> {
    let set: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
    const WORKFLOWS: &[(&str, &[&str])] = &[
        (
            "customer_service",
            &["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            &[
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            &[
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
        (
            "agent_trace_observability",
            &["action", "needs_review", "outcome", "risk", "urgency"],
        ),
    ];
    for (name, sig) in WORKFLOWS {
        let expect: std::collections::BTreeSet<&str> = sig.iter().copied().collect();
        if set == expect {
            return Some(*name);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::DecisionKind;
    use crate::schema::{Criteria, Question, State};
    use indexmap::IndexMap;
    use serde_json::json;

    fn sample_req(state: State) -> SystemOneRequest {
        let mut opts = IndexMap::new();
        opts.insert("billing".into(), None);
        let mut questions = IndexMap::new();
        questions.insert(
            "department".into(),
            Question::new(
                DecisionKind::Choice,
                json!("dept?"),
                Some(Criteria::Choice(opts)),
            )
            .unwrap(),
        );
        SystemOneRequest {
            model: None,
            state,
            questions,
        }
    }

    #[test]
    fn routes_english_to_english() {
        let req = sample_req(State::Text("Please refund the duplicate charge.".into()));
        let decision = Router::default().route(&req, None, None).unwrap();
        assert_eq!(decision.model, CheckpointId::English);
    }

    #[test]
    fn routes_chinese_to_multilingual() {
        let req = sample_req(State::Text("请马上退款，否则取消订阅。".into()));
        let decision = Router::default().route(&req, None, None).unwrap();
        assert_eq!(decision.model, CheckpointId::Multilingual);
    }

    #[test]
    fn explicit_override() {
        let req = sample_req(State::Text("hello".into()));
        let decision = Router::default()
            .route(&req, Some("typed-decisions"), None)
            .unwrap();
        assert_eq!(decision.model, CheckpointId::TypedDecisions);
    }
}
