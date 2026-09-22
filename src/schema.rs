//! System One request / response schema aligned with TypeSafe Jev.
//!
//! Wire shape matches `POST /v1/systemone`:
//! `{ "model", "state", "questions" }` → `{ "model", "answers", "usage" }`.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

use crate::error::{Error, Result};
use crate::primitive::DecisionKind;

/// Content under evaluation: string, JSON object, or JSON array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum State {
    /// Plain text.
    Text(String),
    /// Structured JSON object.
    Object(JsonMap<String, Value>),
    /// Array of values (often text turns).
    Array(Vec<Value>),
}

impl State {
    /// Flatten string leaves for language / script detection.
    pub fn flat_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Object(map) => flatten_value(&Value::Object(map.clone()), 0),
            Self::Array(items) => flatten_value(&Value::Array(items.clone()), 0),
        }
    }

    /// Serialize for the model state segment.
    pub fn model_text(&self) -> Result<String> {
        match self {
            Self::Text(s) => Ok(s.clone()),
            Self::Object(map) => Ok(serde_json::to_string(&Value::Object(map.clone()))?),
            Self::Array(items) => Ok(serde_json::to_string(items)?),
        }
    }
}

fn flatten_value(value: &Value, depth: usize) -> String {
    if depth > 6 {
        return String::new();
    }
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|v| flatten_value(v, depth + 1))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        Value::Object(map) => map
            .values()
            .map(|v| flatten_value(v, depth + 1))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Question instructions: usually a string; structured JSON is allowed.
pub type Instructions = Value;

/// Render instructions to text for packing / prompts.
pub fn instructions_text(instructions: &Instructions) -> String {
    match instructions {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render a criterion gloss (string or structured JSON) to prompt text.
pub fn criterion_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Typed criteria payload, validated against [`DecisionKind`].
#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Choice: ordered option key → optional description (`null` / absent = key only).
    Choice(IndexMap<String, Option<Value>>),
    /// Score: ordered level descriptions (2–10 typical; engine may allow more).
    Score(Vec<Value>),
    /// Noul: optional true/false glosses.
    Noul {
        /// Gloss for true.
        true_gloss: Option<Value>,
        /// Gloss for false.
        false_gloss: Option<Value>,
    },
}

/// One typed question (id lives as the key in [`SystemOneRequest::questions`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Question {
    /// Primitive kind.
    #[serde(rename = "type")]
    pub type_: DecisionKind,
    /// Statement or question text (string or structured JSON).
    pub instructions: Instructions,
    /// Type-specific criteria.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Criteria>,
}

impl Question {
    /// Build and validate a question.
    pub fn new(
        type_: DecisionKind,
        instructions: Instructions,
        criteria: Option<Criteria>,
    ) -> Result<Self> {
        let q = Self {
            type_,
            instructions,
            criteria,
        };
        q.validate()?;
        Ok(q)
    }

    /// Validate criteria against the question type.
    pub fn validate(&self) -> Result<()> {
        if instructions_text(&self.instructions).trim().is_empty() {
            return Err(Error::InvalidQuestion {
                id: String::new(),
                reason: "instructions must be non-empty".into(),
            });
        }
        match self.type_ {
            DecisionKind::Choice => match &self.criteria {
                Some(Criteria::Choice(opts)) if !opts.is_empty() => {
                    if opts.len() > 255 {
                        return Err(Error::InvalidQuestion {
                            id: String::new(),
                            reason: "choice criteria accept at most 255 options".into(),
                        });
                    }
                    if opts.keys().any(|key| key.trim().is_empty()) {
                        return Err(Error::InvalidQuestion {
                            id: String::new(),
                            reason: "choice option keys must be non-empty".into(),
                        });
                    }
                    Ok(())
                }
                _ => Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: "choice questions require a non-empty criteria object".into(),
                }),
            },
            DecisionKind::Score => match &self.criteria {
                Some(Criteria::Score(levels)) if levels.len() >= 2 => Ok(()),
                _ => Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: "score questions require a criteria array with at least 2 levels"
                        .into(),
                }),
            },
            DecisionKind::Noul => match &self.criteria {
                None | Some(Criteria::Noul { .. }) => Ok(()),
                Some(_) => Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: "noul criteria must be omitted or an object with true/false glosses"
                        .into(),
                }),
            },
        }
    }
}

impl<'de> Deserialize<'de> for Question {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(rename = "type")]
            type_: DecisionKind,
            instructions: Instructions,
            #[serde(default)]
            criteria: Option<Value>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let criteria = match raw.type_ {
            DecisionKind::Choice => {
                let value = raw
                    .criteria
                    .ok_or_else(|| serde::de::Error::custom("choice questions require criteria"))?;
                let obj = value
                    .as_object()
                    .ok_or_else(|| serde::de::Error::custom("choice criteria must be an object"))?;
                let mut map = IndexMap::new();
                for (k, v) in obj {
                    let gloss = if v.is_null() { None } else { Some(v.clone()) };
                    map.insert(k.clone(), gloss);
                }
                Some(Criteria::Choice(map))
            }
            DecisionKind::Score => {
                let value = raw
                    .criteria
                    .ok_or_else(|| serde::de::Error::custom("score questions require criteria"))?;
                let arr = value
                    .as_array()
                    .ok_or_else(|| serde::de::Error::custom("score criteria must be an array"))?;
                Some(Criteria::Score(arr.clone()))
            }
            DecisionKind::Noul => match raw.criteria {
                None => None,
                Some(Value::Object(obj)) => Some(Criteria::Noul {
                    true_gloss: obj.get("true").cloned().filter(|v| !v.is_null()),
                    false_gloss: obj.get("false").cloned().filter(|v| !v.is_null()),
                }),
                Some(_) => {
                    return Err(serde::de::Error::custom(
                        "noul criteria must be an object with true/false keys",
                    ));
                }
            },
        };
        let q = Question {
            type_: raw.type_,
            instructions: raw.instructions,
            criteria,
        };
        q.validate().map_err(serde::de::Error::custom)?;
        Ok(q)
    }
}

impl Serialize for Criteria {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Choice(map) => {
                let mut obj = JsonMap::new();
                for (k, v) in map {
                    obj.insert(k.clone(), v.clone().unwrap_or(Value::Null));
                }
                Value::Object(obj).serialize(serializer)
            }
            Self::Score(levels) => Value::Array(levels.clone()).serialize(serializer),
            Self::Noul {
                true_gloss,
                false_gloss,
            } => {
                let mut obj = JsonMap::new();
                if let Some(v) = true_gloss {
                    obj.insert("true".into(), v.clone());
                }
                if let Some(v) = false_gloss {
                    obj.insert("false".into(), v.clone());
                }
                Value::Object(obj).serialize(serializer)
            }
        }
    }
}

/// System One request body (Jev-compatible).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// Model id or alias (required on HTTP; set by the runtime for in-process calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// State under evaluation.
    pub state: State,
    /// Question id → question definition.
    pub questions: IndexMap<String, Question>,
}

/// Token accounting (Jev-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Input tokens.
    pub input_tokens: u32,
    /// Output tokens. Decision engines report `0`; this crate does not generate text.
    pub output_tokens: u32,
}

/// One typed answer (Jev-compatible field set).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Discrete choice among labelled options.
    Choice {
        /// Winning option key.
        choice: String,
        /// Entropy-derived confidence in `[0, 1]`.
        confidence: f32,
        /// Full distribution over option keys.
        probabilities: BTreeMap<String, f32>,
    },
    /// Expected ordinal score on the rubric.
    Score {
        /// Probability-weighted mean of level indices.
        score: f32,
        /// Entropy-derived confidence in `[0, 1]`.
        confidence: f32,
        /// Level index → original criterion text.
        legend: BTreeMap<String, Value>,
        /// Distribution over level indices (`"0"`, `"1"`, …).
        probabilities: BTreeMap<String, f32>,
    },
    /// Calibrated P(true). No separate confidence field.
    Noul {
        /// Probability the statement holds.
        noul: f32,
    },
}

/// System One response body (Jev-compatible).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// Exact model version that answered.
    pub model: String,
    /// Question id → answer.
    pub answers: IndexMap<String, Answer>,
    /// Token usage.
    pub usage: TokenUsage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_roundtrips_jev_shape() {
        let raw = json!({
            "model": "apofasi-0.1.1",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this convey urgency?",
                    "criteria": {
                        "true": "Explicitly time-sensitive",
                        "false": "No urgency expressed"
                    }
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts"
                    }
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]
                }
            }
        });
        let req: SystemOneRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.questions.len(), 3);
        assert_eq!(req.questions["department"].type_, DecisionKind::Choice);
        assert_eq!(req.questions["frustration"].type_, DecisionKind::Score);
        assert!(matches!(
            req.questions["is_urgent"].criteria,
            Some(Criteria::Noul { .. })
        ));
    }

    #[test]
    fn response_roundtrips_jev_shape() {
        let raw = json!({
            "model": "apofasi-0.1.1",
            "answers": {
                "is_urgent": { "type": "noul", "noul": 0.95 },
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.8,
                    "probabilities": {
                        "billing": 0.87,
                        "sales": 0.0,
                        "technical": 0.13
                    }
                },
                "frustration": {
                    "type": "score",
                    "score": 1.04,
                    "confidence": 0.94,
                    "legend": {
                        "0": "Calm",
                        "1": "Frustrated",
                        "2": "Very angry"
                    },
                    "probabilities": { "0": 0.0, "1": 0.96, "2": 0.04 }
                }
            },
            "usage": { "input_tokens": 426, "output_tokens": 73 }
        });
        let res: SystemOneResponse = serde_json::from_value(raw.clone()).unwrap();
        assert!(
            matches!(res.answers["is_urgent"], Answer::Noul { noul } if (noul - 0.95).abs() < 1e-5)
        );
        let back = serde_json::to_value(&res).unwrap();
        assert!((back["answers"]["is_urgent"]["noul"].as_f64().unwrap() - 0.95).abs() < 1e-5);
        assert!(back["answers"]["is_urgent"].get("confidence").is_none());
        assert_eq!(back["answers"]["department"]["choice"], "billing");
        assert_eq!(back["usage"]["input_tokens"], 426);
    }

    #[test]
    fn state_accepts_object_and_array() {
        let obj: State = serde_json::from_value(json!({"ticket": {"message": "refund"}})).unwrap();
        assert!(matches!(obj, State::Object(_)));
        let arr: State = serde_json::from_value(json!(["user: hi", "agent: hello"])).unwrap();
        assert!(matches!(arr, State::Array(_)));
    }

    #[test]
    fn blank_instructions_and_option_keys_are_rejected() {
        let err = Question::new(DecisionKind::Noul, json!("   "), None).unwrap_err();
        assert!(err.to_string().contains("non-empty"));

        let mut opts = IndexMap::new();
        opts.insert(" ".into(), None);
        opts.insert("billing".into(), None);
        let err = Question::new(
            DecisionKind::Choice,
            json!("Which team?"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap_err();
        assert!(err.to_string().contains("option keys"));
    }
}
