//! Decision engines: trait + lean default lexical scorer.
//!
//! Neural backends live behind the `infer` feature ([`crate::infer::NeuralEngine`]).
//! The default [`LexicalEngine`] is pure Rust, dependency-free beyond the crate
//! core, and keeps release artifacts small while exercising the full System One
//! path.

use crate::confidence::TemperatureTable;
use crate::decode::answer_from_logits;
use crate::error::{Error, Result};
use crate::primitive::DecisionKind;
use crate::schema::{instructions_text, Question, SystemOneRequest, SystemOneResponse, TokenUsage};
use crate::sequence::render_options;
use indexmap::IndexMap;

/// Produces typed answers for a System One request.
pub trait DecisionEngine {
    /// Model id reported in [`SystemOneResponse::model`].
    fn model_id(&self) -> &str;

    /// Evaluate every question against the request state.
    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse>;
}

/// Pure-Rust lexical overlap scorer (default, size-minimal).
///
/// Scores each option by token overlap between the state text and the
/// option label/description (plus instruction tokens). Softmax yields a
/// calibrated-looking distribution suitable for integration tests and
/// lightweight host demos — not a frontier neural System-1 model.
#[derive(Debug, Clone)]
pub struct LexicalEngine {
    /// Reported model id.
    pub model_id: String,
    /// Softmax temperatures.
    pub temperatures: TemperatureTable,
}

impl Default for LexicalEngine {
    fn default() -> Self {
        Self {
            model_id: format!("apofasi-lexical-{}", env!("CARGO_PKG_VERSION")),
            temperatures: TemperatureTable::default(),
        }
    }
}

impl DecisionEngine for LexicalEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn decide(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let state_text = request.state.flat_text();
        let state_tokens = tokenize(&state_text);
        let mut answers = IndexMap::new();
        let mut input_tokens = estimate_tokens(&state_text) as u32;

        for (id, question) in &request.questions {
            question.validate().map_err(|err| match err {
                Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                    id: id.clone(),
                    reason,
                },
                other => other,
            })?;
            let options = render_options(question).map_err(|err| match err {
                Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
                    id: id.clone(),
                    reason,
                },
                other => other,
            })?;
            let labels: Vec<String> = options.iter().map(|(k, _)| k.clone()).collect();
            let logits = score_options(&state_tokens, question, &options);
            input_tokens = input_tokens
                .saturating_add(estimate_tokens(&instructions_text(&question.instructions)) as u32);
            for (_, text) in &options {
                input_tokens = input_tokens.saturating_add(estimate_tokens(text) as u32);
            }
            let answer = answer_from_logits(question, &labels, &logits, &self.temperatures)
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

fn score_options(
    state_tokens: &TokenSet,
    question: &Question,
    options: &[(String, String)],
) -> Vec<f32> {
    let ins_tokens = tokenize(&instructions_text(&question.instructions));
    options
        .iter()
        .map(|(label, text)| {
            let mut opt = tokenize(label);
            opt.merge(&tokenize(text));
            let mut score = overlap(state_tokens, &opt) as f32;
            // Instruction tokens that also appear in the state boost the match.
            score += 0.25 * overlap(state_tokens, &ins_tokens) as f32;
            // Tiny length prior so empty options do not dominate.
            score += 0.01 * (opt.len() as f32).sqrt();
            if question.type_ == DecisionKind::Noul && label == "true" {
                // Affirmative cues in the state nudge P(true) up for noul.
                for cue in [
                    "please",
                    "refund",
                    "cancel",
                    "urgent",
                    "fail",
                    "failed",
                    "twice",
                    "duplicate",
                    "threat",
                    "asap",
                ] {
                    if state_tokens.contains(cue) {
                        score += 1.5;
                    }
                }
            }
            if question.type_ == DecisionKind::Noul && label == "false" {
                score += 0.5; // baseline mass on false
            }
            score
        })
        .collect()
}

#[derive(Debug, Default, Clone)]
struct TokenSet {
    counts: IndexMap<String, u32>,
}

impl TokenSet {
    fn len(&self) -> usize {
        self.counts.len()
    }

    fn contains(&self, token: &str) -> bool {
        self.counts.contains_key(token)
    }

    fn merge(&mut self, other: &TokenSet) {
        for (k, v) in &other.counts {
            *self.counts.entry(k.clone()).or_insert(0) += *v;
        }
    }
}

fn tokenize(text: &str) -> TokenSet {
    let mut set = TokenSet::default();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        let t = raw.to_ascii_lowercase();
        if t.len() < 2 {
            continue;
        }
        *set.counts.entry(t).or_insert(0) += 1;
    }
    set
}

fn overlap(a: &TokenSet, b: &TokenSet) -> u32 {
    let mut n = 0u32;
    for (tok, ca) in &a.counts {
        if let Some(cb) = b.counts.get(tok) {
            n += (*ca).min(*cb);
        }
    }
    n
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace()
        .filter(|t| !t.is_empty())
        .count()
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Answer, Criteria, State};
    use serde_json::json;

    #[test]
    fn lexical_routes_refund_to_billing() {
        let mut opts = IndexMap::new();
        opts.insert("billing".into(), Some(json!("invoices payments refunds")));
        opts.insert("technical".into(), Some(json!("bugs outages errors")));
        opts.insert("sales".into(), Some(json!("pricing contracts")));
        let mut questions = IndexMap::new();
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
        let req = SystemOneRequest {
            model: None,
            state: State::Text(
                "Hi, we were billed twice for March. Please refund the duplicate today.".into(),
            ),
            questions,
        };
        let res = LexicalEngine::default().decide(&req).unwrap();
        match &res.answers["department"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "billing"),
            other => panic!("unexpected {other:?}"),
        }
        match &res.answers["refund_requested"] {
            Answer::Noul { noul } => assert!(*noul > 0.5, "noul={noul}"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(res.model.starts_with("apofasi-lexical-"));
        assert!(res.usage.input_tokens > 0);
        assert_eq!(res.usage.output_tokens, 0);
    }
}
