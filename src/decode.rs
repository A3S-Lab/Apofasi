//! Decode option logits into Jev-compatible [`Answer`] values.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::confidence::{confidence_from_probs, TemperatureTable};
use crate::error::{Error, Result};
use crate::primitive::DecisionKind;
use crate::schema::{criterion_text, Answer, Criteria, Question};

/// Temperature-scaled softmax over logits.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-3);
    let mut scaled: Vec<f32> = logits.iter().map(|z| z / t).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for z in &mut scaled {
        *z = (*z - max).exp();
    }
    let sum: f32 = scaled.iter().sum::<f32>().max(1e-12);
    for z in &mut scaled {
        *z /= sum;
    }
    scaled
}

/// Build a typed answer from option labels and raw logits.
pub fn answer_from_logits(
    question: &Question,
    option_labels: &[String],
    logits: &[f32],
    temperatures: &TemperatureTable,
) -> Result<Answer> {
    if option_labels.len() != logits.len() || option_labels.is_empty() {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: format!(
                "logit/label length mismatch ({} vs {})",
                option_labels.len(),
                logits.len()
            ),
        });
    }
    let temp = temperatures.resolve(question.type_, option_labels.len());
    let probs = softmax(logits, temp);
    match question.type_ {
        DecisionKind::Choice => {
            let mut probabilities = BTreeMap::new();
            let mut best_i = 0usize;
            let mut best_p = -1.0f32;
            for (i, (label, p)) in option_labels.iter().zip(probs.iter()).enumerate() {
                probabilities.insert(label.clone(), round4(*p));
                if *p > best_p {
                    best_p = *p;
                    best_i = i;
                }
            }
            Ok(Answer::Choice {
                choice: option_labels[best_i].clone(),
                confidence: round4(confidence_from_probs(&probs)),
                probabilities,
            })
        }
        DecisionKind::Score => {
            let mut probabilities = BTreeMap::new();
            let mut legend = BTreeMap::new();
            let levels = match &question.criteria {
                Some(Criteria::Score(levels)) => levels.clone(),
                _ => {
                    return Err(Error::InvalidQuestion {
                        id: String::new(),
                        reason: "score criteria missing during decode".into(),
                    });
                }
            };
            let mut score = 0.0f32;
            for (i, p) in probs.iter().enumerate() {
                let key = i.to_string();
                probabilities.insert(key.clone(), round4(*p));
                let gloss = levels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| Value::String(format!("level {i}")));
                legend.insert(key, gloss);
                score += (i as f32) * *p;
            }
            Ok(Answer::Score {
                score: round4(score),
                confidence: round4(confidence_from_probs(&probs)),
                legend,
                probabilities,
            })
        }
        DecisionKind::Noul => {
            // Labels are always [false, true] in packer order.
            let true_idx = option_labels
                .iter()
                .position(|l| l == "true")
                .unwrap_or(1.min(probs.len().saturating_sub(1)));
            Ok(Answer::Noul {
                noul: round4(probs[true_idx]),
            })
        }
    }
}

fn round4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

/// Render score legend entry text (for tests / debugging).
pub fn score_legend_text(value: &Value) -> String {
    criterion_text(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Question;
    use indexmap::IndexMap;
    use serde_json::json;

    #[test]
    fn choice_picks_argmax() {
        let mut opts = IndexMap::new();
        opts.insert("a".into(), None);
        opts.insert("b".into(), None);
        let q = Question::new(
            DecisionKind::Choice,
            json!("pick"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap();
        let ans = answer_from_logits(
            &q,
            &["a".into(), "b".into()],
            &[1.0, 3.0],
            &TemperatureTable::default(),
        )
        .unwrap();
        match ans {
            Answer::Choice {
                choice,
                probabilities,
                ..
            } => {
                assert_eq!(choice, "b");
                assert!(probabilities["b"] > probabilities["a"]);
            }
            _ => panic!("expected choice"),
        }
    }

    #[test]
    fn noul_has_no_confidence_field_shape() {
        let q = Question::new(DecisionKind::Noul, json!("true?"), None).unwrap();
        let ans = answer_from_logits(
            &q,
            &["false".into(), "true".into()],
            &[0.0, 2.0],
            &TemperatureTable::default(),
        )
        .unwrap();
        let v = serde_json::to_value(&ans).unwrap();
        assert!(v.get("confidence").is_none());
        assert!(v["noul"].as_f64().unwrap() > 0.7);
    }
}
