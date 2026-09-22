//! Decode option logits into Jev-compatible [`Answer`] values.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::confidence::{confidence_from_probs, TemperatureTable};
use crate::error::{Error, Result};
use crate::primitive::DecisionKind;
use crate::schema::{criterion_text, Answer, Criteria, Question};

/// Temperature-scaled softmax over logits.
///
/// Temperature must be finite and strictly positive. A non-positive value is
/// not clamped to a tiny floor: that floor turns every answer into a near
/// one-hot and a host would treat it as safe to automate.
pub fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "softmax temperature must be finite and positive".into(),
        });
    }
    if logits.iter().any(|z| !z.is_finite()) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "softmax logits must be finite".into(),
        });
    }
    // Center before scaling. `choice:11+` uses a temperature near 0.1, and a
    // large negative mask divided by that temperature overflows f32. The
    // largest centered logit is 0, so exp stays in (0, 1].
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(logits.len());
    for &z in logits {
        let scaled = (z - max) / temperature;
        let weight = if scaled.is_finite() {
            scaled.exp()
        } else {
            0.0
        };
        weights.push(weight);
    }
    let sum: f32 = weights.iter().sum::<f32>().max(1e-12);
    for weight in &mut weights {
        *weight /= sum;
    }
    if weights.iter().any(|p| !p.is_finite()) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "softmax did not produce finite probabilities".into(),
        });
    }
    Ok(weights)
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
    ensure_finite(logits, "logits")?;
    let temp = temperatures.resolve(question.type_, option_labels.len());
    let probs = softmax(logits, temp)?;
    match question.type_ {
        DecisionKind::Choice => {
            let (best_i, probabilities) = published_choice_map(option_labels, &probs)?;
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
            let winner = argmax(&probs);
            let published = publish_probs(&probs, winner);
            let mut score = 0.0f32;
            for (i, p) in probs.iter().enumerate() {
                let key = i.to_string();
                probabilities.insert(key.clone(), published[i]);
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

fn argmax(probs: &[f32]) -> usize {
    // Ties keep the earlier label. `max_by` would keep the last one and
    // change which option a flat distribution names.
    let mut best_index = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (index, prob) in probs.iter().enumerate() {
        if *prob > best {
            best = *prob;
            best_index = index;
        }
    }
    best_index
}

fn is_unit_distribution(probs: &[f32]) -> bool {
    if probs.is_empty() {
        return false;
    }
    let mut sum = 0.0f32;
    for &prob in probs {
        if !prob.is_finite() || !(0.0..=1.0 + 1e-3).contains(&prob) {
            return false;
        }
        sum += prob;
    }
    (sum - 1.0).abs() <= 1e-3
}

/// Publish `probs` at 0.0001 resolution.
///
/// Rounding each entry on its own drifts off 1 as the option count grows, and
/// can make a non-winner look larger than the chosen label. Units of 0.0001
/// keep a real distribution summing to 1, and keep `winner` as an argmax.
fn publish_probs(probs: &[f32], winner: usize) -> Vec<f32> {
    if !is_unit_distribution(probs) || winner >= probs.len() {
        return probs.iter().copied().map(round4).collect();
    }
    let mut units = vec![0i64; probs.len()];
    for (index, prob) in probs.iter().enumerate() {
        if index == winner {
            continue;
        }
        let scaled = (f64::from(*prob) * 10_000.0).round();
        units[index] = scaled.clamp(0.0, 10_000.0) as i64;
    }
    let min_winner = if probs[winner] > 0.0 { 1 } else { 0 };
    let mut others: i64 = units.iter().sum();
    while others > 10_000 - min_winner {
        let donor = units
            .iter()
            .enumerate()
            .filter(|(index, unit)| *index != winner && **unit > 0)
            .max_by_key(|(_, unit)| *unit)
            .map(|(index, _)| index);
        let Some(donor) = donor else {
            break;
        };
        units[donor] -= 1;
        others -= 1;
    }
    units[winner] = (10_000 - others).clamp(0, 10_000);
    loop {
        let rival = units
            .iter()
            .enumerate()
            .filter(|(index, unit)| *index != winner && **unit > units[winner])
            .max_by_key(|(_, unit)| *unit)
            .map(|(index, _)| index);
        let Some(rival) = rival else {
            break;
        };
        units[rival] -= 1;
        units[winner] += 1;
    }
    units.iter().map(|unit| *unit as f32 / 10_000.0).collect()
}

fn published_choice_map(
    option_labels: &[String],
    probs: &[f32],
) -> Result<(usize, BTreeMap<String, f32>)> {
    if option_labels
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != option_labels.len()
    {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "choice labels must be unique".into(),
        });
    }
    let winner = argmax(probs);
    let published = publish_probs(probs, winner);
    let mut probabilities = BTreeMap::new();
    for (label, prob) in option_labels.iter().zip(published) {
        probabilities.insert(label.clone(), prob);
    }
    Ok((winner, probabilities))
}

fn ensure_finite(values: &[f32], what: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: format!("{what} must be finite"),
        });
    }
    Ok(())
}

#[cfg(feature = "infer")]
pub(crate) fn choice_from_probs(option_labels: &[String], probs: &[f32]) -> Result<Answer> {
    if option_labels.len() != probs.len() || option_labels.is_empty() {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: format!(
                "probability/label length mismatch ({} vs {})",
                option_labels.len(),
                probs.len()
            ),
        });
    }
    ensure_finite(probs, "probabilities")?;
    let (best_i, probabilities) = published_choice_map(option_labels, probs)?;
    Ok(Answer::Choice {
        choice: option_labels[best_i].clone(),
        confidence: round4(confidence_from_probs(probs)),
        probabilities,
    })
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

    #[test]
    fn non_finite_logits_are_rejected() {
        let q = Question::new(DecisionKind::Noul, json!("true?"), None).unwrap();
        for logits in [
            [f32::NAN, 1.0],
            [0.0, f32::INFINITY],
            [f32::NEG_INFINITY, 0.0],
        ] {
            let err = answer_from_logits(
                &q,
                &["false".into(), "true".into()],
                &logits,
                &TemperatureTable::default(),
            )
            .expect_err("non-finite logits must not become an answer");
            let message = err.to_string();
            assert!(
                message.contains("finite"),
                "error={message} logits={logits:?}"
            );
        }
    }

    #[test]
    fn wide_choice_probabilities_sum_to_one_and_keep_the_winner() {
        let mut opts = IndexMap::new();
        let mut labels = Vec::new();
        let mut logits = Vec::new();
        for index in 0..80 {
            let key = format!("opt{index:02}");
            opts.insert(key.clone(), None);
            labels.push(key);
            logits.push(if index == 79 { 1.0 } else { 0.0 });
        }
        let question = Question::new(
            DecisionKind::Choice,
            json!("pick"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap();
        let answer =
            answer_from_logits(&question, &labels, &logits, &TemperatureTable::default()).unwrap();
        let Answer::Choice {
            choice,
            probabilities,
            ..
        } = answer
        else {
            panic!("expected choice");
        };
        assert_eq!(choice, "opt79");
        let sum: f32 = probabilities.values().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        let best = probabilities.values().copied().fold(0.0, f32::max);
        assert_eq!(probabilities["opt79"], best);
    }

    #[test]
    fn wide_temperature_keeps_a_finite_distribution() {
        let mut logits = vec![0.2f32; 12];
        logits[4] = 3.0;
        logits[2] = -1.0e38;
        logits[7] = -1.0e38;
        let probs = softmax(&logits, 0.10058281).expect("centered softmax");
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
        let winner = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index);
        assert_eq!(winner, Some(4));
        assert_eq!(probs[2], 0.0);
    }

    #[test]
    fn non_positive_temperature_is_rejected() {
        let question = Question::new(DecisionKind::Noul, json!("true?"), None).unwrap();
        let mut temperatures = TemperatureTable::default();
        temperatures.by_kind[DecisionKind::Noul.type_id() as usize] = 0.0;
        let err = answer_from_logits(
            &question,
            &["false".into(), "true".into()],
            &[0.0, 1.0],
            &temperatures,
        )
        .expect_err("zero temperature must not become a peaked answer");
        assert!(err.to_string().contains("temperature"));
        temperatures.by_kind[DecisionKind::Noul.type_id() as usize] = f32::NAN;
        let err = answer_from_logits(
            &question,
            &["false".into(), "true".into()],
            &[0.0, 1.0],
            &temperatures,
        )
        .expect_err("NaN temperature must not become an answer");
        assert!(err.to_string().contains("temperature"));
    }
}
