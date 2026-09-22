//! Confidence and temperature calibration helpers.

use crate::primitive::DecisionKind;

/// Normalized Shannon entropy confidence: `1 - H(p) / log(k)`.
///
/// Returns 0 when `probs` is not a probability distribution. Clamping zeros
/// inside the entropy formula used to turn an all-zero vector into confidence
/// near 1, which a host would treat as safe to automate.
pub fn confidence_from_probs(probs: &[f32]) -> f32 {
    if !is_distribution(probs) {
        return 0.0;
    }
    let k = probs.len();
    if k < 2 {
        return 1.0;
    }
    let mut ent = 0.0f32;
    for &p in probs {
        let p = p.clamp(1e-12, 1.0);
        ent -= p * p.ln();
    }
    let norm = ent / (k as f32).ln();
    (1.0 - norm).clamp(0.0, 1.0)
}

fn is_distribution(probs: &[f32]) -> bool {
    if probs.is_empty() {
        return false;
    }
    let mut sum = 0.0f32;
    for &p in probs {
        if !p.is_finite() || !(0.0..=1.0 + 1e-3).contains(&p) {
            return false;
        }
        sum += p;
    }
    (sum - 1.0).abs() <= 1e-3
}

/// Temperature lookup bucket key (`"{kind}:{size}"`).
pub fn temp_bucket(kind: DecisionKind, option_count: usize) -> String {
    let size = match option_count {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", kind.as_str())
}

/// Per-kind default temperatures plus optional option-count overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureTable {
    /// Indexed by [`DecisionKind::type_id`].
    pub by_kind: [f32; 3],
    /// Sparse overrides keyed by [`temp_bucket`].
    pub by_options: Vec<(String, f32)>,
}

impl Default for TemperatureTable {
    fn default() -> Self {
        Self {
            by_kind: [1.0, 1.0, 1.0],
            by_options: Vec::new(),
        }
    }
}

impl TemperatureTable {
    /// Resolve temperature for a question.
    ///
    /// The configured value is returned unchanged. Callers reject non-finite
    /// and non-positive temperatures instead of substituting a tiny floor.
    pub fn resolve(&self, kind: DecisionKind, option_count: usize) -> f32 {
        let key = temp_bucket(kind, option_count);
        self.by_options
            .iter()
            .find(|(k, _)| k == &key)
            .map(|(_, v)| *v)
            .unwrap_or(self.by_kind[kind.type_id() as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaked_distribution_is_high_confidence() {
        let c = confidence_from_probs(&[0.95, 0.05]);
        assert!(c > 0.7, "conf={c}");
    }

    #[test]
    fn uniform_distribution_is_low_confidence() {
        let c = confidence_from_probs(&[0.25, 0.25, 0.25, 0.25]);
        assert!(c < 0.05, "conf={c}");
    }

    #[test]
    fn non_distribution_is_zero_confidence() {
        assert_eq!(confidence_from_probs(&[]), 0.0);
        assert_eq!(confidence_from_probs(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(confidence_from_probs(&[0.9, 0.9]), 0.0);
        assert_eq!(confidence_from_probs(&[f32::NAN, 0.0]), 0.0);
        assert_eq!(confidence_from_probs(&[-0.1, 1.1]), 0.0);
        assert_eq!(confidence_from_probs(&[1.0]), 1.0);
    }

    #[test]
    fn temp_bucket_uses_stable_wire_keys() {
        assert_eq!(temp_bucket(DecisionKind::Choice, 4), "choice:3-5");
        assert_eq!(temp_bucket(DecisionKind::Noul, 2), "noul:2");
        assert_eq!(temp_bucket(DecisionKind::Choice, 20), "choice:11+");
    }

    #[test]
    fn resolve_keeps_non_positive_temperature() {
        let mut table = TemperatureTable::default();
        table.by_kind[DecisionKind::Choice.type_id() as usize] = 0.0;
        assert_eq!(table.resolve(DecisionKind::Choice, 4), 0.0);
        table.by_options.push(("choice:3-5".into(), -1.0));
        assert_eq!(table.resolve(DecisionKind::Choice, 4), -1.0);
    }
}
