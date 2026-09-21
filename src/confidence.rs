//! Confidence and temperature calibration helpers.

use crate::primitive::DecisionKind;

/// Normalized Shannon entropy confidence: `1 - H(p) / log(k)`.
pub fn confidence_from_probs(probs: &[f32]) -> f32 {
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
    pub fn resolve(&self, kind: DecisionKind, option_count: usize) -> f32 {
        let key = temp_bucket(kind, option_count);
        self.by_options
            .iter()
            .find(|(k, _)| k == &key)
            .map(|(_, v)| *v)
            .unwrap_or(self.by_kind[kind.type_id() as usize])
            .max(1e-3)
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
    fn temp_bucket_uses_stable_wire_keys() {
        assert_eq!(temp_bucket(DecisionKind::Choice, 4), "choice:3-5");
        assert_eq!(temp_bucket(DecisionKind::Noul, 2), "noul:2");
        assert_eq!(temp_bucket(DecisionKind::Choice, 20), "choice:11+");
    }
}
