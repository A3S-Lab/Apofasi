//! Strictly proper scoring-rule rewards (RLCD core math).
//!
//! Pure Rust, no ML framework. Used by the `train` feature and by hosts that
//! want offline calibration / evaluation without Candle.

use crate::primitive::DecisionKind;

/// Weights for the composite proper reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardWeights {
    /// Spherical score weight.
    pub w_sph: f32,
    /// Ranked probability score weight (score questions only).
    pub w_rps: f32,
    /// Floor applied to `log(q)` terms.
    pub log_floor: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            w_sph: 0.5,
            w_rps: 1.0,
            log_floor: -9.21,
        }
    }
}

/// Strictly proper reward for one reported distribution.
///
/// `q` and `target` are length-`K` probability vectors (already masked / padded
/// with zeros for invalid options). For [`DecisionKind::Score`], the ranked
/// probability score (RPS) term is subtracted.
pub fn proper_reward(
    q: &[f32],
    target: &[f32],
    kind: DecisionKind,
    weights: &RewardWeights,
) -> f32 {
    assert_eq!(q.len(), target.len(), "q/target length mismatch");
    let k = q.len();
    if k == 0 {
        return 0.0;
    }

    let mut log_score = 0.0f32;
    let mut dot = 0.0f32;
    let mut q_norm_sq = 0.0f32;
    for i in 0..k {
        let qi = q[i].max(0.0);
        let ti = target[i].max(0.0);
        let logq = qi.max(1e-12).ln().max(weights.log_floor);
        log_score += ti * logq;
        dot += ti * qi;
        q_norm_sq += qi * qi;
    }
    let sph = dot / q_norm_sq.sqrt().max(1e-9);
    let mut r = log_score + weights.w_sph * sph;

    if kind == DecisionKind::Score && k >= 2 {
        let mut cdf_q = 0.0f32;
        let mut cdf_t = 0.0f32;
        let mut rps = 0.0f32;
        let mut active = 0.0f32;
        for i in 0..k {
            // Treat near-zero pad slots as inactive when both are ~0.
            let active_i = if q[i].abs() > 1e-12 || target[i].abs() > 1e-12 {
                1.0
            } else {
                0.0
            };
            cdf_q += q[i].max(0.0);
            cdf_t += target[i].max(0.0);
            rps += (cdf_q - cdf_t).powi(2) * active_i;
            active += active_i;
        }
        let denom = (active.max(2.0) - 1.0).max(1.0);
        r -= weights.w_rps * (rps / denom);
    }
    r
}

/// Expected Calibration Error across confidence bins.
pub fn ece_score(conf: &[f32], correct: &[bool], bins: usize) -> f32 {
    assert_eq!(conf.len(), correct.len());
    if conf.is_empty() || bins == 0 {
        return f32::NAN;
    }
    let n = conf.len() as f32;
    let mut e = 0.0f32;
    for b in 0..bins {
        let lo = b as f32 / bins as f32;
        let hi = (b + 1) as f32 / bins as f32;
        let mut count = 0u32;
        let mut conf_sum = 0.0f32;
        let mut correct_sum = 0.0f32;
        for i in 0..conf.len() {
            let c = conf[i].clamp(0.0, 1.0);
            let in_bin = if b == 0 {
                (0.0..=hi).contains(&c)
            } else {
                c > lo && c <= hi
            };
            if !in_bin {
                continue;
            }
            count += 1;
            conf_sum += c;
            if correct[i] {
                correct_sum += 1.0;
            }
        }
        if count > 0 {
            let w = count as f32 / n;
            let avg_conf = conf_sum / count as f32;
            let avg_acc = correct_sum / count as f32;
            e += w * (avg_conf - avg_acc).abs();
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_scores_high() {
        let q = [0.0, 1.0];
        let t = [0.0, 1.0];
        let r = proper_reward(&q, &t, DecisionKind::Noul, &RewardWeights::default());
        assert!(r > 0.0, "r={r}");
    }

    #[test]
    fn wrong_prediction_scores_lower() {
        let good = proper_reward(
            &[0.05, 0.95],
            &[0.0, 1.0],
            DecisionKind::Choice,
            &RewardWeights::default(),
        );
        let bad = proper_reward(
            &[0.95, 0.05],
            &[0.0, 1.0],
            DecisionKind::Choice,
            &RewardWeights::default(),
        );
        assert!(good > bad, "good={good} bad={bad}");
    }

    #[test]
    fn score_kind_applies_rps_penalty() {
        let weights = RewardWeights::default();
        let q = [0.1, 0.2, 0.7];
        let t = [0.0, 0.0, 1.0];
        let as_choice = proper_reward(&q, &t, DecisionKind::Choice, &weights);
        let as_score = proper_reward(&q, &t, DecisionKind::Score, &weights);
        // RPS penalty should not increase the reward for score questions.
        assert!(
            as_score <= as_choice + 1e-5,
            "choice={as_choice} score={as_score}"
        );
    }

    #[test]
    fn ece_perfect_calibration_near_zero() {
        let conf = [0.9f32, 0.9, 0.1, 0.1];
        let correct = [true, true, false, false];
        let e = ece_score(&conf, &correct, 10);
        assert!(e < 0.15, "ece={e}");
    }
}
