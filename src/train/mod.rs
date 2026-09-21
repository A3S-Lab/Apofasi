//! Training feature: publish-layout checks and fine-tune loop scaffolding.
//!
//! Reward math lives in [`crate::reward`] (always available). This module owns
//! checkpoint publish contracts and a documented GRPO-style loop entry point.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::infer::CheckpointPaths;
use crate::primitive::DecisionKind;
use crate::reward::{proper_reward, RewardWeights};

/// On-disk layout expected when publishing a trained checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishLayout {
    /// Checkpoint root that hosts will load.
    pub root: PathBuf,
}

impl PublishLayout {
    /// Validate that `root` matches the stable host load contract.
    pub fn validate(root: impl AsRef<Path>) -> Result<Self> {
        let paths = CheckpointPaths::resolve(root)?;
        Ok(Self { root: paths.root })
    }

    /// Required relative paths inside a published checkpoint.
    pub fn required_files() -> &'static [&'static str] {
        &[
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
        ]
    }
}

/// One labelled training example for offline reward evaluation.
#[derive(Debug, Clone)]
pub struct RewardExample {
    /// Reported probabilities.
    pub q: Vec<f32>,
    /// Target distribution (one-hot or soft).
    pub target: Vec<f32>,
    /// Question kind (RPS applies only to score).
    pub kind: DecisionKind,
}

/// Mean proper reward over a batch (evaluation / logging helper).
pub fn mean_batch_reward(examples: &[RewardExample], weights: &RewardWeights) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let sum: f32 = examples
        .iter()
        .map(|ex| proper_reward(&ex.q, &ex.target, ex.kind, weights))
        .sum();
    sum / examples.len() as f32
}

/// Configuration knobs for a domain fine-tune run (scaffold).
#[derive(Debug, Clone)]
pub struct FineTuneConfig {
    /// Base checkpoint to specialize.
    pub base_checkpoint: PathBuf,
    /// Output directory for the published layout.
    pub output_dir: PathBuf,
    /// Reward weights.
    pub reward: RewardWeights,
    /// Max updates before stop (hosts own the real schedule).
    pub max_updates: u64,
}

impl FineTuneConfig {
    /// Basic sanity checks before a host starts the loop.
    pub fn validate(&self) -> Result<()> {
        let _ = CheckpointPaths::resolve(&self.base_checkpoint)?;
        if self.max_updates == 0 {
            return Err(Error::Checkpoint(
                "fine-tune max_updates must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Documented fine-tune step outcome (no optimizer yet — scaffold for hosts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FineTuneStepStats {
    /// Mean proper reward on the micro-batch.
    pub mean_reward: f32,
    /// Number of examples in the step.
    pub batch_size: usize,
}

/// Evaluate a micro-batch with the RLCD proper reward.
///
/// Full GRPO parameter updates stay host/trainer-owned for now; this entry
/// point locks the reward contract those loops must use.
pub fn evaluate_finetune_step(
    examples: &[RewardExample],
    weights: &RewardWeights,
) -> FineTuneStepStats {
    FineTuneStepStats {
        mean_reward: mean_batch_reward(examples, weights),
        batch_size: examples.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_layout_lists_required_files() {
        assert!(PublishLayout::required_files().contains(&"model.safetensors"));
    }

    #[test]
    fn mean_batch_reward_prefers_good_predictions() {
        let good = RewardExample {
            q: vec![0.05, 0.95],
            target: vec![0.0, 1.0],
            kind: DecisionKind::Choice,
        };
        let bad = RewardExample {
            q: vec![0.95, 0.05],
            target: vec![0.0, 1.0],
            kind: DecisionKind::Choice,
        };
        let w = RewardWeights::default();
        let g = evaluate_finetune_step(&[good], &w);
        let b = evaluate_finetune_step(&[bad], &w);
        assert!(g.mean_reward > b.mean_reward);
        assert_eq!(g.batch_size, 1);
    }
}
