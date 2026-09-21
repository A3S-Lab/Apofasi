//! Split a choice that does not fit in the head budget.
//!
//! One forward that shortens every option to the same prefix cannot see which
//! label is which. Groups stay inside the instruction reserve and inside the
//! checkpoint's calibrated option-count buckets. A later pass compares the
//! group winners. Questions that already fit are one group and are unchanged.

use crate::sequence::HEAD_INSTRUCTION_RESERVE;

/// Largest group that still uses a calibrated choice temperature bucket.
const GROUP_CAP: usize = 10;

/// Partition option indices so each group fits in the head budget.
///
/// One group means the caller should pack the question once. More groups mean
/// the full option list would be shortened. Every index appears once, in order.
pub fn plan_choice_groups(costs: &[usize], head_max_len: usize) -> Vec<Vec<usize>> {
    let budget = head_max_len.saturating_sub(HEAD_INSTRUCTION_RESERVE).max(1);
    if costs.is_empty() {
        return Vec::new();
    }
    let total: usize = costs.iter().copied().sum();
    if total <= budget {
        return vec![(0..costs.len()).collect()];
    }

    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut used = 0usize;
    for (index, &cost) in costs.iter().enumerate() {
        let cost = cost.max(1).min(budget);
        let full = !current.is_empty() && (used + cost > budget || current.len() >= GROUP_CAP);
        if full {
            groups.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(index);
        used += cost;
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Fold per-group distributions and the winner distribution into one vector.
///
/// `groups` indexes the original options. `chunk_probs[g][j]` is the
/// probability of local option `j` inside group `g`. `winner_probs[g]` is the
/// probability that group `g`'s winner survives the comparison among winners.
/// The result is aligned with the original option order and sums to 1.
pub fn compose_grouped_probs(
    option_count: usize,
    groups: &[Vec<usize>],
    chunk_probs: &[Vec<f32>],
    winner_probs: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; option_count];
    for (group, (chunk, &p_group)) in groups
        .iter()
        .zip(chunk_probs.iter().zip(winner_probs.iter()))
    {
        if group.is_empty() || chunk.is_empty() {
            continue;
        }
        let win = argmax(chunk);
        let p_win = chunk[win].max(1e-12);
        for (local, &index) in group.iter().enumerate() {
            if index < out.len() && local < chunk.len() {
                out[index] = chunk[local] / p_win * p_group;
            }
        }
    }
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for p in &mut out {
            *p /= sum;
        }
    }
    out
}

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_choice_stays_one_group() {
        let groups = plan_choice_groups(&[8, 8, 8, 8], 192);
        assert_eq!(groups, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn overflow_splits_without_dropping_options() {
        let costs = vec![30usize; 40];
        let groups = plan_choice_groups(&costs, 192);
        assert!(groups.len() > 1);
        let mut seen = Vec::new();
        for group in &groups {
            assert!(!group.is_empty());
            assert!(group.len() <= 10);
            let sum: usize = group.iter().map(|&i| costs[i].min(176)).sum();
            assert!(sum <= 192 - 16, "group cost {sum}");
            seen.extend(group.iter().copied());
        }
        assert_eq!(seen, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn composed_argmax_is_the_group_winner() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let chunk_probs = vec![vec![0.2, 0.8], vec![0.7, 0.3]];
        let winner_probs = vec![0.25, 0.75];
        let probs = compose_grouped_probs(4, &groups, &chunk_probs, &winner_probs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 2);
        assert!(probs[3] < probs[2]);
        assert!(probs[1] < probs[2]);
    }
}
