//! Split a choice that does not fit the head budget.
//!
//! One forward that shortens every option to the same prefix cannot see which
//! label is which. Each group stays inside the head budget and is scored with
//! the temperature bucket for its real size, including `choice:11+`. Options
//! are interleaved across groups so adjacent labels do not always compete in
//! the same first-round comparison. A group of three or more sends its
//! runner-up into the next comparison; a longer group also sends a close
//! third place when its mass is still a large fraction of the local winner.
//! After the tournament, the leading options get one joint forward so the
//! final argmax is not stuck with an approximate composition. Questions that
//! already fit are one group.

use crate::error::{Error, Result};
use crate::sequence::HEAD_INSTRUCTION_RESERVE;

/// Minimum local mass ratio for advancing a third-place option.
/// Kept low so a crowded group does not eliminate a near-contender that only
/// looks weak against many distractors.
const THIRD_PLACE_RATIO: f32 = 0.2;

/// How many composed leaders get a final joint forward after a tournament.
///
/// Only used as an upper bound when scanning leaders. A close race compares
/// just the near contenders so distractors do not reintroduce IIA failures.
pub const REFINE_TOP_K: usize = 8;

/// Minimum `second/best` ratio that triggers a joint refine forward.
pub const REFINE_CLOSE_RATIO: f32 = 0.5;

/// Minimum `third/best` ratio for including third place in a close refine.
pub const REFINE_THIRD_RATIO: f32 = 0.35;

/// How many leading composed options to compare jointly.
///
/// Returns 0 when the leader is already decisive. Otherwise 2, or 3 when a
/// third option is still close to the leader.
pub fn refine_leader_count(composed: &[f32], ranked: &[usize]) -> usize {
    if ranked.len() < 2 {
        return 0;
    }
    let best = composed[ranked[0]];
    let second = composed[ranked[1]];
    if best <= 0.0 || second / best < REFINE_CLOSE_RATIO {
        return 0;
    }
    if ranked.len() >= 3 {
        let third = composed[ranked[2]];
        if third / best >= REFINE_THIRD_RATIO {
            return 3;
        }
    }
    2
}

/// Partition option indices so each group fits in the head budget.
///
/// One group means the caller should pack the question once. More groups mean
/// the full option list would be shortened. Every index appears once. When
/// several groups are required, indices are interleaved rather than sliced
/// contiguously, so nearby criteria (common in label lists) are less often
/// eliminated against each other in round one.
pub fn plan_choice_groups(costs: &[usize], head_max_len: usize) -> Vec<Vec<usize>> {
    let sequential = plan_choice_groups_sequential(costs, head_max_len);
    if sequential.len() <= 1 {
        return sequential;
    }
    match plan_choice_groups_interleaved(costs, head_max_len, sequential.len()) {
        Some(groups) if groups.len() == sequential.len() => groups,
        _ => sequential,
    }
}

fn plan_choice_groups_sequential(costs: &[usize], head_max_len: usize) -> Vec<Vec<usize>> {
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
        let full = !current.is_empty() && used + cost > budget;
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

fn plan_choice_groups_interleaved(
    costs: &[usize],
    head_max_len: usize,
    group_count: usize,
) -> Option<Vec<Vec<usize>>> {
    let budget = head_max_len.saturating_sub(HEAD_INSTRUCTION_RESERVE).max(1);
    if group_count == 0 || costs.is_empty() {
        return None;
    }
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    let mut used = vec![0usize; group_count];
    for (index, &cost) in costs.iter().enumerate() {
        let cost = cost.max(1).min(budget);
        let mut placed = false;
        for offset in 0..group_count {
            let bucket = (index + offset) % group_count;
            if used[bucket] + cost <= budget {
                buckets[bucket].push(index);
                used[bucket] += cost;
                placed = true;
                break;
            }
        }
        if !placed {
            return None;
        }
    }
    let groups: Vec<Vec<usize>> = buckets
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect();
    let mut seen = vec![false; costs.len()];
    for group in &groups {
        for &index in group {
            if index >= costs.len() || seen[index] {
                return None;
            }
            seen[index] = true;
        }
    }
    if seen.iter().any(|present| !present) {
        return None;
    }
    Some(groups)
}

/// Local indexes that continue, best probability first.
///
/// Ties keep the earlier option. Two options have already been compared, so
/// only the winner continues. A longer group also continues its runner-up. A
/// group of five or more also continues third place when that mass is still a
/// large fraction of the local winner.
pub fn survivor_locals(probs: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&left, &right| probs[right].total_cmp(&probs[left]).then(left.cmp(&right)));
    let mut keep = 1.min(order.len());
    if order.len() >= 3 {
        keep = 2;
    }
    if order.len() >= 5 {
        let best = probs[order[0]];
        let third = probs[order[2]];
        if best > 0.0 && third / best >= THIRD_PLACE_RATIO {
            keep = 3;
        }
    }
    order.truncate(keep);
    order
}

/// Upper bound on continued options for a group length (tests).
#[cfg(test)]
pub fn survivor_count(group_len: usize) -> usize {
    if group_len >= 5 {
        3
    } else if group_len >= 3 {
        2
    } else {
        1
    }
}

/// Fold per-group distributions and the survivor distribution into one vector.
///
/// `groups` indexes the original options. `chunk_probs[g][j]` is the
/// probability of local option `j` inside group `g`. `survivor_locals[g]`
/// lists the local indexes that were passed to the next comparison, and
/// `survivor_probs` is that comparison's distribution in the same order,
/// concatenated across groups. An option that did not continue stays strictly
/// below the survivor that beat it, so the composed argmax is one of the
/// survivors. Every original option must appear once.
#[cfg(test)]
pub fn compose_grouped_probs(
    option_count: usize,
    groups: &[Vec<usize>],
    chunk_probs: &[Vec<f32>],
    winner_probs: &[f32],
) -> Result<Vec<f32>> {
    let survivors: Vec<Vec<usize>> = chunk_probs
        .iter()
        .map(|chunk| vec![argmax(chunk)])
        .collect();
    compose_survivor_probs(option_count, groups, chunk_probs, &survivors, winner_probs)
}

/// See [`compose_grouped_probs`] for the single-survivor case.
pub fn compose_survivor_probs(
    option_count: usize,
    groups: &[Vec<usize>],
    chunk_probs: &[Vec<f32>],
    survivor_locals: &[Vec<usize>],
    survivor_probs: &[f32],
) -> Result<Vec<f32>> {
    if groups.len() != chunk_probs.len() || groups.len() != survivor_locals.len() {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "grouped choice probabilities do not cover every group".into(),
        });
    }
    let declared: usize = survivor_locals.iter().map(Vec::len).sum();
    if declared != survivor_probs.len() || survivor_probs.iter().any(|p| !p.is_finite()) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "grouped choice survivors do not match the next comparison".into(),
        });
    }
    let mut out = vec![0.0f32; option_count];
    let mut seen = vec![false; option_count];
    let mut cursor = 0usize;
    for (group, (chunk, locals)) in groups
        .iter()
        .zip(chunk_probs.iter().zip(survivor_locals.iter()))
    {
        let next = &survivor_probs[cursor..cursor + locals.len()];
        cursor += locals.len();
        if group.is_empty()
            || chunk.len() != group.len()
            || locals.is_empty()
            || chunk.iter().any(|p| !p.is_finite())
        {
            return Err(Error::InvalidQuestion {
                id: String::new(),
                reason: "grouped choice chunk does not match its options".into(),
            });
        }
        let mut continued = vec![false; group.len()];
        let mut next_of = vec![0.0f32; group.len()];
        let mut anchor_local = locals[0];
        for (&local, &p_next) in locals.iter().zip(next.iter()) {
            if local >= group.len() || continued[local] {
                return Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: format!("grouped choice survivor {local} is missing or repeated"),
                });
            }
            continued[local] = true;
            next_of[local] = p_next;
            if chunk[local] > chunk[anchor_local] {
                anchor_local = local;
            }
        }
        let anchor_mass = chunk[anchor_local].max(1e-12);
        let anchor_next = next_of[anchor_local];
        for (local, &index) in group.iter().enumerate() {
            if index >= option_count || seen[index] {
                return Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: format!("grouped choice index {index} is missing or repeated"),
                });
            }
            seen[index] = true;
            out[index] = if continued[local] {
                next_of[local]
            } else {
                chunk[local] / anchor_mass * anchor_next
            };
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "grouped choice did not assign every option".into(),
        });
    }
    let sum: f32 = out.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "grouped choice probabilities do not form a distribution".into(),
        });
    }
    for p in &mut out {
        *p /= sum;
    }
    Ok(out)
}

/// Indexes of the strongest composed options, best first.
pub fn top_probability_indexes(probs: &[f32], limit: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&left, &right| probs[right].total_cmp(&probs[left]).then(left.cmp(&right)));
    order.truncate(limit.min(probs.len()));
    order
}

/// Replace the probability mass on `top` with a fresh joint distribution.
///
/// Tournament composition is only approximate. A final forward over the
/// leading options restores a joint score among the candidates that still
/// matter, without shortening the rest of the label set to a shared prefix.
pub fn redistribute_top_probs(composed: &[f32], top: &[usize], fresh: &[f32]) -> Result<Vec<f32>> {
    if top.is_empty() || top.len() != fresh.len() {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "top-choice refine length mismatch".into(),
        });
    }
    if composed.iter().any(|p| !p.is_finite()) || fresh.iter().any(|p| !p.is_finite()) {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "top-choice refine probabilities must be finite".into(),
        });
    }
    let mut seen = vec![false; composed.len()];
    let mut mass = 0.0f32;
    for &index in top {
        if index >= composed.len() || seen[index] {
            return Err(Error::InvalidQuestion {
                id: String::new(),
                reason: format!("top-choice refine index {index} is missing or repeated"),
            });
        }
        seen[index] = true;
        mass += composed[index];
    }
    let mass = mass.max(1e-12);
    let mut out = composed.to_vec();
    for (&index, &p) in top.iter().zip(fresh.iter()) {
        out[index] = p * mass;
    }
    let sum: f32 = out.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InvalidQuestion {
            id: String::new(),
            reason: "top-choice refine did not form a distribution".into(),
        });
    }
    for p in &mut out {
        *p /= sum;
    }
    Ok(out)
}

#[cfg(test)]
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
            let sum: usize = group.iter().map(|&i| costs[i].min(176)).sum();
            assert!(sum <= 192 - 16, "group cost {sum}");
            seen.extend(group.iter().copied());
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn wide_budget_uses_the_eleven_plus_bucket() {
        let costs = vec![4usize; 72];
        let groups = plan_choice_groups(&costs, 192);
        assert!(
            groups.iter().any(|group| group.len() > 10),
            "a head budget that fits more than 10 options must not stop at 10"
        );
        let mut seen = Vec::new();
        for group in &groups {
            let sum: usize = group.iter().map(|&index| costs[index]).sum();
            assert!(sum <= 192 - 16, "group cost {sum}");
            seen.extend(group.iter().copied());
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..72).collect::<Vec<_>>());
    }

    #[test]
    fn wide_groups_are_interleaved() {
        let costs = vec![20usize; 24];
        let groups = plan_choice_groups(&costs, 192);
        assert!(groups.len() > 1);
        // Contiguous packing would put 0..k in the first group. Interleaving
        // spreads early indexes across groups.
        let first = &groups[0];
        assert!(
            first.windows(2).any(|pair| pair[1] > pair[0] + 1),
            "expected non-contiguous indexes in an interleaved group: {first:?}"
        );
    }

    #[test]
    fn composed_argmax_is_the_group_winner() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let chunk_probs = vec![vec![0.2, 0.8], vec![0.7, 0.3]];
        let winner_probs = vec![0.25, 0.75];
        let probs = compose_grouped_probs(4, &groups, &chunk_probs, &winner_probs).unwrap();
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

    #[test]
    fn runner_up_can_win_the_next_comparison() {
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let chunk_probs = vec![vec![0.5, 0.4, 0.1], vec![0.6, 0.3, 0.1]];
        let locals = vec![vec![0, 1], vec![0, 1]];
        // Next comparison prefers the first group's runner-up (global 1).
        let survivor_probs = vec![0.1, 0.7, 0.15, 0.05];
        let probs =
            compose_survivor_probs(6, &groups, &chunk_probs, &locals, &survivor_probs).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 1);
        assert!(probs[2] < probs[0]);
        assert!(probs[5] < probs[3]);
    }

    #[test]
    fn crowded_group_keeps_the_runner_up() {
        let locals = survivor_locals(&[0.5, 0.4, 0.1]);
        assert_eq!(locals, vec![0, 1]);
        assert_eq!(survivor_locals(&[0.2, 0.8]), vec![1]);
        assert_eq!(survivor_count(2), 1);
        assert_eq!(survivor_count(9), 3);
    }

    #[test]
    fn close_third_place_advances_in_large_groups() {
        let locals = survivor_locals(&[0.40, 0.35, 0.20, 0.03, 0.02]);
        assert_eq!(locals, vec![0, 1, 2]);
        let distant = survivor_locals(&[0.80, 0.15, 0.02, 0.02, 0.01]);
        assert_eq!(distant, vec![0, 1]);
    }

    #[test]
    fn refine_can_promote_the_runner_up() {
        let composed = vec![0.45, 0.35, 0.10, 0.10];
        let ranked = top_probability_indexes(&composed, REFINE_TOP_K);
        assert_eq!(refine_leader_count(&composed, &ranked), 2);
        let top = &ranked[..2];
        let refined = redistribute_top_probs(&composed, top, &[0.2, 0.8]).unwrap();
        let best = refined
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 1);
        let sum: f32 = refined.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(refined[2] > 0.0);
        assert!(refined[3] > 0.0);
    }

    #[test]
    fn refine_skips_a_decisive_leader() {
        let composed = vec![0.80, 0.10, 0.05, 0.05];
        let ranked = top_probability_indexes(&composed, REFINE_TOP_K);
        assert_eq!(refine_leader_count(&composed, &ranked), 0);
    }

    #[test]
    fn refine_includes_a_close_third() {
        let composed = vec![0.40, 0.30, 0.20, 0.10];
        let ranked = top_probability_indexes(&composed, REFINE_TOP_K);
        assert_eq!(refine_leader_count(&composed, &ranked), 3);
    }

    #[test]
    fn refine_triggers_on_a_borderline_runner_up() {
        let composed = vec![0.45, 0.25, 0.15, 0.15];
        let ranked = top_probability_indexes(&composed, REFINE_TOP_K);
        assert_eq!(refine_leader_count(&composed, &ranked), 2);
    }

    #[test]
    fn short_chunk_does_not_renormalize_the_rest() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let chunk_probs = vec![vec![0.2, 0.8], vec![1.0]];
        let winner_probs = vec![0.25, 0.75];
        let err = compose_grouped_probs(4, &groups, &chunk_probs, &winner_probs)
            .expect_err("a short chunk must not drop an option");
        assert!(err.to_string().contains("does not match"));
    }
}
