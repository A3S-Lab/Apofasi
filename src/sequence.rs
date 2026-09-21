//! Sequence packing for typed decision prompts.
//!
//! Format:
//! `[CLS] "{kind} question: {instructions}" [SEP] [MASK] opt0 … [SEP] {state} [SEP]`
//!
//! Tokenization is injected via [`Tokenize`] so M0 stays framework-free while M1
//! plugs in HuggingFace `tokenizers`.

use crate::error::{Error, Result};
use crate::primitive::DecisionKind;
use crate::schema::{criterion_text, instructions_text, Criteria, Question, State};

/// Special token ids required by the packer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    /// CLS / BOS id.
    pub cls: u32,
    /// SEP id.
    pub sep: u32,
    /// MASK id (option marker).
    pub mask: u32,
    /// PAD id (right-padding for batched encoder forwards).
    pub pad: u32,
}

/// Tokenizer surface used by the packer.
pub trait Tokenize {
    /// Encode raw text without adding special tokens.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>>;
}

/// Packing configuration (mirrors checkpoint `max_len` / `head_max_len`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceConfig {
    /// Total sequence length.
    pub max_len: usize,
    /// Budget for instructions + option markers.
    pub head_max_len: usize,
    /// Per-option token cap before budget rebalance (default 48).
    pub option_cap: usize,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            max_len: 512,
            head_max_len: 192,
            option_cap: 48,
        }
    }
}

/// Packed question ready for batching / model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedQuestion {
    /// Token ids including specials.
    pub input_ids: Vec<u32>,
    /// Absolute positions of each `[MASK]` marker.
    pub markers: Vec<usize>,
    /// Dense qtype id.
    pub qtype: u8,
    /// Rendered option labels in marker order (for decoding argmax).
    pub option_labels: Vec<String>,
}

/// Render option strings for the decision head prompt.
pub fn render_options(question: &Question) -> Result<Vec<(String, String)>> {
    match question.type_ {
        DecisionKind::Choice => {
            let Some(Criteria::Choice(opts)) = &question.criteria else {
                return Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: "choice criteria missing".into(),
                });
            };
            Ok(opts
                .iter()
                .map(|(k, v)| {
                    let text = match v {
                        None => k.clone(),
                        Some(desc) => {
                            let rendered = criterion_text(desc);
                            if rendered.is_empty() {
                                k.clone()
                            } else {
                                format!("{k}: {rendered}")
                            }
                        }
                    };
                    (k.clone(), text)
                })
                .collect())
        }
        DecisionKind::Score => {
            let Some(Criteria::Score(levels)) = &question.criteria else {
                return Err(Error::InvalidQuestion {
                    id: String::new(),
                    reason: "score criteria missing".into(),
                });
            };
            Ok(levels
                .iter()
                .enumerate()
                .map(|(i, level)| {
                    (
                        i.to_string(),
                        format!("level {i}: {}", criterion_text(level)),
                    )
                })
                .collect())
        }
        DecisionKind::Noul => {
            let (false_g, true_g) = match &question.criteria {
                None => (None, None),
                Some(Criteria::Noul {
                    false_gloss,
                    true_gloss,
                }) => (false_gloss.as_ref(), true_gloss.as_ref()),
                Some(_) => {
                    return Err(Error::InvalidQuestion {
                        id: String::new(),
                        reason: "noul criteria must be noul glosses or omitted".into(),
                    });
                }
            };
            let f = false_g
                .map(criterion_text)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "no, the statement does not hold".into());
            let t = true_g
                .map(criterion_text)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "yes, the statement holds".into());
            Ok(vec![
                ("false".into(), format!("false: {f}")),
                ("true".into(), format!("true: {t}")),
            ])
        }
    }
}

/// Pack one question + state into the canonical token layout.
///
/// `mask_token_str` is stripped from instructions/options/state (replaced with
/// a space) before encoding so option markers stay unambiguous.
pub fn pack_question<T: Tokenize>(
    tok: &T,
    specials: SpecialTokens,
    mask_token_str: &str,
    state: &State,
    question_id: &str,
    question: &Question,
    cfg: SequenceConfig,
) -> Result<PackedQuestion> {
    let rendered = render_options(question).map_err(|err| match err {
        Error::InvalidQuestion { reason, .. } => Error::InvalidQuestion {
            id: question_id.to_string(),
            reason,
        },
        other => other,
    })?;
    let option_labels: Vec<String> = rendered.iter().map(|(k, _)| k.clone()).collect();
    let option_texts: Vec<String> = rendered.into_iter().map(|(_, t)| t).collect();

    let ins = instructions_text(&question.instructions).replace(mask_token_str, " ");
    let head_text = format!("{} question: {}", question.type_.as_str(), ins);
    let mut head_ids = tok.encode_ordinary(&head_text)?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(option_texts.len());
    for text in &option_texts {
        let cleaned = text.replace(mask_token_str, " ");
        let mut ids = vec![specials.mask];
        let mut body = tok.encode_ordinary(&format!(" {cleaned}"))?;
        if body.len() > cfg.option_cap {
            body.truncate(cfg.option_cap);
        }
        ids.append(&mut body);
        opt_ids.push(ids);
    }

    let mut opt_budget = cfg
        .head_max_len
        .saturating_sub(opt_ids.iter().map(|o| o.len()).sum());
    if opt_budget < 16 {
        let per = ((cfg.head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
        for o in &mut opt_ids {
            if o.len() > per {
                o.truncate(per);
            }
        }
        opt_budget = cfg
            .head_max_len
            .saturating_sub(opt_ids.iter().map(|o| o.len()).sum());
    }
    let head_keep = opt_budget.max(8);
    if head_ids.len() > head_keep {
        head_ids.truncate(head_keep);
    }

    let mut ids = Vec::with_capacity(cfg.max_len);
    ids.push(specials.cls);
    ids.extend(head_ids);
    ids.push(specials.sep);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(specials.sep);

    if markers.len() != option_labels.len() {
        return Err(Error::HeadBudgetExceeded {
            id: question_id.to_string(),
            head_max_len: cfg.head_max_len,
        });
    }

    let room = cfg.max_len.saturating_sub(ids.len().saturating_add(1));
    let state_text = state.model_text()?.replace(mask_token_str, " ");
    let mut state_ids = tok.encode_ordinary(&state_text)?;
    if state_ids.len() > room {
        state_ids.truncate(room);
    }
    ids.extend(state_ids);
    ids.push(specials.sep);

    if ids.len() > cfg.max_len {
        ids.truncate(cfg.max_len);
    }
    let markers: Vec<usize> = markers.into_iter().filter(|m| *m < ids.len()).collect();
    if markers.len() != option_labels.len() {
        return Err(Error::HeadBudgetExceeded {
            id: question_id.to_string(),
            head_max_len: cfg.head_max_len,
        });
    }

    Ok(PackedQuestion {
        input_ids: ids,
        markers,
        qtype: question.type_.type_id(),
        option_labels,
    })
}

/// Whitespace / char-level stub tokenizer for unit tests (not for production).
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl Tokenize for ByteTokenizer {
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| u32::from(c) % 50_000 + 16).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Question;
    use indexmap::IndexMap;
    use serde_json::json;

    fn specials() -> SpecialTokens {
        SpecialTokens {
            cls: 0,
            sep: 1,
            mask: 2,
            pad: 3,
        }
    }

    #[test]
    fn packs_choice_with_markers() {
        let mut opts = IndexMap::new();
        opts.insert("billing".into(), Some(json!("refunds")));
        opts.insert("tech".into(), Some(json!("bugs")));
        let q = Question::new(
            DecisionKind::Choice,
            json!("Which department?"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap();
        let state = State::Text("Please refund my invoice.".into());
        let packed = pack_question(
            &ByteTokenizer,
            specials(),
            "[MASK]",
            &state,
            "department",
            &q,
            SequenceConfig::default(),
        )
        .unwrap();
        assert_eq!(packed.markers.len(), 2);
        assert_eq!(packed.option_labels, ["billing", "tech"]);
        assert_eq!(packed.input_ids[0], 0);
        assert_eq!(packed.input_ids[packed.markers[0]], 2);
        assert_eq!(packed.qtype, 0);
    }

    #[test]
    fn noul_defaults_to_false_true() {
        let q = Question::new(DecisionKind::Noul, json!("Refund requested?"), None).unwrap();
        let opts = render_options(&q).unwrap();
        assert_eq!(opts.len(), 2);
        assert_eq!(opts[0].0, "false");
        assert_eq!(opts[1].0, "true");
    }

    #[test]
    fn score_renders_levels() {
        let q = Question::new(
            DecisionKind::Score,
            json!("Urgency"),
            Some(Criteria::Score(vec![json!("low"), json!("high")])),
        )
        .unwrap();
        let opts = render_options(&q).unwrap();
        assert_eq!(opts[0].1, "level 0: low");
        assert_eq!(opts[1].1, "level 1: high");
    }
}
