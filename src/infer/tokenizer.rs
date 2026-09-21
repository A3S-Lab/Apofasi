//! HuggingFace fast tokenizer adapter.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::sequence::{SpecialTokens, Tokenize};

/// Fast tokenizer loaded from `tokenizer.json`.
pub struct HfTokenizer {
    inner: Tokenizer,
    /// Special token ids used by the packer.
    pub specials: SpecialTokens,
    /// Literal mask token string (stripped before encode).
    pub mask_token: String,
}

impl HfTokenizer {
    /// Load from a `tokenizer.json` path and resolve special ids.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref()).map_err(|e| {
            Error::Checkpoint(format!("load tokenizer {}: {e}", path.as_ref().display()))
        })?;
        let (specials, mask_token) = resolve_specials(&inner)?;
        Ok(Self {
            specials,
            mask_token,
            inner,
        })
    }
}

fn resolve_specials(inner: &Tokenizer) -> Result<(SpecialTokens, String)> {
    // ModernBERT English uses [CLS]/[SEP]/[MASK]/[PAD].
    // mmBERT / Gemma-style tokenizers use <s> / </s> / <mask> / <pad>.
    let pick = |candidates: &[&str]| -> Option<(u32, String)> {
        for tok in candidates {
            if let Some(id) = inner.token_to_id(tok) {
                return Some((id, (*tok).to_string()));
            }
        }
        None
    };
    let (cls, _) = pick(&["[CLS]", "<s>", "<bos>"])
        .ok_or_else(|| Error::Checkpoint("tokenizer missing CLS/BOS token".into()))?;
    let (sep, _) = pick(&["[SEP]", "</s>", "<eos>"])
        .ok_or_else(|| Error::Checkpoint("tokenizer missing SEP/EOS token".into()))?;
    let (mask, mask_token) = pick(&["[MASK]", "<mask>"])
        .ok_or_else(|| Error::Checkpoint("tokenizer missing MASK token".into()))?;
    let (pad, _) = pick(&["[PAD]", "<pad>"])
        .ok_or_else(|| Error::Checkpoint("tokenizer missing PAD token".into()))?;
    Ok((
        SpecialTokens {
            cls,
            sep,
            mask,
            pad,
        },
        mask_token,
    ))
}

impl Tokenize for HfTokenizer {
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Infer(format!("tokenize: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}
