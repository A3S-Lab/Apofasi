//! Dependency-free script / language detection for checkpoint routing.
//!
//! Script detection is exact; Latin-language guessing is a best-effort
//! stopword / diacritic heuristic.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::schema::State;

/// Detection evidence attached to a route decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Dominant script label (`latin`, `han`, `devanagari`, …) or `unknown`.
    pub script: String,
    /// Fraction of letters that are non-Latin.
    pub non_latin_fraction: f32,
    /// Best-effort language code for Latin text (`en`, `de`, …).
    pub language: String,
    /// Whether the Latin heuristic believes the text is English.
    pub is_english: bool,
}

const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

fn is_latin_letter(cp: u32) -> bool {
    cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp)
}

fn script_of(cp: u32) -> Option<&'static str> {
    for (name, ranges) in SCRIPT_RANGES {
        for (lo, hi) in *ranges {
            if cp >= *lo && cp <= *hi {
                return Some(*name);
            }
        }
    }
    None
}

/// Dominant script of `text`.
pub fn detect_script(text: &str) -> (String, f32) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut latin = 0usize;
    let mut letters = 0usize;
    for ch in text.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        letters += 1;
        let cp = ch as u32;
        if is_latin_letter(cp) {
            latin += 1;
        } else if let Some(name) = script_of(cp) {
            *counts.entry(name).or_insert(0) += 1;
        }
    }
    if letters == 0 {
        return ("unknown".into(), 0.0);
    }
    let non_latin = letters - latin;
    let non_latin_fraction = non_latin as f32 / letters as f32;
    if non_latin * 2 >= letters {
        let best = counts
            .into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(n, _)| n)
            .unwrap_or("unknown");
        (best.into(), non_latin_fraction)
    } else {
        ("latin".into(), non_latin_fraction)
    }
}

fn stopwords() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        (
            "en",
            vec![
                "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
                "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
                "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
            ],
        ),
        (
            "de",
            vec![
                "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit",
                "für", "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind",
                "oder", "aber",
            ],
        ),
        (
            "fr",
            vec![
                "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec",
                "sur", "pas", "plus", "nous", "vous", "cette", "mais", "sont", "ont", "aux", "ce",
            ],
        ),
        (
            "es",
            vec![
                "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
                "pero", "son", "este", "esta", "todo", "muy", "hay", "sus",
            ],
        ),
    ])
}

fn guess_latin_language(text: &str) -> (String, bool) {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return ("en".into(), true);
    }
    let stops = stopwords();
    let mut scores: HashMap<&str, i32> = HashMap::new();
    for (lang, list) in &stops {
        let mut score = 0i32;
        for w in &words {
            if list.iter().any(|s| s == w) {
                score += 1;
            }
        }
        scores.insert(*lang, score);
    }
    let en = *scores.get("en").unwrap_or(&0);
    let mut best_lang = "en";
    let mut best = en;
    for (lang, score) in &scores {
        if *lang == "en" {
            continue;
        }
        if *score > best {
            best = *score;
            best_lang = lang;
        }
    }
    // Require a margin before calling non-English.
    let is_english = best_lang == "en" || en + 2 >= best;
    (
        if is_english {
            "en".into()
        } else {
            best_lang.into()
        },
        is_english,
    )
}

/// Analyse flattened state text for routing.
pub fn analyse_text(text: &str) -> Detection {
    let (script, non_latin_fraction) = detect_script(text);
    if script == "latin" {
        let (language, is_english) = guess_latin_language(text);
        Detection {
            script,
            non_latin_fraction,
            language,
            is_english,
        }
    } else {
        Detection {
            script,
            non_latin_fraction,
            language: "und".into(),
            is_english: false,
        }
    }
}

/// Analyse a structured [`State`].
pub fn analyse(state: &State) -> Detection {
    analyse_text(&state.flat_text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_han_script() {
        let d = analyse_text("我们被重复扣费了，请马上退款");
        assert_eq!(d.script, "han");
        assert!(!d.is_english);
    }

    #[test]
    fn detects_english_latin() {
        let d = analyse_text("Please refund the duplicate charge on my invoice.");
        assert_eq!(d.script, "latin");
        assert!(d.is_english);
    }
}
