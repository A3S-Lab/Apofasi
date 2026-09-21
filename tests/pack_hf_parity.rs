//! HF tokenizer packing parity (requires `infer` + `APOFASI_CHECKPOINT`).

#![cfg(feature = "infer")]

use a3s_apofasi::{
    pack_question, Criteria, DecisionKind, HfTokenizer, Question, SequenceConfig, State,
};
use indexmap::IndexMap;
use serde_json::json;
use std::path::PathBuf;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from)
}

#[test]
fn hf_pack_choice_has_mask_markers() {
    let Some(root) = checkpoint_dir() else {
        eprintln!("skip: set APOFASI_CHECKPOINT for HF packing parity");
        return;
    };
    let tok_path = root.join("tokenizer/tokenizer.json");
    let tok = HfTokenizer::from_file(&tok_path).expect("load tokenizer");

    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("refunds")));
    opts.insert("technical".into(), Some(json!("bugs")));
    let q = Question::new(
        DecisionKind::Choice,
        json!("Which department should handle this request?"),
        Some(Criteria::Choice(opts)),
    )
    .unwrap();
    let packed = pack_question(
        &tok,
        tok.specials,
        &tok.mask_token,
        &State::Text("Please refund my invoice.".into()),
        "department",
        &q,
        SequenceConfig::default(),
    )
    .expect("pack");

    assert_eq!(packed.input_ids[0], tok.specials.cls);
    assert_eq!(*packed.input_ids.last().unwrap(), tok.specials.sep);
    assert_eq!(packed.markers.len(), 2);
    for &m in &packed.markers {
        assert_eq!(packed.input_ids[m], tok.specials.mask);
    }
    // Determinism: packing twice yields identical ids.
    let packed2 = pack_question(
        &tok,
        tok.specials,
        &tok.mask_token,
        &State::Text("Please refund my invoice.".into()),
        "department",
        &q,
        SequenceConfig::default(),
    )
    .unwrap();
    assert_eq!(packed.input_ids, packed2.input_ids);
    assert_eq!(packed.markers, packed2.markers);
}
