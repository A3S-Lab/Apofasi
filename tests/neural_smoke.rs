//! Neural System-1 smoke (feature `infer`).
//!
//! Set `APOFASI_CHECKPOINT` to a checkpoint directory that follows the
//! published layout (`rl_agent_config.json`, `model.safetensors`,
//! `encoder/config.json`, `tokenizer/tokenizer.json`).

#![cfg(feature = "infer")]

use a3s_apofasi::{
    Criteria, DecisionEngine, DecisionKind, DeviceRequest, NeuralEngine, Question, State,
    SystemOneRequest,
};
use indexmap::IndexMap;
use serde_json::json;
use std::path::PathBuf;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from)
}

fn refund_request() -> SystemOneRequest {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("invoices payments refunds")));
    opts.insert("technical".into(), Some(json!("bugs outages errors")));
    opts.insert("sales".into(), Some(json!("pricing contracts")));
    let mut questions = IndexMap::new();
    questions.insert(
        "department".into(),
        Question::new(
            DecisionKind::Choice,
            json!("Which department should handle this request?"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap(),
    );
    questions.insert(
        "refund_requested".into(),
        Question::new(
            DecisionKind::Noul,
            json!("Does the user explicitly request a refund?"),
            None,
        )
        .unwrap(),
    );
    SystemOneRequest {
        model: None,
        state: State::Text(
            "Hi, we were billed twice for March. Please refund the duplicate today.".into(),
        ),
        questions,
    }
}

#[test]
fn neural_english_smoke_cpu() {
    let Some(root) = checkpoint_dir() else {
        eprintln!("skip: set APOFASI_CHECKPOINT to run neural smoke");
        return;
    };
    let engine = NeuralEngine::load_with(&root, DeviceRequest::Cpu)
        .unwrap_or_else(|e| panic!("load checkpoint {}: {e}", root.display()));
    let res = engine.decide(&refund_request()).expect("decide");
    assert!(res.model.starts_with("apofasi-"), "model id {}", res.model);
    match &res.answers["department"] {
        a3s_apofasi::Answer::Choice { choice, .. } => {
            assert_eq!(choice, "billing", "department={choice}");
        }
        other => panic!("unexpected department answer {other:?}"),
    }
    match &res.answers["refund_requested"] {
        a3s_apofasi::Answer::Noul { noul } => {
            assert!(*noul > 0.5, "noul={noul}");
        }
        other => panic!("unexpected noul answer {other:?}"),
    }
    assert!(res.usage.input_tokens > 0);
}

#[test]
#[cfg(feature = "metal")]
fn neural_english_smoke_metal() {
    let Some(root) = checkpoint_dir() else {
        eprintln!("skip: set APOFASI_CHECKPOINT to run neural smoke");
        return;
    };
    let engine = match NeuralEngine::load_with(&root, DeviceRequest::Metal) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip metal: {e}");
            return;
        }
    };
    let res = engine.decide(&refund_request()).expect("decide metal");
    match &res.answers["department"] {
        a3s_apofasi::Answer::Choice { choice, .. } => {
            assert_eq!(choice, "billing", "department={choice}");
        }
        other => panic!("unexpected department answer {other:?}"),
    }
}
