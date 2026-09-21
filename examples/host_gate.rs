//! Host-side decision gate glue for Code / Desktop.
//!
//! Shows the recommended pattern: route, call System One, then map answers
//! through [`GatePolicy`] before automation.
//!
//! ```bash
//! cargo run --example host_gate
//! cargo run --example host_gate --features infer -- --checkpoint "$APOFASI_CHECKPOINT"
//! ```

use std::env;
use std::process::ExitCode;

use a3s_apofasi::{
    any_escalate, gate_response, Criteria, DecisionEngine, DecisionKind, GateAction, GatePolicy,
    LexicalEngine, Question, State, SystemOneRequest,
};
use indexmap::IndexMap;
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut checkpoint = env::var_os("APOFASI_CHECKPOINT").map(std::path::PathBuf::from);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                checkpoint = Some(std::path::PathBuf::from(
                    args.get(i).ok_or("--checkpoint needs a path")?,
                ));
            }
            "--help" | "-h" => {
                eprintln!("Usage: host_gate [--checkpoint DIR]");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}")),
        }
        i += 1;
    }

    let req = sample_request();
    let response = if let Some(root) = checkpoint {
        #[cfg(feature = "infer")]
        {
            use a3s_apofasi::{CheckpointRegistry, DeviceRequest};
            let mut registry = CheckpointRegistry::open(&root, DeviceRequest::Auto, 3)
                .map_err(|e| e.to_string())?;
            let (route, response) = registry
                .system_one_routed(req, None, None)
                .map_err(|e| e.to_string())?;
            println!("route={} reason={}", route.model.as_str(), route.reason);
            response
        }
        #[cfg(not(feature = "infer"))]
        {
            eprintln!(
                "warning: checkpoint {} ignored (rebuild with --features infer); using LexicalEngine",
                root.display()
            );
            LexicalEngine::default()
                .decide(&req)
                .map_err(|e| e.to_string())?
        }
    } else {
        LexicalEngine::default()
            .decide(&req)
            .map_err(|e| e.to_string())?
    };

    let policy = GatePolicy::default();
    let gates = gate_response(&response, &policy);
    println!("model={}", response.model);
    for (id, answer) in &response.answers {
        let gate = gates.get(id).copied().unwrap_or(GateAction::Escalate);
        println!("  {id}: {answer:?} → {}", gate.as_str());
    }
    if any_escalate(&gates) {
        println!("host_action=escalate (at least one answer below threshold)");
    } else {
        println!("host_action=auto");
    }
    Ok(())
}

fn sample_request() -> SystemOneRequest {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("refunds invoices")));
    opts.insert("sales".into(), Some(json!("pricing product interest")));
    opts.insert("other".into(), Some(json!("everything else")));
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
    SystemOneRequest {
        model: None,
        state: State::Text("Hi, I just wanted to learn more about your product.".into()),
        questions,
    }
}
