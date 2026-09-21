//! `a3s-apofasi` CLI — smoke, latency bench, and batch suite.
//!
//! Build: `cargo run --features cli --bin a3s-apofasi -- smoke`

#[path = "apofasi/suite.rs"]
mod suite;

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use a3s_apofasi::{
    gate_response, Answer, Criteria, DecisionEngine, DecisionKind, DeviceRequest, GatePolicy,
    LexicalEngine, NeuralEngine, Question, State, SystemOneRequest, SystemOneResponse,
};
use indexmap::IndexMap;
use serde_json::{json, Value};

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return ExitCode::from(2);
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "smoke" => match run_smoke(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("smoke failed: {e}");
                ExitCode::from(1)
            }
        },
        "bench" => match run_bench(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("bench failed: {e}");
                ExitCode::from(1)
            }
        },
        "suite" => match suite::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("suite failed: {e}");
                ExitCode::from(1)
            }
        },
        "help" | "--help" | "-h" => {
            print_help();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command {other}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    eprintln!(
        "Usage:\n  a3s-apofasi smoke [--lexical] [--json] [--checkpoint DIR] [--device auto|metal|cuda|cpu]\n  a3s-apofasi bench [--checkpoint DIR] [--device auto|metal|cuda|cpu] [--iters N] [--warmup N] [--case smoke|triage]\n  a3s-apofasi suite --cases FILE [--checkpoint DIR] [--device auto|metal|cuda|cpu] [--warmup N] [--iters N]\n"
    );
}

struct CommonOpts {
    checkpoint: Option<PathBuf>,
    device: DeviceRequest,
    lexical: bool,
    json: bool,
    iters: usize,
    warmup: usize,
    case: String,
}

fn parse_common(args: &[String]) -> Result<CommonOpts, String> {
    let mut opts = CommonOpts {
        checkpoint: env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from),
        device: DeviceRequest::Auto,
        lexical: false,
        json: false,
        iters: 20,
        warmup: 3,
        case: "smoke".into(),
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lexical" => opts.lexical = true,
            "--json" => opts.json = true,
            "--case" => {
                i += 1;
                opts.case = args.get(i).ok_or("--case needs a value")?.clone();
            }
            "--checkpoint" => {
                i += 1;
                opts.checkpoint = Some(PathBuf::from(
                    args.get(i).ok_or("--checkpoint needs a path")?,
                ));
            }
            "--device" => {
                i += 1;
                let raw = args.get(i).ok_or("--device needs a value")?;
                opts.device = match raw.as_str() {
                    "auto" => DeviceRequest::Auto,
                    "metal" => DeviceRequest::Metal,
                    "cuda" => DeviceRequest::Cuda,
                    "cpu" => DeviceRequest::Cpu,
                    other => return Err(format!("unknown device {other}")),
                };
            }
            "--iters" => {
                i += 1;
                opts.iters = args
                    .get(i)
                    .ok_or("--iters needs a value")?
                    .parse()
                    .map_err(|_| "--iters must be an integer")?;
            }
            "--warmup" => {
                i += 1;
                opts.warmup = args
                    .get(i)
                    .ok_or("--warmup needs a value")?
                    .parse()
                    .map_err(|_| "--warmup must be an integer")?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(opts)
}

fn sample_request() -> SystemOneRequest {
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

fn triage_request() -> SystemOneRequest {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("invoices, payments, refunds")));
    opts.insert(
        "technical".into(),
        Some(json!("bugs, outages, system errors")),
    );
    opts.insert("sales".into(), Some(json!("pricing, new contracts")));
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
    questions.insert(
        "urgency".into(),
        Question::new(
            DecisionKind::Score,
            json!("How urgent is this request?"),
            Some(Criteria::Score(vec![
                json!("not urgent"),
                json!("soon"),
                json!("critical deadline or blocking issue"),
            ])),
        )
        .unwrap(),
    );
    questions.insert(
        "churn_risk".into(),
        Question::new(
            DecisionKind::Noul,
            json!("Does the user threaten to cancel or leave?"),
            None,
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
        state: State::Object(serde_json::Map::from_iter([
            ("from".into(), json!("user@acme.com")),
            ("subject".into(), json!("Duplicate charge on invoice #4411")),
            (
                "body".into(),
                json!(
                    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
                ),
            ),
        ])),
        questions,
    }
}

fn compact_answers(res: &SystemOneResponse) -> Value {
    let mut out = serde_json::Map::new();
    for (key, answer) in &res.answers {
        let slim = match answer {
            Answer::Choice {
                choice, confidence, ..
            } => json!({
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
            }),
            Answer::Score {
                score, confidence, ..
            } => json!({
                "type": "score",
                "score": score,
                "confidence": confidence,
            }),
            Answer::Noul { noul } => json!({
                "type": "noul",
                "noul": noul,
            }),
        };
        out.insert(key.clone(), slim);
    }
    Value::Object(out)
}

fn gates_json(gates: &std::collections::BTreeMap<String, a3s_apofasi::GateAction>) -> Value {
    let mut out = serde_json::Map::new();
    for (k, v) in gates {
        out.insert(k.clone(), json!(format!("{v:?}")));
    }
    Value::Object(out)
}

fn run_smoke(args: &[String]) -> Result<(), String> {
    let opts = parse_common(args)?;
    let req = sample_request();
    if opts.lexical {
        let engine = LexicalEngine::default();
        let res = engine.decide(&req).map_err(|e| e.to_string())?;
        let gates = gate_response(&res, &GatePolicy::default());
        if opts.json {
            println!(
                "{}",
                json!({
                    "backend": "lexical",
                    "model": res.model,
                    "answers": compact_answers(&res),
                    "gates": gates_json(&gates),
                })
            );
        } else {
            println!(
                "ok lexical model={} answers={} gates={gates:?}",
                res.model,
                res.answers.len()
            );
        }
        return Ok(());
    }
    let ckpt = opts
        .checkpoint
        .ok_or("set APOFASI_CHECKPOINT or pass --checkpoint (or use --lexical)")?;
    let engine = NeuralEngine::load_with(&ckpt, opts.device).map_err(|e| e.to_string())?;
    let res = engine.decide(&req).map_err(|e| e.to_string())?;
    let gates = gate_response(&res, &GatePolicy::default());
    if opts.json {
        println!(
            "{}",
            json!({
                "backend": "neural",
                "model": res.model,
                "device": format!("{:?}", engine.device()),
                "answers": compact_answers(&res),
                "gates": gates_json(&gates),
            })
        );
    } else {
        println!(
            "ok neural model={} device={:?} answers={} gates={gates:?}",
            res.model,
            engine.device(),
            res.answers.len()
        );
    }
    Ok(())
}

fn run_bench(args: &[String]) -> Result<(), String> {
    let opts = parse_common(args)?;
    let ckpt = opts
        .checkpoint
        .ok_or("bench requires APOFASI_CHECKPOINT or --checkpoint")?;
    let engine = NeuralEngine::load_with(&ckpt, opts.device).map_err(|e| e.to_string())?;
    let req = match opts.case.as_str() {
        "smoke" => sample_request(),
        "triage" => triage_request(),
        other => return Err(format!("unknown --case {other} (smoke|triage)")),
    };
    for _ in 0..opts.warmup {
        let _ = engine.decide(&req).map_err(|e| e.to_string())?;
    }
    let mut times = Vec::with_capacity(opts.iters);
    for _ in 0..opts.iters {
        let t0 = Instant::now();
        let _ = engine.decide(&req).map_err(|e| e.to_string())?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "bench case={} questions={} model={} device={:?} iters={} warmup={} p50_ms={:.2} mean_ms={:.2}",
        opts.case,
        req.questions.len(),
        engine.model_id(),
        engine.device(),
        opts.iters,
        opts.warmup,
        p50,
        mean
    );
    Ok(())
}
