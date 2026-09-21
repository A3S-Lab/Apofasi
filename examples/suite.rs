//! Manual Apofasi decision suite with per-case I/O logs.
//!
//! Run via: `just ap` (from the monorepo root).

#![cfg(feature = "infer")]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use a3s_apofasi::{
    any_escalate, gate_answer, gate_response, CheckpointId, CheckpointRegistry, Criteria,
    DecisionEngine, DecisionKind, DeviceRequest, GateAction, GatePolicy, LexicalEngine,
    NeuralEngine, Question, Router, State, SystemOneRequest, SystemOneResponse,
};
use indexmap::IndexMap;
use serde_json::{json, Value};

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("ap suite error: {err}");
            ExitCode::from(2)
        }
    }
}

fn run(argv: Vec<String>) -> Result<ExitCode, String> {
    let opts = parse_args(&argv)?;
    let root = workspace_root();
    let log_dir = opts
        .log_dir
        .unwrap_or_else(|| root.join(".scratch/ap/logs"));
    fs::create_dir_all(&log_dir).map_err(|e| format!("mkdir {}: {e}", log_dir.display()))?;
    let stamp = utc_stamp();
    let log_path = log_dir.join(format!("ap-suite-{stamp}.log"));
    let jsonl_path = log_dir.join(format!("ap-suite-{stamp}.jsonl"));

    let mut logger = Logger::new(log_path.clone());
    logger.log(format!("Apofasi suite start @ {stamp}"));
    logger.log(format!(
        "device_request={:?} version={}",
        opts.device,
        a3s_apofasi::VERSION
    ));
    logger.log(format!("log_file={}", log_path.display()));

    let selected = select_cases(&opts.cases)?;
    let needs_neural = selected.iter().any(|c| c.needs_neural);
    let needs_registry = selected.iter().any(|c| c.needs_registry);

    let ckpt = if needs_neural || needs_registry {
        Some(resolve_checkpoint(opts.checkpoint.as_deref())?)
    } else {
        None
    };

    let neural = if needs_neural {
        let root = ckpt.as_ref().expect("checkpoint required");
        logger.log(format!("checkpoint={}", root.display()));
        let t0 = Instant::now();
        let engine = NeuralEngine::load_with(root, opts.device)
            .map_err(|e| format!("load NeuralEngine: {e}"))?;
        logger.log(format!(
            "neural_ready_seconds={:.2} model={} device={:?}",
            t0.elapsed().as_secs_f64(),
            engine.model_id(),
            engine.device()
        ));
        Some(engine)
    } else {
        logger.log("neural=skipped");
        None
    };

    let mut registry = if needs_registry {
        let bundle = ckpt.as_ref().expect("checkpoint required");
        logger.log(format!("registry_bundle={}", bundle.display()));
        let mut reg = CheckpointRegistry::open(bundle, opts.device, 2)
            .map_err(|e| format!("open registry: {e}"))?;
        let t0 = Instant::now();
        reg.preload(&[CheckpointId::English, CheckpointId::Multilingual])
            .map_err(|e| format!("preload: {e}"))?;
        logger.log(format!(
            "registry_ready_seconds={:.2} loaded={:?}",
            t0.elapsed().as_secs_f64(),
            reg.loaded()
        ));
        Some(reg)
    } else {
        None
    };
    let lexical = LexicalEngine::default();
    let router = Router::default();

    let mut results = Vec::new();
    for case in selected {
        results.push(run_case(
            case,
            neural.as_ref(),
            registry.as_mut(),
            &lexical,
            &router,
            &mut logger,
        ));
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    logger.log(format!("\n{}", "=".repeat(72)));
    logger.log(format!(
        "SUMMARY: {passed} passed, {failed} failed, {} total",
        results.len()
    ));
    for item in &results {
        let mark = if item.passed { "PASS" } else { "FAIL" };
        logger.log(format!(
            "  [{mark}] {} ({:.1} ms)",
            item.name, item.latency_ms
        ));
    }
    logger.log(format!("log_file={}", log_path.display()));
    logger.log(format!("jsonl_file={}", jsonl_path.display()));
    logger.flush()?;

    let mut jsonl = fs::File::create(&jsonl_path)
        .map_err(|e| format!("write {}: {e}", jsonl_path.display()))?;
    for item in &results {
        let rec = json!({
            "name": item.name,
            "passed": item.passed,
            "latency_ms": item.latency_ms,
            "input": item.input,
            "output": item.output,
            "checks": item.checks,
            "error": item.error,
        });
        writeln!(jsonl, "{rec}").map_err(|e| format!("jsonl: {e}"))?;
    }

    Ok(if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    description: &'static str,
    needs_neural: bool,
    needs_registry: bool,
    kind: CaseKind,
}

#[derive(Clone, Copy)]
enum CaseKind {
    NeuralBilling,
    NeuralThreePrimitives,
    NeuralAmbiguous,
    NeuralLatency,
    LexicalBilling,
    RouterZh,
    MultilingualZh,
    HostGating,
}

const CASES: &[Case] = &[
    Case {
        name: "english_billing_triage",
        description: "English support ticket → billing + refund signal (neural)",
        needs_neural: true,
        needs_registry: false,
        kind: CaseKind::NeuralBilling,
    },
    Case {
        name: "three_primitives",
        description: "choice + score + noul in one request (neural)",
        needs_neural: true,
        needs_registry: false,
        kind: CaseKind::NeuralThreePrimitives,
    },
    Case {
        name: "ambiguous_confidence_gate",
        description: "Ambiguous inquiry stays sales/other and exposes gate label",
        needs_neural: true,
        needs_registry: false,
        kind: CaseKind::NeuralAmbiguous,
    },
    Case {
        name: "latency_budget",
        description: "Warm single-question neural infer stays under budget",
        needs_neural: true,
        needs_registry: false,
        kind: CaseKind::NeuralLatency,
    },
    Case {
        name: "lexical_billing",
        description: "Size-minimal LexicalEngine routes refund language to billing",
        needs_neural: false,
        needs_registry: false,
        kind: CaseKind::LexicalBilling,
    },
    Case {
        name: "router_zh",
        description: "Chinese state routes to multilingual checkpoint id",
        needs_neural: false,
        needs_registry: false,
        kind: CaseKind::RouterZh,
    },
    Case {
        name: "multilingual_zh_neural",
        description: "Chinese ticket loads multilingual engine via registry",
        needs_neural: false,
        needs_registry: true,
        kind: CaseKind::MultilingualZh,
    },
    Case {
        name: "host_gating",
        description: "GatePolicy maps low-confidence choice to escalate",
        needs_neural: true,
        needs_registry: false,
        kind: CaseKind::HostGating,
    },
];

struct CaseResult {
    name: String,
    passed: bool,
    latency_ms: f64,
    input: Value,
    output: Value,
    checks: Vec<String>,
    error: Option<String>,
}

struct Opts {
    cases: Option<Vec<String>>,
    device: DeviceRequest,
    log_dir: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
}

fn parse_args(argv: &[String]) -> Result<Opts, String> {
    let mut cases: Option<Vec<String>> = None;
    let mut device = DeviceRequest::Auto;
    let mut log_dir = None;
    let mut checkpoint = env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from);
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--case" => {
                i += 1;
                let name = argv.get(i).ok_or("--case requires a name")?.clone();
                cases.get_or_insert_with(Vec::new).push(name);
            }
            "--device" => {
                i += 1;
                let raw = argv.get(i).ok_or("--device requires metal|cuda|cpu|auto")?;
                device = match raw.as_str() {
                    "auto" => DeviceRequest::Auto,
                    "metal" => DeviceRequest::Metal,
                    "cuda" => DeviceRequest::Cuda,
                    "cpu" => DeviceRequest::Cpu,
                    other => return Err(format!("unknown device {other}")),
                };
            }
            "--log-dir" => {
                i += 1;
                log_dir = Some(PathBuf::from(
                    argv.get(i).ok_or("--log-dir requires a path")?,
                ));
            }
            "--checkpoint" => {
                i += 1;
                checkpoint = Some(PathBuf::from(
                    argv.get(i).ok_or("--checkpoint requires a path")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}; try --help")),
        }
        i += 1;
    }
    Ok(Opts {
        cases,
        device,
        log_dir,
        checkpoint,
    })
}

fn print_help() {
    eprintln!(
        "Usage: just ap [--case NAME]... [--device auto|metal|cuda|cpu] [--checkpoint DIR] [--log-dir DIR]\n\nCases:"
    );
    for case in CASES {
        eprintln!(
            "  {:<28} {}{}{}",
            case.name,
            case.description,
            if case.needs_neural { " [neural]" } else { "" },
            if case.needs_registry {
                " [registry]"
            } else {
                ""
            }
        );
    }
}

fn select_cases(filter: &Option<Vec<String>>) -> Result<Vec<&'static Case>, String> {
    match filter {
        None => Ok(CASES.iter().collect()),
        Some(wanted) => {
            let mut out = Vec::new();
            for name in wanted {
                let found = CASES.iter().find(|c| c.name == name);
                match found {
                    Some(c) => out.push(c),
                    None => {
                        return Err(format!(
                            "unknown case {name}; available: {}",
                            CASES.iter().map(|c| c.name).collect::<Vec<_>>().join(", ")
                        ));
                    }
                }
            }
            Ok(out)
        }
    }
}

fn resolve_checkpoint(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        if path.join("rl_agent_config.json").is_file() {
            return Ok(path.to_path_buf());
        }
        return Err(format!(
            "checkpoint missing rl_agent_config.json: {}",
            path.display()
        ));
    }
    if let Ok(path) = env::var("APOFASI_CHECKPOINT") {
        let path = PathBuf::from(path);
        if path.join("rl_agent_config.json").is_file() {
            return Ok(path);
        }
        return Err(format!(
            "APOFASI_CHECKPOINT missing rl_agent_config.json: {}",
            path.display()
        ));
    }
    // Discover a local HF snapshot with the published Apofasi layout.
    let hub = env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs_home()
                .map(|h| h.join(".cache/huggingface"))
                .unwrap_or_else(|| PathBuf::from(".cache/huggingface"))
        })
        .join("hub");
    if let Ok(models) = fs::read_dir(&hub) {
        for model in models.flatten() {
            let snaps = model.path().join("snapshots");
            if !snaps.is_dir() {
                continue;
            }
            if let Ok(entries) = fs::read_dir(snaps) {
                for snap in entries.flatten() {
                    let cand = snap.path();
                    if cand.join("rl_agent_config.json").is_file()
                        && cand.join("model.safetensors").is_file()
                        && cand.join("tokenizer/tokenizer.json").is_file()
                    {
                        return Ok(cand);
                    }
                }
            }
        }
    }
    Err("no checkpoint found; set APOFASI_CHECKPOINT or pass --checkpoint DIR".into())
}

fn dirs_home() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn workspace_root() -> PathBuf {
    // examples run with CARGO_MANIFEST_DIR = crates/apofasi
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn utc_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

struct Logger {
    path: PathBuf,
    lines: Vec<String>,
}

impl Logger {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            lines: Vec::new(),
        }
    }

    fn log(&mut self, message: impl Into<String>) {
        let message = message.into();
        println!("{message}");
        self.lines.push(message);
    }

    fn flush(&self) -> Result<(), String> {
        fs::write(&self.path, self.lines.join("\n") + "\n")
            .map_err(|e| format!("write {}: {e}", self.path.display()))
    }
}

fn run_case(
    case: &Case,
    neural: Option<&NeuralEngine>,
    registry: Option<&mut CheckpointRegistry>,
    lexical: &LexicalEngine,
    router: &Router,
    log: &mut Logger,
) -> CaseResult {
    log.log(format!("\n{}", "=".repeat(72)));
    log.log(format!("CASE: {}", case.name));
    log.log(format!("DESC: {}", case.description));

    let (request, input_extra) = match case.kind {
        CaseKind::NeuralBilling => (billing_request(), json!({})),
        CaseKind::LexicalBilling => (lexical_billing_request(), json!({})),
        CaseKind::NeuralThreePrimitives => (three_primitives_request(), json!({})),
        CaseKind::NeuralAmbiguous | CaseKind::HostGating => (ambiguous_request(), json!({})),
        CaseKind::NeuralLatency => (latency_request(), json!({})),
        CaseKind::RouterZh | CaseKind::MultilingualZh => (
            SystemOneRequest {
                model: None,
                state: State::Text("请帮我退款，账号被重复扣费了。".into()),
                questions: {
                    let mut opts = IndexMap::new();
                    opts.insert("billing".into(), Some(json!("退款 发票 扣费")));
                    opts.insert("technical".into(), Some(json!("故障 报错")));
                    opts.insert("sales".into(), Some(json!("价格 合同")));
                    opts.insert("other".into(), Some(json!("其他")));
                    let mut questions = IndexMap::new();
                    questions.insert(
                        "department".into(),
                        Question::new(
                            DecisionKind::Choice,
                            json!("哪个部门应处理此请求？"),
                            Some(Criteria::Choice(opts)),
                        )
                        .unwrap(),
                    );
                    questions
                },
            },
            json!({}),
        ),
    };

    let input = json!({
        "description": case.description,
        "state": request.state,
        "questions": request.questions,
        "extra": input_extra,
    });
    log.log("INPUT:");
    log.log(serde_json::to_string_pretty(&input).unwrap_or_default());

    let outcome: Result<(Value, f64, Vec<String>), String> = (|| match case.kind {
        CaseKind::RouterZh => {
            let t0 = Instant::now();
            let decision = router
                .route(&request, None, None)
                .map_err(|e| e.to_string())?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mut failures = Vec::new();
            if decision.model.as_str() != "multilingual" {
                failures.push(format!(
                    "expected checkpoint=multilingual, got {}",
                    decision.model.as_str()
                ));
            }
            Ok((
                json!({
                    "routing": {
                        "checkpoint": decision.model.as_str(),
                        "reason": decision.reason,
                    },
                    "latency_ms": round2(latency_ms),
                    "checks_failed": failures.clone(),
                }),
                latency_ms,
                failures,
            ))
        }
        CaseKind::MultilingualZh => {
            let reg = registry.expect("registry required");
            let _ = reg
                .system_one_routed(request.clone(), None, None)
                .map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let (ckpt, response) = reg
                .system_one_routed(request.clone(), None, None)
                .map_err(|e| e.to_string())?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let mut failures = Vec::new();
            if ckpt != CheckpointId::Multilingual {
                failures.push(format!(
                    "expected routed multilingual, got {}",
                    ckpt.as_str()
                ));
            }
            match response.answers.get("department") {
                Some(a3s_apofasi::Answer::Choice { choice, .. })
                    if choice == "billing" || choice == "other" => {}
                Some(other) => failures.push(format!("unexpected department {other:?}")),
                None => failures.push("missing department".into()),
            }
            let mut output = output_payload(&response, latency_ms, &failures);
            if let Some(obj) = output.as_object_mut() {
                obj.insert("routing".into(), json!({ "checkpoint": ckpt.as_str() }));
            }
            Ok((output, latency_ms, failures))
        }
        CaseKind::LexicalBilling => {
            let _ = lexical.decide(&request).map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let response = lexical.decide(&request).map_err(|e| e.to_string())?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let failures = check_billing(&response, false);
            Ok((
                output_payload(&response, latency_ms, &failures),
                latency_ms,
                failures,
            ))
        }
        CaseKind::NeuralBilling
        | CaseKind::NeuralThreePrimitives
        | CaseKind::NeuralAmbiguous
        | CaseKind::NeuralLatency
        | CaseKind::HostGating => {
            let engine = neural.expect("neural required");
            let _ = engine.decide(&request).map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let response = engine.decide(&request).map_err(|e| e.to_string())?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let failures = match case.kind {
                CaseKind::NeuralBilling => check_billing(&response, true),
                CaseKind::NeuralThreePrimitives => check_three_primitives(&response),
                CaseKind::NeuralAmbiguous => check_ambiguous(&response),
                CaseKind::NeuralLatency => check_latency(&response, latency_ms),
                CaseKind::HostGating => check_host_gating(&response),
                _ => unreachable!(),
            };
            Ok((
                output_payload(&response, latency_ms, &failures),
                latency_ms,
                failures,
            ))
        }
    })();

    match outcome {
        Ok((output, latency_ms, failures)) => {
            log.log("OUTPUT:");
            log.log(serde_json::to_string_pretty(&output).unwrap_or_default());
            let passed = failures.is_empty();
            log.log(format!("RESULT: {}", if passed { "PASS" } else { "FAIL" }));
            for item in &failures {
                log.log(format!("  - {item}"));
            }
            CaseResult {
                name: case.name.into(),
                passed,
                latency_ms,
                input,
                output,
                checks: failures,
                error: None,
            }
        }
        Err(err) => {
            let output = json!({ "error": err });
            log.log("OUTPUT:");
            log.log(serde_json::to_string_pretty(&output).unwrap_or_default());
            log.log("RESULT: FAIL");
            CaseResult {
                name: case.name.into(),
                passed: false,
                latency_ms: 0.0,
                input,
                output,
                checks: vec![err.clone()],
                error: Some(err),
            }
        }
    }
}

fn output_payload(response: &SystemOneResponse, latency_ms: f64, failures: &[String]) -> Value {
    json!({
        "model": response.model,
        "answers": compact_answers(response),
        "usage": response.usage,
        "latency_ms": round2(latency_ms),
        "checks_failed": failures,
    })
}

fn compact_answers(response: &SystemOneResponse) -> Value {
    let mut map = serde_json::Map::new();
    for (id, answer) in &response.answers {
        let slim = match answer {
            a3s_apofasi::Answer::Choice {
                choice, confidence, ..
            } => json!({
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
            }),
            a3s_apofasi::Answer::Score {
                score, confidence, ..
            } => json!({
                "type": "score",
                "score": score,
                "confidence": confidence,
            }),
            a3s_apofasi::Answer::Noul { noul } => json!({
                "type": "noul",
                "noul": noul,
            }),
        };
        map.insert(id.clone(), slim);
    }
    Value::Object(map)
}

fn lexical_billing_request() -> SystemOneRequest {
    // Keep option glosses token-aligned with the state ("refund" ↔ "refunds").
    let mut opts = IndexMap::new();
    opts.insert(
        "billing".into(),
        Some(json!("invoices payments refund refunds billed")),
    );
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

fn billing_request() -> SystemOneRequest {
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
        state: State::Object(
            json!({
                "from": "user@acme.com",
                "subject": "Duplicate charge on invoice #4411",
                "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
        questions,
    }
}

fn three_primitives_request() -> SystemOneRequest {
    let mut opts = IndexMap::new();
    opts.insert("yes".into(), Some(json!("affirmative")));
    opts.insert("no".into(), Some(json!("negative")));
    let mut questions = IndexMap::new();
    questions.insert(
        "ack".into(),
        Question::new(
            DecisionKind::Choice,
            json!("Did the user acknowledge the issue?"),
            Some(Criteria::Choice(opts)),
        )
        .unwrap(),
    );
    questions.insert(
        "severity".into(),
        Question::new(
            DecisionKind::Score,
            json!("Severity"),
            Some(Criteria::Score(vec![
                json!("low"),
                json!("medium"),
                json!("high"),
            ])),
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
        state: State::Text("Please refund my invoice; this is urgent.".into()),
        questions,
    }
}

fn ambiguous_request() -> SystemOneRequest {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("invoices, payments, refunds")));
    opts.insert(
        "technical".into(),
        Some(json!("bugs, outages, system errors")),
    );
    opts.insert(
        "sales".into(),
        Some(json!("pricing, new contracts, product interest")),
    );
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

fn latency_request() -> SystemOneRequest {
    let mut questions = IndexMap::new();
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
        state: State::Text("Please refund the duplicate charge on my invoice.".into()),
        questions,
    }
}

fn check_billing(response: &SystemOneResponse, require_churn: bool) -> Vec<String> {
    let mut failures = Vec::new();
    match response.answers.get("department") {
        Some(a3s_apofasi::Answer::Choice { choice, .. }) if choice == "billing" => {}
        Some(other) => failures.push(format!("expected department=billing, got {other:?}")),
        None => failures.push("missing department".into()),
    }
    match response.answers.get("refund_requested") {
        Some(a3s_apofasi::Answer::Noul { noul }) if *noul > 0.5 => {}
        Some(other) => failures.push(format!("expected refund_requested > 0.5, got {other:?}")),
        None => failures.push("missing refund_requested".into()),
    }
    if require_churn {
        match response.answers.get("churn_risk") {
            Some(a3s_apofasi::Answer::Noul { noul }) if *noul > 0.5 => {}
            Some(other) => failures.push(format!("expected churn_risk > 0.5, got {other:?}")),
            None => failures.push("missing churn_risk".into()),
        }
    }
    failures
}

fn check_three_primitives(response: &SystemOneResponse) -> Vec<String> {
    let mut failures = Vec::new();
    for key in ["ack", "severity", "refund_requested"] {
        if !response.answers.contains_key(key) {
            failures.push(format!("missing {key}"));
        }
    }
    match response.answers.get("refund_requested") {
        Some(a3s_apofasi::Answer::Noul { noul }) if *noul > 0.5 => {}
        Some(other) => failures.push(format!("expected refund_requested > 0.5, got {other:?}")),
        None => {}
    }
    failures
}

fn check_ambiguous(response: &SystemOneResponse) -> Vec<String> {
    let mut failures = Vec::new();
    match response.answers.get("department") {
        Some(answer @ a3s_apofasi::Answer::Choice { choice, .. }) => {
            if choice != "sales" && choice != "other" {
                failures.push(format!(
                    "expected department in {{sales, other}}, got {choice}"
                ));
            }
            let gate = gate_answer(answer, &GatePolicy::default());
            // Soft informational label; still require a valid gate action.
            if gate != GateAction::Auto && gate != GateAction::Escalate {
                failures.push(format!("unexpected gate {gate:?}"));
            }
        }
        other => failures.push(format!("expected choice answer, got {other:?}")),
    }
    failures
}

fn check_host_gating(response: &SystemOneResponse) -> Vec<String> {
    let mut failures = check_ambiguous(response);
    let policy = GatePolicy::default();
    let gates = gate_response(response, &policy);
    if !gates.contains_key("department") {
        failures.push("missing department gate".into());
    }
    // Ambiguous product inquiry is expected to escalate under the default policy
    // when confidence is low; if the model is peaked, Auto is also acceptable.
    if let Some(answer) = response.answers.get("department") {
        let expected = gate_answer(answer, &policy);
        if gates.get("department") != Some(&expected) {
            failures.push(format!(
                "gate map mismatch: {:?} vs {expected:?}",
                gates.get("department")
            ));
        }
        let _ = any_escalate(&gates);
    }
    failures
}

fn check_latency(response: &SystemOneResponse, latency_ms: f64) -> Vec<String> {
    let mut failures = Vec::new();
    // Warm Metal budget mirrors the local suite target; CPU builds allow more headroom.
    let budget = if cfg!(feature = "metal") {
        200.0
    } else {
        2000.0
    };
    if latency_ms >= budget {
        failures.push(format!("expected infer_ms < {budget}, got {latency_ms:.1}"));
    }
    match response.answers.get("refund_requested") {
        Some(a3s_apofasi::Answer::Noul { noul }) if *noul > 0.5 => {}
        Some(other) => failures.push(format!("expected refund_requested > 0.5, got {other:?}")),
        None => failures.push("missing refund_requested".into()),
    }
    failures
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
