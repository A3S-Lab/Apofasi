//! One-process suite runner.
//!
//! Models stay loaded. Each case is warmed, then timed for `--iters` calls.
//! Reported `latency_ms` is the p50 of those samples (upper median, same index
//! as `len/2` after sorting). Choice-object key order is preserved by
//! `serde_json`'s `preserve_order` feature so option slots match the request file.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use a3s_apofasi::{Answer, CheckpointId, CheckpointRegistry, DeviceRequest, SystemOneRequest};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct SuiteFile {
    cases: Vec<SuiteCase>,
}

#[derive(Deserialize)]
struct SuiteCase {
    name: String,
    state: Value,
    questions: serde_json::Map<String, Value>,
    #[serde(default)]
    model: Option<String>,
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut cases_path: Option<PathBuf> = None;
    let mut checkpoint = std::env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from);
    let mut device = DeviceRequest::Auto;
    let mut warmup: usize = 1;
    let mut iters: usize = 1;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cases" => {
                i += 1;
                cases_path = Some(PathBuf::from(args.get(i).ok_or("--cases needs a path")?));
            }
            "--checkpoint" => {
                i += 1;
                checkpoint = Some(PathBuf::from(
                    args.get(i).ok_or("--checkpoint needs a path")?,
                ));
            }
            "--device" => {
                i += 1;
                let raw = args.get(i).ok_or("--device needs a value")?;
                device = match raw.as_str() {
                    "auto" => DeviceRequest::Auto,
                    "metal" => DeviceRequest::Metal,
                    "cuda" => DeviceRequest::Cuda,
                    "cpu" => DeviceRequest::Cpu,
                    other => return Err(format!("unknown device {other}")),
                };
            }
            "--warmup" => {
                i += 1;
                warmup = parse_count(args.get(i).ok_or("--warmup needs a value")?, "--warmup")?;
            }
            "--iters" => {
                i += 1;
                iters = parse_count(args.get(i).ok_or("--iters needs a value")?, "--iters")?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    let cases_path = cases_path.ok_or("suite requires --cases FILE")?;
    let checkpoint = checkpoint.ok_or("suite requires APOFASI_CHECKPOINT or --checkpoint")?;
    let raw = fs::read_to_string(&cases_path)
        .map_err(|e| format!("read {}: {e}", cases_path.display()))?;
    let file: SuiteFile = serde_json::from_str(&raw).map_err(|e| format!("parse cases: {e}"))?;

    let mut registry =
        CheckpointRegistry::open(&checkpoint, device, 3).map_err(|e| e.to_string())?;
    registry
        .preload(&[
            CheckpointId::English,
            CheckpointId::Multilingual,
            CheckpointId::TypedDecisions,
        ])
        .map_err(|e| e.to_string())?;

    let mut rows = Vec::with_capacity(file.cases.len());
    for case in file.cases {
        let request = request_from_case(&case)?;
        let model = case.model.as_deref();
        for _ in 0..warmup {
            registry
                .system_one_routed(request.clone(), model, None)
                .map_err(|e| format!("{}: {e}", case.name))?;
        }
        let mut samples = Vec::with_capacity(iters);
        let mut last = None;
        for _ in 0..iters {
            let t0 = Instant::now();
            let timed = registry
                .system_one_routed(request.clone(), model, None)
                .map_err(|e| format!("{}: {e}", case.name))?;
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            last = Some(timed);
        }
        let (ckpt, response) = last.ok_or_else(|| format!("{}: no samples", case.name))?;
        let p50 = percentile_50(&samples);
        let min_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max_ms = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        rows.push(json!({
            "name": case.name,
            "latency_ms": round_ms(p50),
            "p50_ms": round_ms(p50),
            "min_ms": round_ms(min_ms),
            "max_ms": round_ms(max_ms),
            "warmup": warmup,
            "iters": iters,
            "routing": { "model": ckpt.as_str() },
            "answers": compact_answers(&response.answers),
            "usage": response.usage,
        }));
    }
    println!(
        "{}",
        serde_json::to_string(&rows).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn parse_count(raw: &str, flag: &str) -> Result<usize, String> {
    let count: usize = raw
        .parse()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if count == 0 {
        return Err(format!("{flag} must be a positive integer"));
    }
    Ok(count)
}

fn percentile_50(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ordered[ordered.len() / 2]
}

fn round_ms(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn request_from_case(case: &SuiteCase) -> Result<SystemOneRequest, String> {
    let mut body = serde_json::Map::new();
    body.insert("state".into(), case.state.clone());
    body.insert("questions".into(), Value::Object(case.questions.clone()));
    if let Some(model) = &case.model {
        body.insert("model".into(), Value::String(model.clone()));
    }
    serde_json::from_value(Value::Object(body)).map_err(|e| format!("{}: {e}", case.name))
}

fn compact_answers(answers: &indexmap::IndexMap<String, Answer>) -> Value {
    let mut map = serde_json::Map::new();
    for (id, answer) in answers {
        let slim = match answer {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => json!({
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }),
            Answer::Score {
                score, confidence, ..
            } => json!({
                "type": "score",
                "score": score,
                "confidence": confidence,
            }),
            Answer::Noul { noul } => {
                let confidence = (noul.max(1.0 - noul) * 10_000.0).round() / 10_000.0;
                json!({
                    "type": "noul",
                    "noul": noul,
                    "confidence": confidence,
                })
            }
        };
        map.insert(id.clone(), slim);
    }
    Value::Object(map)
}
