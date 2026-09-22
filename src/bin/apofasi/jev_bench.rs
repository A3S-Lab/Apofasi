//! Official jev-benchmarks pilot manifest, one model load, no sample cut.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use a3s_apofasi::{
    Answer, CheckpointRegistry, Criteria, DecisionKind, DeviceRequest, Question, State,
    SystemOneRequest,
};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};

const DEFAULT_QUESTION: &str = "Which single label best describes the input text?";

pub fn run(args: &[String]) -> Result<(), String> {
    let opts = parse(args)?;
    let rows = read_jsonl(&opts.manifest)?;
    if rows.is_empty() {
        return Err(format!("manifest is empty: {}", opts.manifest.display()));
    }
    let done = completed_ids(&opts.output)?;
    let pending: Vec<&Value> = rows
        .iter()
        .filter(|row| {
            row.get("example_id")
                .and_then(Value::as_str)
                .is_some_and(|id| !done.contains(id))
        })
        .collect();
    eprintln!(
        "jev-bench manifest={} pending={} completed={}",
        rows.len(),
        pending.len(),
        done.len()
    );
    if pending.is_empty() {
        return Ok(());
    }

    drop_failed_rows(&opts.output)?;
    let mut registry = CheckpointRegistry::open(&opts.checkpoint, opts.device, 3)
        .map_err(|err| err.to_string())?;
    let warmup = example_request(pending[0], &opts.question)?;
    registry
        .system_one_routed(warmup, Some(&opts.model), None)
        .map_err(|err| format!("warmup failed: {err}"))?;

    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&opts.output)
        .map_err(|err| format!("open {}: {err}", opts.output.display()))?;

    for (index, row) in pending.iter().enumerate() {
        let example_id = row
            .get("example_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let request = example_request(row, &opts.question)?;
        let started = Instant::now();
        let scored = registry.system_one_routed(request, Some(&opts.model), None);
        let latency_seconds = started.elapsed().as_secs_f64();
        let record = match scored {
            Ok((_route, response)) => success_record(
                &opts.experiment_id,
                &opts.model,
                row,
                &response,
                latency_seconds,
            )?,
            Err(err) => json!({
                "experiment_id": opts.experiment_id,
                "backend": "apofasi",
                "model_requested": opts.model,
                "model_resolved": "unknown",
                "dataset": row.get("dataset").and_then(Value::as_str).unwrap_or(""),
                "example_id": example_id,
                "target_index": row.get("target_index").and_then(Value::as_i64).unwrap_or(-1),
                "predicted_index": -1,
                "labels": row.get("labels").cloned().unwrap_or(Value::Array(Vec::new())),
                "probabilities": [],
                "latency_seconds": latency_seconds,
                "error": err.to_string(),
            }),
        };
        writeln!(out, "{record}").map_err(|err| format!("write prediction: {err}"))?;
        out.flush()
            .map_err(|err| format!("flush prediction: {err}"))?;
        eprintln!("[apofasi] {}/{} {example_id}", index + 1, pending.len());
    }
    Ok(())
}

fn success_record(
    experiment_id: &str,
    model: &str,
    row: &Value,
    response: &a3s_apofasi::SystemOneResponse,
    latency_seconds: f64,
) -> Result<Value, String> {
    let labels = row
        .get("labels")
        .and_then(Value::as_array)
        .ok_or("manifest row is missing labels")?;
    let answer = response
        .answers
        .get("label")
        .ok_or("response is missing the label answer")?;
    let Answer::Choice { probabilities, .. } = answer else {
        return Err("label answer is not a choice".into());
    };
    let mut probs = Vec::with_capacity(labels.len());
    for index in 0..labels.len() {
        let key = format!("label_{index:03}");
        let probability = probabilities
            .get(&key)
            .copied()
            .ok_or_else(|| format!("choice distribution is missing {key}"))?;
        probs.push(probability);
    }
    let predicted_index = probs
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, probability)| {
            if *probability > best.1 {
                (index, *probability)
            } else {
                best
            }
        })
        .0;
    Ok(json!({
        "experiment_id": experiment_id,
        "backend": "apofasi",
        "model_requested": model,
        "model_resolved": response.model,
        "dataset": row.get("dataset").and_then(Value::as_str).unwrap_or(""),
        "example_id": row.get("example_id").and_then(Value::as_str).unwrap_or(""),
        "target_index": row.get("target_index").and_then(Value::as_i64).unwrap_or(-1),
        "predicted_index": predicted_index,
        "labels": labels,
        "probabilities": probs,
        "latency_seconds": latency_seconds,
        "input_tokens": response.usage.input_tokens,
        "probability_sum_raw": probs.iter().sum::<f32>(),
    }))
}

fn example_request(row: &Value, question: &str) -> Result<SystemOneRequest, String> {
    let text = row
        .get("text")
        .and_then(Value::as_str)
        .ok_or("manifest row is missing text")?;
    let labels = row
        .get("labels")
        .and_then(Value::as_array)
        .ok_or("manifest row is missing labels")?;
    let mut criteria = IndexMap::new();
    for (index, label) in labels.iter().enumerate() {
        let description = label.as_str().ok_or("label hypothesis must be a string")?;
        criteria.insert(format!("label_{index:03}"), Some(json!(description)));
    }
    let mut questions = IndexMap::new();
    questions.insert(
        "label".to_string(),
        Question::new(
            DecisionKind::Choice,
            json!(question),
            Some(Criteria::Choice(criteria)),
        )
        .map_err(|err| err.to_string())?,
    );
    let mut state = Map::new();
    state.insert("text".to_string(), Value::String(text.to_string()));
    Ok(SystemOneRequest {
        model: None,
        state: State::Object(state),
        questions,
    })
}

struct Opts {
    manifest: PathBuf,
    output: PathBuf,
    checkpoint: PathBuf,
    device: DeviceRequest,
    model: String,
    question: String,
    experiment_id: String,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut manifest = None;
    let mut output = None;
    let mut checkpoint = std::env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from);
    let mut device = DeviceRequest::Auto;
    let mut model = "english".to_string();
    let mut question = DEFAULT_QUESTION.to_string();
    let mut experiment_id = "btzsc-pilot-v1".to_string();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = || {
            args.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
            "--model" => model = value()?,
            "--question" => question = value()?,
            "--experiment" => experiment_id = value()?,
            "--device" => {
                device = match value()?.as_str() {
                    "auto" => DeviceRequest::Auto,
                    "metal" => DeviceRequest::Metal,
                    "cuda" => DeviceRequest::Cuda,
                    "cpu" => DeviceRequest::Cpu,
                    other => return Err(format!("unknown device {other}")),
                };
            }
            other => return Err(format!("unknown flag {other}")),
        }
        index += 2;
    }
    Ok(Opts {
        manifest: manifest.ok_or("--manifest is required")?,
        output: output.ok_or("--output is required")?,
        checkpoint: checkpoint.ok_or("--checkpoint or APOFASI_CHECKPOINT is required")?,
        device,
        model,
        question,
        experiment_id,
    })
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<Value>, String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("read {}: {err}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|err| format!("{}:{}: {err}", path.display(), line_no + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn drop_failed_rows(path: &PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut kept = String::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line)
            .map_err(|err| format!("{}:{}: {err}", path.display(), line_no + 1))?;
        if row.get("error").is_some_and(|error| !error.is_null()) {
            continue;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    std::fs::write(path, kept).map_err(|err| format!("rewrite {}: {err}", path.display()))
}

fn completed_ids(path: &PathBuf) -> Result<std::collections::HashSet<String>, String> {
    let mut done = std::collections::HashSet::new();
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(done),
        Err(err) => return Err(format!("open {}: {err}", path.display())),
    };
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|err| format!("read {}: {err}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|err| format!("parse {}: {err}", path.display()))?;
        if row.get("error").is_some_and(|error| !error.is_null()) {
            continue;
        }
        if let Some(id) = row.get("example_id").and_then(Value::as_str) {
            done.insert(id.to_string());
        }
    }
    Ok(done)
}
