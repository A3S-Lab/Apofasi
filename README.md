# Apofasi

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

Typed decisions in one forward pass. No generation, nothing to parse.

*Apófasi* (απόφαση) means **decision**. A host gives Apofasi a state and a
map of typed questions. Apofasi returns a map of typed answers: `choice`,
`score`, or `noul` (yes/no). The JSON contract matches TypeSafe
[System One (Jev)](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
so a host can speak one schema to either engine.

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
Roadmap: [`ROADMAP.md`](ROADMAP.md)

## Why this exists

A decision is not a completion. The host already knows the question type,
the options, and the score scale. A generative model spends a network
round trip and a decode loop to emit text the host must then parse.
Apofasi scores the typed questions directly and returns the answer object.

Default builds pull no ML framework. Neural inference is an opt-in feature.

## Use

```rust
use a3s_apofasi::{Client, Criteria, DecisionKind, Question, State, SystemOneRequest};
use indexmap::IndexMap;
use serde_json::json;

fn main() -> a3s_apofasi::Result<()> {
    let mut opts = IndexMap::new();
    opts.insert("billing".into(), Some(json!("refunds")));
    opts.insert("other".into(), Some(json!("else")));
    let mut questions = IndexMap::new();
    questions.insert(
        "department".into(),
        Question::new(
            DecisionKind::Choice,
            json!("Which department?"),
            Some(Criteria::Choice(opts)),
        )?,
    );
    let res = Client::default().system_one(SystemOneRequest {
        model: None,
        state: State::Text("Please refund my invoice.".into()),
        questions,
    })?;
    let _ = res;
    Ok(())
}
```

`Client::default` is the lexical engine: pure Rust, no weights. Enable
`infer` for the neural checkpoint. On Apple Silicon, `metal` and `mlx`
select the GPU path; `mlx` is the fast path and links a prebuilt
`libmlx` via `MLX_ROOT`. On NVIDIA hosts use `cuda`. For warm CPU
latency, enable `ort` and place `encoder.onnx` (or `encoder.opt.onnx`)
next to the checkpoint — see `scripts/ort_encoder_probe.py`. Optional
`mkl` helps the Candle head; Windows also needs `libiomp5md.dll` next to
the binary.

```bash
cargo build --release
cargo build --release --features cli,metal,mlx --bin a3s-apofasi
cargo build --release --features cli,cuda,mkl,ort --bin a3s-apofasi
```

CUDA loads `bf16` when the checkpoint `amp_dtype` says so (override with
`APOFASI_DTYPE`). Warm triage on an RTX 4090 is ~22–28 ms p50. CPU ORT
warm triage is ≤ 500 ms when the ONNX encoder is present. Set
`APOFASI_PROFILE=1` to split `pack_ms` / `fwd_ms`. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for targets and evidence.
## Versus Jev

Apofasi 0.1.1 was measured on Apple Silicon with the MLX forward path.
Jev was **not** run on this machine. Jev figures below are the published
hosted-API numbers, and they include the network hop. Apofasi figures are
warm on-device time. The speed column is how many times faster that local
time is than the published Jev p50. It is not a paired replay on one machine.

The accuracy rows are the unmodified
[jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) `pilot-v1`
manifest: 100 examples each of AG News, Banking77, and DAIR Emotion, zero
failures. The benchmark repository and its config were not edited. Apofasi
saw the same question text and the same `label_000`… option keys as that
package's Jev adapter. Probabilities were scored by its `group_scores`
(10 ECE bins, 5% error budget). Checkpoint weights were not changed.

| Dataset | Apofasi accuracy | Jev accuracy | Apofasi p50 | Jev p50 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG News | 0.960 | 0.910 | 9.8 ms | 255.9 ms | 26.1× |
| DAIR Emotion | 0.420 | 0.480 | 13.1 ms | 236.3 ms | 18.0× |
| Banking77 | 0.560 | 0.870 | 76.1 ms | 246.4 ms | 3.2× |

AG News is ahead on accuracy and 26.1× faster. Its Brier is 0.098 against
Jev's 0.146; its ECE is 0.137 against Jev's 0.064. DAIR Emotion is 0.060
behind on accuracy (Brier 0.976 vs 0.846, ECE 0.404 vs 0.351) and 18.0×
faster. Its six labels already fit in one forward. Banking77 has 72 labels.
One forward used to cut every option down to the same token prefix
(accuracy 0.020, true-label probability 0 on 62% of examples). Choices that
do not fit `head_max_len` are now split into groups that do, then the group
winners are compared. Accuracy is 0.560, Brier 0.599, ECE 0.134, and no true
label is given probability 0. That is still 0.310 behind published Jev, and
just behind published GLiNER2.5 (0.610 accuracy, 295.5 ms p50). The extra
forwards make this row 3.2× rather than a single-pass ratio. Published
GLiNER2.5 accuracy / p50 on the other two sets: AG News 0.700 / 44.9 ms,
DAIR Emotion 0.440 / 43.3 ms.

Published Jev latency outside this pilot is 236–256 ms p50 on 4- and 6-label
tasks in jev-benchmarks, and 264–276 ms p50 in
[decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark).
Taken together, one published Jev question is **236–276 ms**.

## Gate before a generative call

Use Apofasi when the host already knows the question type and would otherwise
spend a full completion to pick a label, a score, or a yes/no. Run the typed
request first, then apply the host gate. `GatePolicy::default()` keeps an
answer on `Auto` only when choice/score confidence is at least 0.7, or when
noul extremity `max(noul, 1 - noul)` is at least 0.7. If every answer is
`Auto`, keep the typed answer and do not call the generative model. If any
answer is `Escalate`, call that model at most once. Its text is host
evidence. It does not become the typed `Answer`.

```rust
let response = engine.decide(&request)?;
let gates = a3s_apofasi::gate_response(&response, &a3s_apofasi::GatePolicy::default());
if a3s_apofasi::any_escalate(&gates) {
    // One host-side generation. Do not parse its prose into an Answer.
} else {
    // Use response.answers. This path makes zero generations.
}
```

`Client::default` is the lexical engine: no weights, suitable for tests.
On the six tasks below its overlap scores stayed under 0.7, so the gate
always escalated and the generative call was not saved. Do not lower the
threshold to force that path onto `Auto`; that accepts an uncertain overlap
score.

The neural english checkpoint is what can clear 0.7. Load it with `infer`
(`NeuralEngine::load` or `load_with`). On Apple Silicon pass `metal` or
`mlx`; `mlx` is the fast path when `MLX_ROOT` is set. Point
`APOFASI_CHECKPOINT` at the bundle root and let the router select `english`
for English state text.

The table is one paired run of those six single-question tasks, not a
benchmark suite. Lexical times are debug-build medians (5 warmup, 50
calls). Neural times are release Candle Metal on an Apple M5 Max: the
published english checkpoint, 3 warmup calls, then the p50 of 20 calls
(`sorted[len/2]`). The generative column is one hosted DeepSeek V4.1 Flash
completion per task (128 output-token cap, 90 s timeout), prompt plus
completion tokens. It was not re-run beside the neural samples. Neural
encoder usage on these rows was 40–57 input tokens and 8 output tokens per
forward; that is not the generative token column. Labels are the gate
signal, not a claim that they match the generative model.

| Task | Lexical | Neural p50 | Neural gate | Generative model |
| --- | --- | ---: | --- | --- |
| Refund route | 97 µs, escalate (billing 0.11) | 16.1 ms | **Auto** (billing 0.87) | 2552 ms, 152/56 |
| Checkout outage | 81 µs, escalate (noul 0.62) | 15.7 ms | **Auto** (noul 0.85) | 2839 ms, 144/115 |
| Secret debug diff | 83 µs, escalate (noul 0.62) | 16.1 ms | Escalate (noul 0.42) | 2483 ms, 152/93 |
| Search command | 87 µs, escalate (read-only 0.11) | 16.2 ms | Escalate (read-only 0.16) | 2301 ms, 151/69 |
| Clean test command | 89 µs, escalate (mutate 0.11) | 16.0 ms | Escalate (mutate 0.55) | 2573 ms, 148/91 |
| Force push | 78 µs, escalate (score 1.00, 0.00) | 15.1 ms | Escalate (score 1.15, 0.03) | 3495 ms, 122/128 |

Two of the six neural answers cleared 0.7 and skipped the completion: about
5.4 s and 467 tokens, at about 16 ms each. The other four still escalate, so
the neural forward is about 16 ms added before the same completion, not a
saving. The speedup exists only on `Auto`. A lexical forward is much
shorter, but on this set it never reached `Auto`, so it saved nothing.

```bash
cargo run --release --features cli,metal --bin a3s-apofasi -- suite \
  --cases cases.json --checkpoint "$APOFASI_CHECKPOINT" \
  --device metal --warmup 3 --iters 20
```

The seven cases below are a separate local suite. Each was warmed 12 times,
then timed for 40 calls. The figure is the p50 (upper median:
`sorted[len/2]`). They have no published Jev accuracy to put beside them.
All seven passed the answer checks used while recording the samples. Only
the two 1-question rows match Jev's published "one decision" shape.

| Case | Questions | Apofasi p50 | Versus 236–276 ms |
| --- | ---: | ---: | --- |
| Single refund question | 1 | 7.02 ms | 33.6×–39.3× |
| Ambiguous department choice | 1 | 7.32 ms | 32.2×–37.7× |
| Three primitives | 3 | 8.77 ms | 26.9×–31.5× |
| English billing triage | 4 | 12.57 ms | 18.8×–22.0× |
| Chinese billing triage | 4 | 5.04 ms | 46.8×–54.8× |
| Guard preset | 5 | 12.76 ms | 18.5×–21.6× |
| Explicit typed checkpoint | 4 | 12.82 ms | 18.4×–21.5× |

## Checkpoint

Neural runs need this tree (`APOFASI_CHECKPOINT`):

```text
checkpoint/
├── rl_agent_config.json
├── model.safetensors
├── encoder/config.json
└── tokenizer/tokenizer.json
```

A bundle may nest `english/`, `multilingual/`, and `typed-decisions/`
under one root. The router picks a checkpoint before the forward pass.
See [`docs/publish-layout.md`](docs/publish-layout.md).

## Crate

| Item | Value |
| --- | --- |
| Package | `a3s-apofasi` |
| Version | 0.1.1 |
| Repository | [A3S-Lab/Apofasi](https://github.com/A3S-Lab/Apofasi) |
| License | MIT |

## License

MIT © A3S Lab
