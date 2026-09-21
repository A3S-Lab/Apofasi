# Apofasi

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
`libmlx` via `MLX_ROOT`.

```bash
cargo build --release
cargo build --release --features cli,metal,mlx --bin a3s-apofasi
```

## Latency versus Jev

Apofasi 0.1.0 was measured on Apple Silicon with the MLX forward path.
Checkpoints stayed loaded. Each case was warmed 12 times, then timed for
40 calls. The figure is the p50 (upper median: `sorted[len/2]`).

Jev was **not** run on this machine. The band below is the published
hosted-API p50 for one typed decision:

- [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks): Jev 1.13.0,
  236–256 ms p50 on 4- and 6-label tasks, 246 ms p50 at 72 labels.
  Latency includes the network hop from the benchmark client.
- [decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark):
  Jev p50 264–276 ms, flat from 2 to 255 options.

Taken together, published Jev p50 for one question is **236–276 ms**.
Apofasi's numbers are on-device forward time. The ratio is how much less
time the local call took, not a paired replay of those studies.

| Case | Questions | Apofasi p50 | Versus 236–276 ms |
| --- | ---: | ---: | --- |
| Single refund question | 1 | 7.02 ms | 33.6×–39.3× |
| Ambiguous department choice | 1 | 7.32 ms | 32.2×–37.7× |
| Three primitives | 3 | 8.77 ms | 26.9×–31.5× |
| English billing triage | 4 | 12.57 ms | 18.8×–22.0× |
| Chinese billing triage | 4 | 5.04 ms | 46.8×–54.8× |
| Guard preset | 5 | 12.76 ms | 18.5×–21.6× |
| Explicit typed checkpoint | 4 | 12.82 ms | 18.4×–21.5× |

Only the two 1-question rows match Jev's published "one decision" shape.
The other rows finish a heavier call still under that 1-question band.
All seven cases passed the same answer checks used to record the samples.

This table is not an accuracy comparison. Jev's published label accuracy
(AG News 0.910, Banking77 0.870, DAIR Emotion 0.480 in jev-benchmarks)
was measured on those datasets. Apofasi 0.1.0 was not scored on them.

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
| Version | 0.1.0 |
| Repository | [A3S-Lab/Apofasi](https://github.com/A3S-Lab/Apofasi) |
| License | MIT |

## License

MIT © A3S Lab
