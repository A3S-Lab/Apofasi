# Apofasi Architecture

Apofasi is A3S's typed **System-1** decision engine. The stack is entirely
**Rust** (no Python runtime, no `libtorch`): non-autoregressive typed
primitives, marker scoring, strictly proper scoring-rule training, and
script-aware routing before any forward pass.

*Apófasi* (απόφαση) means **decision**.

## First principles

| Principle | Why it exists | Implication |
| --- | --- | --- |
| No text generation | Hosts need structured answers, not parseable prose | Outputs are typed enums/structs, never free-form strings |
| One forward pass per question batch | Latency is the product | Batch all questions; single encoder call |
| Strictly proper scores (RLCD) | Confidence must be usable for gating | Train/eval math lives in core, independent of tensors |
| Route before inference | English-only encoders collapse on non-Latin while staying confident | Pure-Rust script/lang detect; no model call for routing |
| Minimal core + swappable backends | Devices and encoder families change | Traits for `Encoder`, `Tokenizer`, `DecisionEngine` |
| Deterministic checkpoint layout | Hosts and converters share one on-disk contract | Config JSON + safetensors + tokenizer tree |

What Apofasi is **not**:

- Not a generative LLM wrapper.
- Not a replacement for A3S Flow (orchestration) or Use (capabilities).
- Not a cloud-only service — the library must run in-process on Desktop/CLI/Code hosts.

## Layered crate shape

Single repository. The root package `a3s-apofasi` is the facade hosts depend on.
Heavy ML stays behind features so Code/CLI can compile the type surface without
Metal/CUDA toolchains when unused.

```text
a3s-apofasi/                         # facade: Agent, Router, presets
├── core (in-tree modules)           # ALWAYS pure Rust, no ML framework
│   ├── primitive                    # Choice / Score / Noul
│   ├── schema                       # Question, Criteria, State
│   ├── sequence                     # prompt pack + marker positions
│   ├── confidence                   # entropy confidence, ECE helpers
│   ├── calibration                  # temperature + option-count buckets
│   └── detect                       # script / Latin-language heuristics
├── model               (feature)    # DecisionNet + Candle ModernBERT/mmBERT
├── runtime             (feature)    # load, batch, decide, device policy
├── router                           # checkpoint selection (pure; uses detect)
├── train               (feature)    # RLCD / GRPO-style policy gradient loop
└── cli                 (optional)   # local smoke / bench / convert
```

Dependency direction (never reverse):

```text
cli → facade → runtime → model → core
                 ↘ router → core
train → model → core
```

### Core contracts (stable, Jev-aligned)

Public JSON matches TypeSafe System One / Jev:

**Request** — `POST`-shaped body (in-process or HTTP):

```json
{
  "model": "apofasi-0.1.1",
  "state": "string | object | array",
  "questions": {
    "<id>": {
      "type": "choice | score | noul",
      "instructions": "string | object | array",
      "criteria": { }
    }
  }
}
```

| `type` | `criteria` | Answer fields |
| --- | --- | --- |
| `choice` | object: option → description (`null` allowed), ≤255 | `choice`, `probabilities`, `confidence` |
| `score` | ordered array of level descriptions | `score`, `legend`, `probabilities`, `confidence` |
| `noul` | optional `{ "true", "false" }` glosses | `noul` only (no `confidence`) |

**Response**:

```json
{
  "model": "apofasi-0.1.1",
  "answers": { "<id>": { "type": "...", "...": "..." } },
  "usage": { "input_tokens": 0, "output_tokens": 0 }
}
```

Rust types: [`SystemOneRequest`], [`SystemOneResponse`], [`Question`], [`Answer`].

Hosts never see logits. For Choice/Score, use `confidence` to gate automation;
for Noul, the `noul` probability itself is the signal (near 0 or 1 = confident).
## Inference architecture

### Sequence layout

```text
[CLS]  "{kind} question: {instructions}"  [SEP]
[MASK] opt0  [MASK] opt1  …  [SEP]
{serialized state}  [SEP]
```

- Head budget: `head_max_len` (options + instructions).
- State budget: `max_len - head_len - 1`.
- Markers: absolute token indices of each `[MASK]`.
- High-cardinality choices that do not fit in `head_max_len` are split into
  interleaved groups that do. Each group is one forward with the full option
  text. Crowded groups also carry a runner-up (and a close third place) into
  the next comparison. When the composed leaders are close, those leaders get
  one joint forward. A choice that already fits is still one forward. The same
  state token ids are reused across group packs; encoder states are not.

### Network

```text
input_ids, attention_mask
        │
        ▼
┌───────────────────┐
│ Encoder backbone  │  ModernBERT-large (en) / mmBERT-base (multi)
│ bidirectional     │
└─────────┬─────────┘
          │ last_hidden_state H  [B, L, D]
          │ + type_emb(qtype) broadcast
          ▼
┌───────────────────┐
│ Decision head     │  TransformerEncoder × head_layers (norm_first)
└─────────┬─────────┘
          │
    gather H at marker_pos
          │
          ▼
     scorer → logits [B, K]     (masked invalid options → -1e4)
          │
          ├─ softmax / temperature → option probabilities
          │
          └─ feats(top1, margin, entropy, K/255) ⊕ H[:,0]
                    │
                    ▼
               act_head → P(act) / P(escalate)
```

### Device policy

| Preference | Backend |
| --- | --- |
| Apple Silicon | MLX when feature `mlx` is on and the device is Metal; otherwise Candle Metal |
| NVIDIA | Candle CUDA (when enabled) |
| Fallback | Candle CPU (f32) |

Runtime resolves device once at load, with explicit override
(`DeviceRequest::{Auto, Metal, Cuda, Cpu}`). OOM on accelerator falls back to
CPU with a structured warning.

### Checkpoint layout

```text
checkpoint/
├── rl_agent_config.json      # max_len, head_max_len, temperatures, encoder id
├── model.safetensors         # encoder.* + type_emb.* + head.* + scorer.* + act_head.*
├── tokenizer/                # tokenizers JSON (HF fast)
└── encoder/config.json       # backbone geometry for from-config build
```

Loaders enforce **strict name/shape checks**. Training publishes the same
layout so hosts stay checkpoint-stable across releases.

## Router

Routing is a **pure function of state (+ optional question ids)** and must not
load a model.

Precedence:

1. Explicit `model=`
2. Explicit `task=` / typed-decisions workflow id match (opt-in)
3. Explicit `lang=`
4. Detected script / English-vs-other Latin
5. Default (`english`)

Checkpoints:

| Name | Backbone | Context | Role |
| --- | --- | --- | --- |
| `english` | ModernBERT-large | 512 | English System-1 |
| `multilingual` | mmBERT-base | 1024 | 100+ languages |
| `typed-decisions` | ModernBERT-large | 1024 | Fine-tuned workflow pack |

Resident set is LRU-capped (`max_loaded`). Servers call `preload([...])` so
language flips never pay cold load.

## Training (RLCD)

Feature `train`. Pure scoring math stays in core so unit tests do not need a
GPU.

Strictly proper reward:

```text
R = log_score(q, target)
  + w_sph * spherical(q, target)
  - w_rps * RPS(q, target)          # score questions only
```

Policy updates follow a GRPO-style gradient on the decision head (and optional
encoder LoRA later). Calibration temperatures are fit **after** training per
`(DecisionKind, option_bucket)` and stored in `rl_agent_config.json`.

Product rule: base checkpoints are a fast specialization base; domain accuracy
comes from fine-tuning, not zero-shot magic.

## Host integration (A3S)

```text
┌──────────── Desktop / Code / CLI / Cloud ────────────┐
│  ACL schemas · Flow steps · Use tools · UI gates     │
└─────────────────────┬────────────────────────────────┘
                      │ SystemOneRequest / SystemOneResponse
                      ▼
               a3s-apofasi (facade)
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Router        Agent         Presets
        │             │
        ▼             ▼
     detect        runtime (Candle)
                      │
                      ▼
                   DecisionNet
```

Ownership:

| Concern | Owner |
| --- | --- |
| Typed decide API, calibration, routing | Apofasi |
| When to call, thresholds, HITL | Host (Code/Desktop/Cloud) |
| Package install / env identity | Use |
| Multi-step workflows | Flow |
| Generative answers | Code model path (not Apofasi) |

Wire events (when emitted): `apofasi.decision.completed`,
`apofasi.route.selected`, `apofasi.calibration.fitted` — lowercase dot keys.

## Feature flags

| Feature | Default | Contents |
| --- | --- | --- |
| `router` | on | Router over script/language detection |
| `infer` | off | Candle `DecisionNet` + runtime |
| `metal` | off | Candle Metal |
| `mlx` | off | Apple MLX decision forward (prebuilt `libmlx` via `MLX_ROOT`) |
| `cuda` | off | Candle CUDA |
| `mkl` | off | Candle CPU matmul via Intel MKL |
| `ort` | off | ONNX Runtime ModernBERT encoder on CPU. FP32 `encoder.opt.onnx`, else `encoder.onnx`. INT8 only when `APOFASI_ORT_QUANT=1` |
| `train` | off | RLCD loop + dataset IO |
| `cli` | off | `a3s-apofasi` binary |

`default = ["router"]`. Types, sequence packing, detection, and confidence
compile in every build; they are not a separate feature. Desktop ships
`infer` + `metal` on macOS. Darwin `just ap*` also enables `mlx` when a
prebuilt MLX package is available.

## Performance targets

| Path | Target (warm, 4 questions) | Notes |
| --- | --- | --- |
| Apple Silicon (`mlx`) | ≤ 80 ms p50 | Fused MLX kernels; f32 weights so logits match Candle |
| Metal without `mlx` | ≤ 80 ms p50 | Candle Metal SDPA encoder; skip unused act head on decide |
| CUDA (Ampere+) | ≤ 40 ms p50 | Candle CUDA; loads `bf16` when checkpoint `amp_dtype` says so (`APOFASI_DTYPE` overrides). Mask cache is device-wide. |
| CPU | ≤ 500 ms p50 | Feature `ort` + `encoder.onnx` (prefer `encoder.opt.onnx`). Candle+MKL alone cannot hit the gate on ModernBERT-large. |

Batching N questions in one forward pass is mandatory for the hot path.
The 0.1.1 Apple Silicon measurements are in the README, next to the published
hosted Jev accuracy and latency. Use `APOFASI_PROFILE=1` to split `pack_ms` /
`fwd_ms` on any device.

### Scale 1 / Scale 3 evidence (this Windows host, warm triage)

| Build / path | p50 | Gate |
| --- | ---: | --- |
| CUDA `APOFASI_DTYPE=bf16` english-large | 22.98 ms | ≤ 40 ms PASS; smoke answers match CPU ORT and CUDA f32 |
| CUDA `APOFASI_DTYPE=f32` english-large | 22.38 ms | ≤ 40 ms PASS |
| CUDA BF16 example suite | 8/8 PASS | answer checks + latency_budget |
| CPU Candle+MKL english-large | ~1.3–1.6 s | ≤ 500 ms FAIL |
| CPU ORT FP32 multilingual (`encoder.onnx`) | ~230–285 ms | ≤ 500 ms PASS |
| CPU ORT FP32 english-large (`encoder.opt.onnx`) | 463.95 ms | ≤ 500 ms PASS (exact smoke parity vs CUDA f32) |
| CUDA BF16 Banking77 (100, pilot-v1 seed `20260917`) | 143.1 ms | accuracy 0.710. Interleaved groups; runner-up / close third; close-set joint refine (top 2–3) when leaders are close |
| CUDA BF16 DAIR Emotion (100, same pilot) | 22.4 ms | accuracy 0.430. Colon-template hypothesis boilerplate stripped before packing |
| CPU ORT INT8 english-large (`APOFASI_ORT_QUANT=1`) | ~300 ms | faster but **breaks** confidence/gates — not default |

Profile: `pack_ms` ≪ `fwd_ms`. Export ONNX with `scripts/ort_encoder_probe.py`.
Do not share Candle/MKL weights across threads. INT8 stays opt-in.

## Roadmap (architecture milestones)

| Milestone | Deliverable |
| --- | --- |
| **M0** | ✅ Core types, sequence, detect, confidence, Jev I/O, `LexicalEngine`, packing fixtures |
| **M1** | ✅ Candle `DecisionNet` + safetensors load; one padded encoder forward per question batch |
| **M2** | ✅ `CheckpointRegistry` preload + LRU; multilingual neural routing |
| **M3** | ✅ Action head + temperature buckets; `GatePolicy` / `gate_answer` helpers |
| **M4** | ✅ CLI smoke / bench (`cli`); host gate example |
| **M5** | ✅ RLCD rewards + train publish-layout scaffold |

## Non-goals (near term)

- Autoregressive generation or tool-calling inside Apofasi.
- Replacing Power/MoE for large generative models.
- Binding to Python/PyTorch for inference (conversion tools may use other
  languages offline; the runtime path stays Rust).
- Silent typed-decisions auto-routing without opt-in.

## References

- A3S monorepo submodule path: `crates/apofasi`
- Hugging Face `tokenizers` (Rust) and Candle for Metal/CUDA/CPU inference
