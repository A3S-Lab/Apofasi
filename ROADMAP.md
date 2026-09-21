# Apofasi Roadmap

Architecture source of truth: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## M0 — Core contracts + lean System One path

- [x] Crate scaffold (`a3s-apofasi`)
- [x] Architecture document (pure Rust)
- [x] `DecisionKind` / Jev-aligned `SystemOneRequest` / `SystemOneResponse`
- [x] Sequence builder with marker positions
- [x] Script / language detect + Router
- [x] Confidence + temperature decode (`logits` → `Answer`)
- [x] `DecisionEngine` trait + default `LexicalEngine`
- [x] `Client::system_one` facade + tests
- [x] Size-focused release profiles (`release`, `release-size`)
- [x] Golden fixtures for packing rules (ByteTokenizer + HF parity test)

## M1 — Candle inference (feature `infer`, opt-in)

- [x] `DecisionNet` on Candle ModernBERT path (`encoder.*` → Candle `model.*` remap)
- [x] Strict safetensors load for the published checkpoint layout
- [x] Device policy: Metal → CUDA → CPU (`DeviceRequest`)
- [x] English checkpoint smoke on Metal/CPU (`tests/neural_smoke.rs`, `APOFASI_CHECKPOINT`)
- [x] `NeuralEngine` → Jev `SystemOneResponse` via packer + temperature decode
- [x] Act / escalate head computed on the neural forward path
- [x] One encoder forward per question batch (right-pad; logits match per-question forward)
- [x] Metal SDPA path for ModernBERT + decision head (vendored encoder; mask cache)
- [x] Hot-path skips unused act head; `just ap*` / `ap-bench` use `--release`
- [x] Apple Silicon `mlx` feature: decision forward on prebuilt MLX (f32), selected when Metal is available

## M2 — Router residency

- [x] Checkpoint registry preload (`english` / `multilingual` / `typed-decisions`)
- [x] LRU `max_loaded`
- [x] Multilingual script routing with neural engine (`CheckpointRegistry`)

## M3 — Calibration and action head

- [x] Per-(kind, option-bucket) temperatures from checkpoint config
- [x] Act / escalate head on neural path
- [x] Host gating helper (`auto` vs `escalate`)

## M4 — Host tooling

- [x] `a3s-apofasi` CLI smoke / latency bench (`cli` feature; `just ap-smoke` / `just ap-bench`)
- [x] Example glue for Code / Desktop decision gates (`examples/host_gate.rs`)
- [x] Monorepo weight default: `scripts/ap/resolve_checkpoint.sh` + `just ap-checkpoint` / `ap-parity` (hub layout; crate stays brand-free)

## M5 — Training

- [x] RLCD proper-scoring rewards in Rust (`src/reward.rs`)
- [x] Fine-tune scaffold + Hub publish layout (`train` feature, `docs/publish-layout.md`; GRPO weight updates remain host/trainer-owned)

## Operating note — weights first

Use a compatible published checkpoint for development and host integration
before investing in domain fine-tunes. Training (M5) specializes; it is not a
gate for first neural smoke.

## Scale 1 — Latency (warm p50, 4-question triage)

- [x] Profile hook (`APOFASI_PROFILE`) — pack ≪ fwd on CUDA/CPU
- [x] Device-wide ModernBERT mask cache; F32 mask build, cast to activation dtype
- [x] CUDA `amp_dtype` / `APOFASI_DTYPE` BF16 load; logits cast to F32; choice parity vs F32
- [x] CUDA ≤ 40 ms on english-large (RTX 4090 BF16 22.98 ms / F32 22.38 ms, warmup 12, iters 40); suite 8/8 PASS
- [x] Optional `mkl` feature + Windows OpenMP redistributable note
- [x] CPU ≤ 500 ms via Scale 3 ORT encoder (`ort` feature + `encoder.onnx` / `encoder.opt.onnx`); english-large opt graph 463.95 ms p50 (warmup 12, iters 40) with exact smoke parity vs CUDA f32

## Scale 2 — Wide choice (Banking77)

- [x] Fixed case `bench --case wide` and `APOFASI_PROFILE` timings for grouped forwards
- [x] Pilot-v1 Banking77, 100 examples, seed `20260917+2`: accuracy 0.56 (published 0.560), CUDA BF16 p50 74.16 ms
- [x] Groups already fill `head_max_len` (8 sequences, batch seq 235, then one winner forward). One unshortened forward does not fit `max_len`. Option shortening, skipped groups, and a CUDA port of Metal-only Candle SDPA were not taken

## Enterprise GA

The bar is the architecture contract: typed answers, one encoder forward per
batch, route-before-inference, and confidence that hosts can gate on. Domain
accuracy (Banking77, DAIR) stays a fine-tune concern. Option shortening,
skipped groups, default INT8, and a thread count copied from one bench machine
are not GA work.

- [x] INT8 encoder loads only when `APOFASI_ORT_QUANT` is an explicit opt-in (`1`, `true`, `yes`, `on`). A directory that contains only `encoder.int8.onnx` keeps the Candle encoder
- [x] ORT intra-op threads follow `ORT_INTRA_THREADS` or the host's available parallelism
- [x] Feature-flag table matches `Cargo.toml` (`ort`, `mkl`; no phantom `core` feature)
- [x] CI workflow covers `cargo fmt`, `cargo clippy -D warnings`, and `cargo test` for the default build and `--features infer`
- [x] CLI gate JSON uses the wire labels `auto` / `escalate`. Suite preload loads only checkpoints present in the bundle. Suite Noul rows do not invent a `confidence` field
- [x] `smoke`, `bench`, and `host_gate` route through `CheckpointRegistry` before the forward. A bundle root is not forced onto one encoder, and a missing routed checkpoint fails closed
- [x] Decision engines report `output_tokens: 0`. Usage no longer invents 8 tokens per question
- [x] `system_one_routed` returns the full route record (checkpoint and reason), not only the checkpoint id
- [x] Gates escalate when confidence or noul is non-finite or outside `[0, 1]`
- [x] Decode rejects non-finite logits and probabilities instead of emitting an answer
- [x] Wide-choice composition errors when a group does not cover its options, instead of renormalizing the labels that remain
- [ ] GitHub CI green on that workflow (local tests are not a substitute for the hosted run)
- [ ] Crate consumed by an A3S host (Desktop / Code / CLI) through the documented `crates/apofasi` submodule
