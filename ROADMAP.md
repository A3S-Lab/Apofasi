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
