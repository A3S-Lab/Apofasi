# Checkpoint publish layout

Hosts load exactly this tree (see `CheckpointPaths`):

```text
checkpoint/
├── rl_agent_config.json
├── model.safetensors
├── encoder/config.json
└── tokenizer/tokenizer.json
```

Optional CPU encoder graphs sit beside that tree. The `ort` feature loads
`encoder.opt.onnx` when present, otherwise `encoder.onnx`. Both are FP32.
`encoder.int8.onnx` loads only when `APOFASI_ORT_QUANT=1`; INT8 is faster and
changes confidence enough to break host gates, so it is not a fallback.

```text
checkpoint/
├── encoder.onnx          # optional FP32 ModernBERT encoder
├── encoder.opt.onnx      # optional graph-optimized FP32 encoder
└── encoder.int8.onnx     # optional; explicit APOFASI_ORT_QUANT=1 only
```

Bundles may nest `english/`, `multilingual/`, and `typed-decisions/` under one
root. Validate in Rust with `PublishLayout::validate` (`train` feature).
