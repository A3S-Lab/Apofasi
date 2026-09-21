# Checkpoint publish layout

Hosts load exactly this tree (see `CheckpointPaths`):

```text
checkpoint/
├── rl_agent_config.json
├── model.safetensors
├── encoder/config.json
└── tokenizer/tokenizer.json
```

Bundles may nest `english/`, `multilingual/`, and `typed-decisions/` under one
root. Validate in Rust with `PublishLayout::validate` (`train` feature).
