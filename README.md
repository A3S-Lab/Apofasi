# A3S Apofasi

Typed **System-1** decision engine for A3S hosts.

*Apófasi* (απόφαση) means **decision**. Apofasi evaluates typed questions
(`choice`, `score`, `noul`) over a host-supplied state in a single forward
pass — no free-form generation, nothing to parse, nothing to hallucinate into
structure.

This repository is an A3S monorepo submodule at `crates/apofasi`.

## Status

`0.1.0` scaffolds the crate contract. Inference backends, routers, and host
SDKs land in follow-up releases behind stable types.

## Crate

| Item | Value |
| --- | --- |
| Rust package | `a3s-apofasi` |
| GitHub | [A3S-Lab/Apofasi](https://github.com/A3S-Lab/Apofasi) |
| Submodule path | `crates/apofasi` |
| License | MIT |

## Develop

```bash
cargo test
cargo fmt
cargo clippy --all-targets -- -D warnings
```

## License

MIT © A3S Lab
