# Crate-local recipes. Neural commands need a checkpoint in APOFASI_CHECKPOINT.

default:
    @just --list

test:
    cargo test

test-infer:
    cargo test --features infer

ap-smoke:
    cargo run --release --features cli --bin a3s-apofasi -- smoke

ap-bench:
    cargo run --release --features cli --bin a3s-apofasi -- bench --case triage
