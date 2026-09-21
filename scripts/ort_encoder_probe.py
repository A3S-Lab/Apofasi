#!/usr/bin/env python3
"""Export a checkpoint's ModernBERT encoder to ONNX and time ORT on CPU.

The Rust `ort` feature loads `encoder.onnx` / `encoder.opt.onnx` from the
checkpoint directory. This script only exports and measures; it does not
quantize. INT8 is not the default path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, ModernBertModel


def remap_encoder_state(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Checkpoint stores `encoder.*`; HuggingFace ModernBertModel expects bare keys.
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not k.startswith("encoder."):
            continue
        out[k[len("encoder.") :]] = v
    return out

def export_onnx(model: ModernBertModel, onnx_path: Path, seq_len: int = 96) -> None:
    model.eval()
    ids = torch.zeros(1, seq_len, dtype=torch.long)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        model,
        (ids, mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def maybe_optimize(onnx_path: Path, hidden_size: int, num_heads: int) -> Path | None:
    """Write encoder.opt.onnx beside the FP32 export when transformers optimizer works."""
    try:
        from onnxruntime.transformers import optimizer
    except ImportError:
        return None
    opt_path = onnx_path.with_name("encoder.opt.onnx")
    opt = optimizer.optimize_model(
        str(onnx_path),
        model_type="bert",
        num_heads=num_heads,
        hidden_size=hidden_size,
    )
    opt.save_model_to_file(str(opt_path))
    return opt_path


def bench_ort(onnx_path: Path, batch: int, seq: int, warmup: int, iters: int) -> float:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    ids = np.zeros((batch, seq), dtype=np.int64)
    mask = np.ones((batch, seq), dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    for _ in range(warmup):
        sess.run(None, feeds)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to APOFASI_CHECKPOINT.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(".tmp-ort"),
        help="Directory for encoder.onnx and encoder.opt.onnx.",
    )
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=91)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        env = os.environ.get("APOFASI_CHECKPOINT")
        if not env:
            ap.error("pass --checkpoint or set APOFASI_CHECKPOINT")
        checkpoint = Path(env)

    enc_cfg = checkpoint / "encoder" / "config.json"
    weights = checkpoint / "model.safetensors"
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "encoder.onnx"

    cfg = AutoConfig.from_pretrained(enc_cfg)
    # Prefer local architecture even if hub name differs.
    model = ModernBertModel(cfg)
    state = remap_encoder_state(load_file(str(weights)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"loaded encoder weights missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print("missing sample:", missing[:8], flush=True)

    if not onnx_path.exists():
        print(f"exporting {onnx_path}", flush=True)
        export_onnx(model, onnx_path, seq_len=args.seq)
    else:
        print(f"reusing {onnx_path}", flush=True)

    opt_path = onnx_path.with_name("encoder.opt.onnx")
    if not opt_path.exists():
        written = maybe_optimize(
            onnx_path,
            hidden_size=int(getattr(cfg, "hidden_size", 768)),
            num_heads=int(getattr(cfg, "num_attention_heads", 12)),
        )
        if written is not None:
            print(f"wrote {written}", flush=True)
        else:
            print("optimizer unavailable; skip encoder.opt.onnx", flush=True)
    bench_path = opt_path if opt_path.exists() else onnx_path

    # Torch baseline (same shapes)
    model.eval()
    ids = torch.zeros(args.batch, args.seq, dtype=torch.long)
    mask = torch.ones(args.batch, args.seq, dtype=torch.long)
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=ids, attention_mask=mask)
        tms: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            tms.append((time.perf_counter() - t0) * 1000.0)
        tms.sort()
        torch_p50 = tms[len(tms) // 2]

    ort_p50 = bench_ort(bench_path, args.batch, args.seq, args.warmup, args.iters)
    print(
        json.dumps(
            {
                "batch": args.batch,
                "seq": args.seq,
                "onnx": str(bench_path),
                "torch_cpu_p50_ms": round(torch_p50, 2),
                "ort_cpu_p50_ms": round(ort_p50, 2),
                "budget_encoder_ms": 450,
                "ort_under_budget": ort_p50 <= 450,
            },
            indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
