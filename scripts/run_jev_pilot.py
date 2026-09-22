"""Run the unmodified jev-benchmarks pilot-v1 manifest through a3s-apofasi.

The config file is the upstream ``configs/pilot-v1.yaml``: seed 20260917,
100 examples for each of AG News, Banking77, and DAIR Emotion. This script
does not edit that file or drop examples. Scoring uses ``group_scores``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PILOT_CONFIG = "configs/pilot-v1.yaml"
EXPERIMENT = "btzsc-pilot-v1"


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: run_jev_pilot.py JEV_BENCHMARKS_DIR APOFASI_BIN CHECKPOINT_DIR"
        )
    bench_dir = Path(sys.argv[1])
    binary = Path(sys.argv[2])
    checkpoint = Path(sys.argv[3])
    config = bench_dir / PILOT_CONFIG
    if not config.is_file():
        raise SystemExit(f"missing official config: {config}")
    text = config.read_text(encoding="utf-8")
    if "samples_per_dataset: 100" not in text or "seed: 20260917" not in text:
        raise SystemExit("pilot-v1.yaml is not the unmodified 100-example contract")

    env = os.environ.copy()
    src = str(bench_dir / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    sys.path.insert(0, src)
    from jev_benchmarks.config import load_config
    from jev_benchmarks.data import prepare_manifest

    loaded = load_config(config)
    prepare_manifest(loaded)
    manifest = bench_dir / "results" / "runs" / EXPERIMENT / "manifest.jsonl"
    output = bench_dir / "results" / "runs" / EXPERIMENT / "predictions-apofasi.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"prepare did not write {manifest}")
    count = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    if count != 300:
        raise SystemExit(f"manifest has {count} rows; the pilot contract is 300")

    subprocess.run(
        [
            str(binary),
            "jev-bench",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda",
            "--model",
            "english",
            "--experiment",
            EXPERIMENT,
            "--question",
            "Which single label best describes the input text?",
        ],
        check=True,
    )

    from jev_benchmarks.models import Prediction
    from jev_benchmarks.metrics import group_scores

    predictions = []
    failures = 0
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            failures += 1
            continue
        predictions.append(Prediction.from_dict(row))
    if len(predictions) + failures != 300:
        raise SystemExit(
            f"scored {len(predictions)} predictions and {failures} failures, expected 300 rows"
        )
    scores = group_scores(predictions, ece_bins=10, error_budget=0.05)
    report = bench_dir / "results" / "runs" / EXPERIMENT / "apofasi-group-scores.json"
    report.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(predictions), "failures": failures, "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
