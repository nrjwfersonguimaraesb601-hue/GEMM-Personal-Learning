#!/usr/bin/env python3
"""Run one controlled 4096^3 FP32 comparison and write a unified CSV."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "rtx4060_laptop" / "comparison_4096.csv"
SHAPE = (4096, 4096, 4096)

STAGES = [
    ("1", "Naive", "naive_bench", ["--bx", "32", "--by", "32"]),
    ("2", "Global Memory Coalescing", "coalesced_bench", ["--bx", "32", "--by", "32"]),
    ("3", "Shared Memory Tiling", "smem_bench", ["--bx", "32", "--by", "32"]),
    ("4", "1D Register Tiling", "register_1d_bench", ["--bx", "512", "--by", "1"]),
    ("5", "2D Register Tiling", "register_2d_bench", ["--bx", "64", "--by", "1"]),
    ("6", "Vectorized Access", "vectorized_bench", ["--bx", "64", "--by", "1"]),
    ("7", "Shared Memory Padding", "padding_bench", []),
    ("10", "Warp Tiling", "warp_tiling_bench", []),
    ("11", "Double Buffering", "double_buffering_bench", []),
    ("baseline", "cuBLAS FP32", "cublas_bench", ["--math", "fp32"]),
]


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer")
    return value


def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout or ""


def one_csv_row(output: str) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV row, got {len(rows)}")
    return rows[0]


def device_name() -> str:
    try:
        return run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def normalized_row(
    stage: str,
    kernel: str,
    raw: dict[str, str],
    warmup: int,
    iterations: int,
    correctness: str,
    timestamp: str,
    gpu: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "kernel": kernel,
        "M": str(SHAPE[0]),
        "N": str(SHAPE[1]),
        "K": str(SHAPE[2]),
        "precision": "FP32",
        "min_ms": raw["min_ms"],
        "avg_ms": raw["avg_ms"],
        "max_ms": raw["max_ms"],
        "avg_gflops": raw["avg_gflops"],
        "best_gflops": raw["best_gflops"],
        "correctness": correctness,
        "warmup": str(warmup),
        "iterations": str(iterations),
        "gpu": gpu,
        "timestamp_utc": timestamp,
        "source": "controlled_same_script_run",
        "notes": "sequential laptop GPU run; clocks, power, and temperature not locked",
    }


def main() -> None:
    warmup = positive_env("WARMUP", 10)
    iterations = positive_env("ITERS", 50)
    skip_smoke = os.environ.get("SKIP_SMOKE") == "1"

    run(["make", "all"])
    if not skip_smoke:
        print("[comparison] running 256^3 correctness suite", flush=True)
        run([str(ROOT / "scripts" / "smoke_test.sh")])

    correctness = "NOT_RUN" if skip_smoke else "PASS@256"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    gpu = device_name()
    common = [
        *(str(value) for value in SHAPE),
        "--warmup", str(warmup),
        "--iters", str(iterations),
        "--no-check",
        "--csv",
    ]

    rows: list[dict[str, str]] = []
    for stage, kernel, binary, extra in STAGES:
        print(f"[comparison] {kernel}", flush=True)
        output = run([str(ROOT / "build" / binary), *common, *extra], capture=True)
        raw = one_csv_row(output)
        rows.append(normalized_row(
            stage, kernel, raw, warmup, iterations,
            correctness, timestamp, gpu,
        ))

    print("[comparison] Autotuned C08", flush=True)
    with tempfile.TemporaryDirectory(prefix="gemm-comparison-") as temp_dir:
        stage8_csv = Path(temp_dir) / "stage8.csv"
        run([
            str(ROOT / "build" / "autotuning_bench"),
            "--suite", "quick",
            "--warmup", str(warmup),
            "--iters", str(iterations),
            "--no-verify",
            "--csv", str(stage8_csv),
        ], capture=True)
        with stage8_csv.open(newline="", encoding="utf-8") as handle:
            candidates = [
                row for row in csv.DictReader(handle)
                if row["config"] == "C08_128x64x16_8x8"
                and row["case"] == "square_4096"
            ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Stage 8 C08 4096 row, got {len(candidates)}")
    rows.insert(7, normalized_row(
        "8", "Autotuned C08", candidates[0], warmup, iterations,
        correctness, timestamp, gpu,
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_output = OUTPUT.with_suffix(".tmp")
    with temp_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp_output.replace(OUTPUT)
    print(f"[comparison] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
