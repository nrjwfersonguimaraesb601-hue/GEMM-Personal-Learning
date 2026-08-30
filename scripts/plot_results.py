#!/usr/bin/env python3
"""Plot the controlled FP32 comparison without a Python plotting dependency."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "rtx4060_laptop" / "comparison_4096.csv"
SVG = ROOT / "docs" / "assets" / "performance_4096.svg"
PNG = ROOT / "docs" / "assets" / "performance_4096.png"


def esc(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["precision"] == "FP32"]
    if not rows:
        raise SystemExit(f"no FP32 rows in {INPUT}")

    width, height = 1280, 700
    left, right, top, bottom = 95, 35, 70, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row["avg_gflops"]) / 1000.0 for row in rows]
    max_value = max(2.0, math.ceil(max(values) * 1.15 / 2.0) * 2.0)
    step = plot_w / len(rows)
    bar_w = step * 0.66

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151}.grid{stroke:#d1d5db}.label{font-size:13px}.small{font-size:11px}.title{font-size:23px;font-weight:700}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" class="title">4096³ FP32 SGEMM Controlled Comparison</text>',
        f'<text x="{width / 2}" y="52" text-anchor="middle" class="small">Avg throughput; warmup {esc(rows[0]["warmup"])}, iterations {esc(rows[0]["iterations"])}</text>',
    ]

    tick = 0
    while tick <= max_value:
        y = top + plot_h * (1 - tick / max_value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="label">{tick:g}</text>')
        tick += 2

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>')
    svg.append(f'<text x="{left}" y="62" text-anchor="start" class="label">Avg TFLOPS</text>')

    short_labels = {
        "Global Memory Coalescing": "Coalesced",
        "Shared Memory Tiling": "SMEM",
        "1D Register Tiling": "1D Reg",
        "2D Register Tiling": "2D Reg",
        "Vectorized Access": "Vectorized",
        "Shared Memory Padding": "Padding",
        "Autotuned C08": "C08",
        "Double Buffering": "Double Buffer",
    }
    for index, (row, value) in enumerate(zip(rows, values)):
        x = left + index * step + (step - bar_w) / 2
        bar_h = plot_h * value / max_value
        y = top + plot_h - bar_h
        color = "#2563eb" if row["stage"] == "baseline" else "#059669"
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        lx = x + bar_w / 2
        ly = top + plot_h + 22
        label = short_labels.get(row["kernel"], row["kernel"])
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" class="small">{esc(label)}</text>')

    svg.extend([
        f'<rect x="{width-255}" y="55" width="12" height="12" fill="#059669"/><text x="{width-237}" y="66" class="small">Custom FP32</text>',
        f'<rect x="{width-135}" y="55" width="12" height="12" fill="#2563eb"/><text x="{width-117}" y="66" class="small">cuBLAS FP32</text>',
        '<text x="640" y="684" text-anchor="middle" class="small">Sequential laptop GPU run; clocks, power, and temperature were not locked.</text>',
        '</svg>',
    ])

    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text("\n".join(svg) + "\n", encoding="utf-8")
    converter = shutil.which("convert")
    if converter is None:
        raise SystemExit(f"wrote {SVG}; ImageMagick 'convert' is required for PNG output")
    subprocess.run([converter, str(SVG), str(PNG)], check=True)
    print(PNG)


if __name__ == "__main__":
    main()
