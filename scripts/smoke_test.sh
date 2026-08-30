#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

make all

run_case() {
    local label=$1
    shift
    echo "[smoke] $label"
    "$@"
}

run_case naive \
    build/naive_bench 256 256 256 --warmup 2 --iters 3 --bx 32 --by 32
run_case coalesced \
    build/coalesced_bench 256 256 256 --warmup 2 --iters 3 --bx 32 --by 32
run_case smem \
    build/smem_bench 256 256 256 --warmup 2 --iters 3 --bx 32 --by 32
run_case 1d-register-tiling \
    build/register_1d_bench 256 256 256 --warmup 2 --iters 3 --bx 512 --by 1
run_case 2d-register-tiling \
    build/register_2d_bench 256 256 256 --warmup 2 --iters 3 --bx 64 --by 1
run_case vectorized \
    build/vectorized_bench 256 256 256 --warmup 2 --iters 3 --bx 64 --by 1 --max-check-dim 256
run_case padding \
    build/padding_bench 256 256 256 --warmup 2 --iters 3 --max-check-dim 256
run_case autotuning \
    build/autotuning_bench --suite quick --warmup 1 --iters 1 \
    --verify-size 256 --csv build/stage8_smoke.csv
run_case warp-tiling \
    build/warp_tiling_bench 256 256 256 --warmup 2 --iters 3 --max-check-dim 256
run_case double-buffering \
    build/double_buffering_bench 256 256 256 --warmup 2 --iters 3 --max-check-dim 256
run_case cublas-fp32 \
    build/cublas_bench 256 256 256 --warmup 2 --iters 3 --max-check-dim 256 --math fp32

echo "[smoke] all stage correctness checks completed"
