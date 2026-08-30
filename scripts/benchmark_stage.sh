#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 <stage> [M N K]" >&2
    echo "Stages: naive coalesced smem 1d 2d vectorized padding autotuning warp double-buffering cublas-fp32 cublas-tf32" >&2
    exit 2
fi

stage=$1
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

warmup=${WARMUP:-10}
iters=${ITERS:-50}
shape=("$@")

case "$stage" in
    naive) target=naive; binary=build/naive_bench; extra=(--bx 32 --by 32) ;;
    coalesced) target=coalesced; binary=build/coalesced_bench; extra=(--bx 32 --by 32) ;;
    smem) target=smem; binary=build/smem_bench; extra=(--bx 32 --by 32) ;;
    1d) target=1d; binary=build/register_1d_bench; extra=(--bx 512 --by 1) ;;
    2d) target=2d; binary=build/register_2d_bench; extra=(--bx 64 --by 1) ;;
    vectorized) target=vectorized; binary=build/vectorized_bench; extra=(--bx 64 --by 1) ;;
    padding) target=padding; binary=build/padding_bench; extra=() ;;
    warp) target=warp; binary=build/warp_tiling_bench; extra=() ;;
    double-buffering) target=double-buffering; binary=build/double_buffering_bench; extra=() ;;
    cublas-fp32) target=cublas; binary=build/cublas_bench; extra=(--math fp32) ;;
    cublas-tf32) target=cublas; binary=build/cublas_bench; extra=(--math tf32) ;;
    autotuning)
        exec 8.Autoing_kernel/run_autotune.sh full
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac

make "$target"
exec "$binary" "${shape[@]}" --warmup "$warmup" --iters "$iters" --no-check "${extra[@]}"
