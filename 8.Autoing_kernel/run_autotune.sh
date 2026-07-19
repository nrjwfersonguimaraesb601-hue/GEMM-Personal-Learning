#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-quick}"
NVCC_BIN="${NVCC:-nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_89}"
BINARY="autotune_padding_bench"

case "$MODE" in
  quick)
    WARMUP=5
    ITERS=20
    CSV="autotune_quick.csv"
    ;;
  full)
    WARMUP=10
    ITERS=50
    CSV="autotune_full.csv"
    ;;
  *)
    echo "Usage: ./run_autotune.sh [quick|full]" >&2
    exit 2
    ;;
esac

echo "[1/2] Compiling for ${CUDA_ARCH}..."
"$NVCC_BIN" \
  -O3 \
  -std=c++17 \
  -lineinfo \
  -arch="$CUDA_ARCH" \
  --ptxas-options=-v \
  autotune_padding_benchmark.cu \
  -o "$BINARY" \
  2>&1 | tee "compile_${MODE}.log"

echo "[2/2] Running ${MODE} autotuning suite..."
./"$BINARY" \
  --suite "$MODE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --verify-size 256 \
  --csv "$CSV" \
  | tee "autotune_${MODE}.log"

echo
echo "Results: $SCRIPT_DIR/$CSV"
echo "Console log: $SCRIPT_DIR/autotune_${MODE}.log"
echo "Compile resource log: $SCRIPT_DIR/compile_${MODE}.log"