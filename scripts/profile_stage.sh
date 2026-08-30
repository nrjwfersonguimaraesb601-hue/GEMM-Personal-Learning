#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <stage> <full|instr>" >&2
    echo "Stages: naive coalesced smem 1d 2d vectorized padding warp double-buffering" >&2
    exit 2
fi

stage=$1
mode=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

case "$stage" in
    naive) target=naive; binary=build/naive_bench; regex='.*calculate_Matrix.*'; out_dir=1.naive_kernel/profiling/raw; prefix=naive; extra=(--bx 32 --by 32) ;;
    coalesced) target=coalesced; binary=build/coalesced_bench; regex='.*calculate_Matrix.*'; out_dir=2.Global_Memory_Coalescing_kernel/profiling/raw; prefix=gmemc; extra=(--bx 32 --by 32) ;;
    smem) target=smem; binary=build/smem_bench; regex='.*calculate_Matrix.*'; out_dir=3.SMEM_kernel/profiling/raw; prefix=smem; extra=(--bx 32 --by 32) ;;
    1d) target=1d; binary=build/register_1d_bench; regex='.*calculate_Matrix.*'; out_dir=4.1D_Blocktiling_kernel/profiling/raw; prefix=1D_Blocktiling; extra=(--bx 512 --by 1) ;;
    2d) target=2d; binary=build/register_2d_bench; regex='.*sgemm_2d_register_tiling.*'; out_dir=5.2D_Blocktiling_kernel/profiling/raw; prefix=2D_Blocktiling; extra=(--bx 64 --by 1) ;;
    vectorized) target=vectorized; binary=build/vectorized_bench; regex='.*sgemm_vectorize_GEMM_SMEM.*'; out_dir=6.Vectorize_kernel/profiling/raw; prefix=Vectorize; extra=(--bx 64 --by 1) ;;
    padding) target=padding; binary=build/padding_bench; regex='.*sgemm_shared_memory_layout_padding.*'; out_dir=7.Shared_Memory_Layout_Optimization/profiling/raw; prefix=Shared_Memory_Layout_Padding; extra=() ;;
    warp) target=warp; binary=build/warp_tiling_bench; regex='.*sgemm_2d_register_tiling.*'; out_dir=10.Wraptiling_kernel/profiling/raw; prefix=Wraptiling; extra=() ;;
    double-buffering) target=double-buffering; binary=build/double_buffering_bench; regex='.*sgemm_double_buffering.*'; out_dir=11.Double_Buffering/profiling/raw; prefix=Double_Buffering; extra=() ;;
    *) echo "Unknown or unsupported profiling stage: $stage" >&2; exit 2 ;;
esac

case "$mode" in
    full) section=(--set full); suffix=full ;;
    instr) section=(--section InstructionStats); suffix=instr ;;
    *) echo "Mode must be full or instr" >&2; exit 2 ;;
esac

make "$target"
mkdir -p "$out_dir"
exec ncu -f "${section[@]}" \
    --kernel-name-base demangled \
    --kernel-name "regex:$regex" \
    --launch-skip 1 --launch-count 1 \
    -o "$out_dir/${prefix}_${suffix}" \
    "$binary" 1024 1024 1024 --warmup 1 --iters 1 --no-check "${extra[@]}"
