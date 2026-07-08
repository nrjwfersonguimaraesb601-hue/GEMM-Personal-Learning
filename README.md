# CUDA GEMM Optimization Learning

## Overview

这是一个 CUDA SGEMM/GEMM 优化学习项目。

项目目标是沿着经典 GEMM 优化路线，从最基础的 naive kernel 开始，逐步学习和验证：

- global memory coalescing
- shared memory tiling
- 1D register tiling
- 2D register tiling

每一版都会尽量保留代码、benchmark 结果和 Nsight Compute 观察，方便回看“这一阶段到底改变了什么、性能为什么变化”。

这不是工业级 GEMM 库，也不声称达到 cuBLAS 水平。当前重点是学习优化路线、建立稳定 benchmark 口径，并用 profiling 工具理解瓶颈。

## Hardware Environment

当前记录中的 benchmark 主要来自：

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute Capability: `8.9`
- Global Memory: `8.00 GiB`
- SM Count: `24`
- Max Threads Per Block: `1024`

## Optimization Progress

| Stage | Directory | Status |
|---|---|---|
| Naive Kernel | [`naive_kernel`](./naive_kernel/README.md) | completed baseline |
| Global Memory Coalescing | [`Global_Memory_Coalescing_kernel`](./Global_Memory_Coalescing_kernel/README.md) | completed |
| Shared Memory Tiling | [`SMEM_kernel`](./SMEM_kernel/README.md) | completed |
| 1D Register Tiling | [`1D_Blocktiling_kernel`](./1D_Blocktiling_kernel/README.md) | completed |
| 2D Register Tiling | [`2D_Blocktiling_kernel`](./2D_Blocktiling_kernel/README.md) | basic implementation + benchmark completed |

后续方向包括 shared memory layout、vectorized memory access、bank conflict optimization，以及更系统的 `BM / BN / BK / TM / TN` 调参。

## Current Status

项目当前推进到 `2D Register Tiling` 阶段。

2D 基础实现已经跑通 benchmark，在大矩阵上相对 1D register tiling 有提升。例如 `4096^3` 从 1D 的约 `3659.73 GFLOPS` 提升到 2D 的约 `4505.61 GFLOPS`。

不过当前 2D kernel 还不是最终优化版本。Nsight Compute 显示它仍然存在几个明显问题：

- shared memory bank conflict
- global store access pattern 不理想
- uncoalesced shared access
- register pressure 较高

所以当前阶段先记录结果和问题，不继续修改 kernel 代码。

## Benchmark Summary

完整性能表和相对上一版的提升记录在：

- [`PERFORMANCE_SUMMARY.md`](./PERFORMANCE_SUMMARY.md)

简要结论：

- Naive square baseline 约为 `~123 GFLOPS`
- Global memory coalescing 把 square case 推到 `~616-1026 GFLOPS`
- Shared memory tiling 进一步稳定到 `~641-1039 GFLOPS`
- 1D register tiling 在 `4096^3` 上约 `3659.73 GFLOPS`
- 2D register tiling 在 `4096^3` 上约 `4505.61 GFLOPS`
- 2D 相对 1D 在 `4096^3` 上提升约 `23.1%`
- 小矩阵上 2D 不一定更快，可能受到 kernel 开销、寄存器压力和访存模式影响

## Notes

性能表主要用于观察优化趋势。部分 benchmark 使用 `--no-check`，这表示关闭 CPU reference 校验，只记录 kernel benchmark 性能。

正确性需要单独开启 CPU check，或者运行对应目录下的 correctness-first 版本。
