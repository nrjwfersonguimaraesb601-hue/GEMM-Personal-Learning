# CUDA GEMM Optimization Learning

一个从朴素 SGEMM 出发、逐步学习 CUDA GEMM 优化的实践项目。每个阶段保留 kernel、benchmark 与 Nsight Compute 记录，用于理解性能变化及瓶颈迁移。

> 这是学习项目，不是工业级 GEMM 库，也不以替代 cuBLAS 为目标。

## Optimization Path

| Stage | Implementation | Result |
|---|---|---:|
| 1 | [Naive kernel](./naive_kernel/README.md) | `~123 GFLOPS` |
| 2 | [Global memory coalescing](./Global_Memory_Coalescing_kernel/README.md) | `~1.03 TFLOPS` |
| 3 | [Shared memory tiling](./SMEM_kernel/README.md) | `~1.04 TFLOPS` |
| 4 | [1D register tiling](./1D_Blocktiling_kernel/README.md) | `~3.70 TFLOPS` |
| 5 | [2D register tiling](./2D_Blocktiling_kernel/README.md) | `~4.51 TFLOPS` |

表中为各阶段 square benchmark 的代表性平均性能；完整数据见 [PERFORMANCE_SUMMARY.md](./PERFORMANCE_SUMMARY.md)。

## Current Stage: 2D Register Tiling

当前 kernel 使用 `BM=64, BN=64, BK=8, TM=8, TN=8`，每个线程在寄存器中累加一个 `8 x 8` micro tile。

- `4096³`: `4505.61 GFLOPS`，相对 1D 版本提升 `23.1%`
- 小矩阵收益有限，`1024³` 与 1D 版本基本持平
- Nsight Compute 显示主要限制来自 shared-memory bank conflict、低效 global store、MIO throttle 与较高寄存器占用

完整分析与截图见 [2D Blocktiling profiling](./2D_Blocktiling_kernel/profiling/README.md)。

## Test Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute Capability: 8.9
- SMs: 24
- Global Memory: 8 GiB

Benchmark 默认关注 kernel 性能，部分数据使用 `--no-check`。正确性验证请移除该参数并启用 CPU reference check。

## Next Steps

- 优化 shared-memory layout 与 bank conflict
- 尝试 vectorized load/store
- 调整 `BM / BN / BK / TM / TN`，平衡复用率、寄存器压力与 occupancy
