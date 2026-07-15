# CUDA GEMM Optimization Learning

一个从朴素 SGEMM 出发、逐步学习 CUDA GEMM 优化的实践项目。每个阶段保留
kernel、benchmark 与 Nsight Compute 记录，用来观察优化后性能如何变化、
瓶颈如何迁移。

> 这是学习项目，不是工业级 GEMM 库，也不以替代 cuBLAS 为目标。

## 优化路线

下表统一使用 `4096^3` 的 Avg GFLOPS：

| Stage | Implementation | Avg Performance |
| ----: | -------------- | --------------: |
| 1 | [Naive kernel](./1.naive_kernel/README.md) | `0.123 TFLOPS` |
| 2 | [Global memory coalescing](./2.Global_Memory_Coalescing_kernel/README.md) | `0.871 TFLOPS` |
| 3 | [Shared memory tiling](./3.SMEM_kernel/README.md) | `0.958 TFLOPS` |
| 4 | [1D register tiling](./4.1D_Blocktiling_kernel/README.md) | `3.660 TFLOPS` |
| 5 | [2D register tiling](./5.2D_Blocktiling_kernel/README.md) | `4.506 TFLOPS` |
| 6 | [Vectorized memory access](./6.Vectorize_kernel/README.md) | `7.802 TFLOPS` |

完整的逐尺寸数据见 [PERFORMANCE_SUMMARY.md](./PERFORMANCE_SUMMARY.md)。

## 当前阶段：Vectorized Memory Access

当前版本沿用 `BM=64, BN=64, BK=8, TM=8, TN=8` 的 2D register tiling，
并加入以下变化：

- A、B 使用 `float4` 从 global memory 加载
- A tile 在写入 shared memory 时转置，改善计算阶段的读取方式
- C 使用 `float4` 写回，每次存储连续 4 个结果
- SASS 已生成 `LDG.E.128` / `STG.E.128`

主要结果：

- 7 个 benchmark case 全部通过 CPU reference check
- `1024^3`: `6412.41 GFLOPS`，比 2D 版本提升 `88.8%`
- `4096^3`: `7802.28 GFLOPS`，比 2D 版本提升 `73.2%`
- 7 个共同合法尺寸上的提升范围为 `63.2%–88.8%`

当前实现要求 `M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`，因此没有把
旧测试中的 `1023^3` 用例直接搬过来。

## 编译和运行当前版本

```bash
cd 6.Vectorize_kernel

nvcc -O3 -lineinfo -arch=sm_89 \
  Vectorize_kernel_benchmark.cu \
  -o Vectorize_bench

./Vectorize_bench \
  --warmup 10 --iters 50 --bx 64 --by 1
```

不传位置参数 `M N K` 时，会依次输出全部内置尺寸的 latency、Avg GFLOPS、
Best GFLOPS 和 correctness 结果。

## 测试环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute Capability: 8.9
- SMs: 24
- Global Memory: 8 GiB
- Benchmark: warmup 10 次，正式迭代 50 次

## 下一步

- 根据 Nsight Compute 结果降低 shared-memory conflict，并提高 C store sector 利用率
- 改进 A 的 warp-level global-load 合并访问
- 支持非整除尺寸和尾部 tile
- 调整 tile 参数并尝试 double buffering / warp tiling
- 增加统一的 cuBLAS baseline 和性能曲线
