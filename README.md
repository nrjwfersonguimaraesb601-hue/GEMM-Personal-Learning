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
| 7 | [Shared-memory layout padding](./7.Shared_Memory_Layout_Optimization/README.md) | `8.582 TFLOPS` |

完整的逐尺寸数据见 [PERFORMANCE_SUMMARY.md](./PERFORMANCE_SUMMARY.md)。

## 当前阶段：Shared-memory Layout Padding

当前版本沿用 Stage 6 的 `BM=64, BN=64, BK=8, TM=8, TN=8`、`float4`
访存和 2D register tiling，集中优化 Nsight Compute 暴露出的 shared-memory
bank conflict：

- As 的物理布局由 `[BK][BM]` 改为 `[BK][BM + 4]`
- Bs 的物理布局由 `[BK][BN]` 改为 `[BK][BN + 4]`
- As/Bs 合计使用 `4352 bytes` static shared memory
- 通过多轮 Nsight Compute 分析，主要 bank conflict 已经基本消除
- XOR swizzle 暂时作为后续扩展，本阶段只掌握 padding

主要结果：

- `256^3`、`512^3`、`1024^3` 均通过 CPU reference check
- `1024^3`: `7429.68 GFLOPS`，比 Vectorized 版本提升 `15.9%`
- `4096^3`: 完整 suite 为 `8581.60 GFLOPS`，提升 `10.0%`
- `4096^3` 单独复测达到 `8984.18 GFLOPS`，平均吞吐接近 `9.0 TFLOPS`
- `256x4096x4096` 提升 `25.1%`

当前实现要求 `M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`，因此没有把
旧测试中的 `1023^3` 用例直接搬过来。

## 编译和运行当前版本

```bash
cd 7.Shared_Memory_Layout_Optimization

nvcc -O3 -lineinfo -arch=sm_89 \
  Shared_Memory_Layout_Padding_benchmark.cu \
  -o Shared_Memory_Layout_Padding_bench

./Shared_Memory_Layout_Padding_bench \
  --warmup 10 --iters 50
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

- 改进 A 的 warp-level global-load 合并访问
- 支持非整除尺寸和尾部 tile
- 调整 tile 参数并尝试 double buffering / warp tiling
- 增加统一的 cuBLAS baseline 和性能曲线
- 将 XOR swizzle 保留为后续 shared-memory layout 扩展实验
