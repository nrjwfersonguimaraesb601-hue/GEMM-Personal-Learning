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
| 8 | [Compile-time autotuning](./8.Autoing_kernel/README.md) | `9.665 TFLOPS*` |

完整的逐尺寸数据见 [PERFORMANCE_SUMMARY.md](./PERFORMANCE_SUMMARY.md)。

`*` Stage 8 的结果是一次完整 autotune suite 中 C08 候选配置的 `4096^3` Avg
GFLOPS，不代表所有尺寸都应使用同一组参数。

## 当前阶段：Compile-time Autotuning

当前版本沿用 Stage 7 的计算、`float4` 访存和 As/Bs padding，只把
`BM`、`BN`、`BK`、`TM`、`TN` 做编译期枚举。每个候选都会先在 `256^3` 上做
CPU reference check，再在 7 个矩阵上用 CUDA event 测速。

本轮共比较 14 组配置。综合 7 个 case 的最佳候选为 C08：
`BM=128, BN=64, BK=16, TM=8, TN=8`，几何平均相对 C00 加速 `1.0805x`。
在 `4096^3` 上达到 `9664.98 GFLOPS`，比同一轮 C00 的 `9103.69 GFLOPS`
高 `6.2%`。

需要注意：

- Stage 8 是搜索器，不是运行时自动选择器；候选参数需要手动固定
- `256^3`、`512^3` 等小尺寸的最佳配置与大尺寸不同
- 当前 kernel 没有边界分支，输入尺寸必须满足所选 tile 的整除条件
- Laptop GPU 的温度、功耗和频率会影响单轮结果

完整的配置和逐尺寸数据见 [8.Autoing_kernel/README.md](./8.Autoing_kernel/README.md)
与 [PERFORMANCE_SUMMARY.md](./PERFORMANCE_SUMMARY.md)。

## 编译和运行 Stage 8

```bash
cd 8.Autoing_kernel

# 快速筛选
./run_autotune.sh quick

# 完整比较
./run_autotune.sh full
```

脚本会使用 `sm_89` 编译所有模板实例，并生成 CSV 和排名结果。

## 测试环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute Capability: 8.9
- SMs: 24
- Global Memory: 8 GiB
- Benchmark: warmup 10 次，正式迭代 50 次

## 下一步

- 改进 A 的 warp-level global-load 合并访问
- 支持非整除尺寸和尾部 tile
- 对 C08、C13 等候选重复测试并固定下一版 kernel 参数
- 尝试 double buffering、warp tiling 和 `cp.async`
- 增加统一的 cuBLAS baseline 和性能曲线
- 将 XOR swizzle 保留为 shared-memory layout 扩展实验
