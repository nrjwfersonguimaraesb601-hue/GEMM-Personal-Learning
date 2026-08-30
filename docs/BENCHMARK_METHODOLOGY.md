# Benchmark Methodology

本项目用三类互相独立的运行回答三个不同问题。

## Correctness

CPU reference 用于判断结果是否正确。`PASS` 表示实际执行并通过比较，`FAIL`
表示发现误差，`SKIP` 仅表示该尺寸没有运行 CPU reference。任何性能表都不能把
`SKIP` 改写成正确性已经验证。

优化 kernel 通常要求矩阵尺寸满足 tile 整除和 `float4` 对齐条件。具体约束以
各 Stage README 和 benchmark 的参数检查为准。

## Benchmark

正式 latency 使用 CUDA Event，只包围 kernel 或 cuBLAS GEMM 调用。以下工作不在
计时区间内：

- host/device allocation
- H2D 和 D2H copy
- CPU reference
- cuBLAS handle 初始化

历史正式设置通常是 warmup 10 次、测量 50 次。项目同时记录 Avg 和 Best
GFLOPS；主比较使用 Avg。不同历史轮次受 Laptop GPU 温度、功耗和 boost clock
影响，几个百分点的差距不应解释成稳定胜负。

统一入口：

```bash
# 全阶段 256³ correctness + 同轮 4096³ FP32 comparison
./scripts/run_comparison.sh

# 单阶段纯性能运行
./scripts/benchmark_stage.sh <stage>
WARMUP=10 ITERS=50 ./scripts/benchmark_stage.sh <stage> 4096 4096 4096
```

`run_comparison.sh` 将统一行写入
`results/rtx4060_laptop/comparison_4096.csv`。CSV 中的 `PASS@256` 表示同一个
脚本先对该实现运行了 256³ CPU reference；它不表示对 4096³ 执行了 CPU
reference。

## Profiling

Nsight Compute 用于解释性能瓶颈。profiler 会 replay kernel，因此界面或程序在
`ncu` 下打印的 latency 不进入正式性能表。原始报告保存在各阶段的
`profiling/raw/`，截图和人工解释分别位于 `full_report/` 与 `instr_report/`。

```bash
./scripts/profile_stage.sh <stage> full
./scripts/profile_stage.sh <stage> instr
```

## Precision

自定义 kernel 使用 FP32 CUDA Core arithmetic。cuBLAS 主基准使用
`CUBLAS_COMPUTE_32F`。`CUBLAS_COMPUTE_32F_FAST_TF32` 是单独的 Tensor Core
参考，不进入 FP32 排名。

## Authoritative Data

- 当前同轮 4096³ FP32 主比较：`results/rtx4060_laptop/comparison_4096.csv`
- Stage 1–8 历史多尺寸数据：根目录 `PERFORMANCE_SUMMARY.md`
- 不同历史轮次的 v1.0 4096³ 汇总：`results/final_4096.csv`
- Stage 8 原始搜索结果：`8.Autoing_kernel/results/`
- Stage 10/11 与 cuBLAS 最终记录：根性能汇总中的独立章节
