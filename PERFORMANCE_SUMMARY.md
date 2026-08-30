# Performance Summary

这份文件统一记录 CUDA GEMM 学习项目中各阶段 kernel 的 benchmark 结果。

## Hardware

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute Capability: `8.9`
- Global Memory: `8.00 GiB`
- SM Count: `24`
- Max Threads Per Block: `1024`

## Benchmark Notes

- Warmup: `10`
- Iterations: `50`
- 使用 CUDA event 统计 kernel latency，不包含 H2D / D2H 和 CPU reference 时间
- Stage 1–5 的历史性能表来自 `--no-check` 运行
- Stage 6 Vectorized 的历史运行启用了 CPU check，7 个 case 全部 `PASS`
- Stage 7 Padding 的默认 `max_check_dim=1024`：前三个用例 `PASS`，更大用例
  显示 `SKIP`
- Stage 8 Autotuning 在 `256^3` 上验证了 14 个配置，全部 `PASS`；完整 suite
  使用 warmup 10、正式迭代 50，比较 14 个配置在 7 个 case 上的 Avg GFLOPS
- CPU check 位于计时区间之外，因此启用检查的 kernel 数据仍可与历史结果比较

Laptop GPU 会受功耗、温度和频率变化影响，表格主要用于观察优化趋势，而不是
作为跨机器的绝对性能结论。

## Controlled 4096³ Comparison (2026-08-30)

这是当前优先使用的同轮 FP32 对比：所有 benchmark 由
`scripts/run_comparison.sh` 顺序执行，warmup 10、iterations 50。256³ CPU
reference 全部通过；4096³ 性能循环不执行 CPU reference。

| Stage | Avg (ms) | Avg TFLOPS | Best TFLOPS |
| ----: | -------: | ----------: | -----------: |
| 1 Naive | 1202.6100 | 0.114 | 0.115 |
| 2 Coalesced | 168.8200 | 0.814 | 0.818 |
| 3 SMEM | 154.9630 | 0.887 | 0.894 |
| 4 1D Register | 40.7154 | 3.376 | 3.441 |
| 5 2D Register | 32.3371 | 4.250 | 4.322 |
| 6 Vectorized | 18.2516 | 7.530 | 8.022 |
| 7 Padding | 16.3700 | 8.396 | 8.977 |
| 8 Autotuned C08 | 15.4178 | 8.914 | 9.406 |
| 10 Warp Tiling | 20.6457 | 6.657 | 6.932 |
| 11 Double Buffering | 15.6738 | 8.769 | 9.224 |
| cuBLAS FP32 | 15.7804 | 8.709 | 9.254 |

本轮最佳自定义 Avg 相对 Naive 为 `78.0×`。C08 Avg 比同轮 cuBLAS FP32 高
约 `2.35%`，但 GPU 时钟、功耗和温度未锁定，单次顺序运行不能证明稳定胜出。
原始数据见
[`results/rtx4060_laptop/comparison_4096.csv`](./results/rtx4060_laptop/comparison_4096.csv)。

## Benchmark Commands

```bash
# 全阶段 correctness + 同轮 4096³ FP32 比较
./scripts/run_comparison.sh

# 单独运行一个阶段
./scripts/benchmark_stage.sh vectorized 4096 4096 4096

# 单独运行完整 Stage 8 autotuning suite
./8.Autoing_kernel/run_autotune.sh full
```

统一脚本从仓库根目录运行；可用 `WARMUP` 和 `ITERS` 环境变量覆盖默认值。

## Historical Kernel Performance Table

统一比较 Avg GFLOPS：

| Case              |    Naive | Coalesced |      SMEM | 1D Register | 2D Register | Vectorized |   Padding |
| ----------------- | -------: | --------: | --------: | ----------: | ----------: | ---------: | --------: |
| `256^3`           |  92.2776 |  616.2947 |  640.7902 |   1258.8250 |    801.0145 |  1335.4937 | 1185.0726 |
| `512^3`           | 109.5825 |  830.3416 |  836.6070 |   2676.5946 |   2431.5510 |  4243.2712 | 3058.6335 |
| `1024^3`          | 123.4315 |  886.0027 |  904.6960 |   3397.1641 |   3395.6509 |  6412.4137 | 7429.6795 |
| `2048^3`          | 123.5758 | 1026.0384 | 1039.4091 |   3698.2388 |   4054.4146 |  7290.9726 | 8471.1701 |
| `4096^3`          | 123.0080 |  870.8718 |  958.3544 |   3659.7348 |   4505.6055 |  7802.2829 | 8581.5961 |
| `1023^3`          | 407.9408 |  941.6104 |  961.0163 |   3705.5630 |   3768.9978 |          — |         — |
| `4096x256x4096`   | 116.5439 | 1013.2102 |  941.9762 |   3596.3530 |   4121.9961 |  6726.1882 | 6882.4758 |
| `256x4096x4096`   | 123.6838 |  879.4382 |  936.1697 |   3463.6959 |   3904.3388 |  6447.6574 | 8063.4148 |

Vectorized 和 Padding kernel 都不支持 `1023^3`：当前实现没有边界分支，并要求
`M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`。

## Relative to Previous Kernel

相对上一阶段的 Avg GFLOPS 变化：

| Case              | Coalesced vs Naive | SMEM vs Coalesced | 1D vs SMEM | 2D vs 1D | Vectorized vs 2D | Padding vs Vectorized |
| ----------------- | -----------------: | ----------------: | ---------: | -------: | ---------------: | --------------------: |
| `256^3`           |            +567.9% |             +4.0% |     +96.4% |   -36.4% |           +66.7% |                -11.3% |
| `512^3`           |            +657.7% |             +0.8% |    +219.9% |    -9.2% |           +74.5% |                -27.9% |
| `1024^3`          |            +617.8% |             +2.1% |    +275.5% |   -0.04% |           +88.8% |                +15.9% |
| `2048^3`          |            +730.3% |             +1.3% |    +255.8% |    +9.6% |           +79.8% |                +16.2% |
| `4096^3`          |            +608.0% |            +10.0% |    +281.9% |   +23.1% |           +73.2% |                +10.0% |
| `1023^3`          |            +130.8% |             +2.1% |    +285.6% |    +1.7% |                — |                     — |
| `4096x256x4096`   |            +769.4% |             -7.0% |    +281.8% |   +14.6% |           +63.2% |                 +2.3% |
| `256x4096x4096`   |            +611.0% |             +6.5% |    +270.0% |   +12.7% |           +65.1% |                +25.1% |

## Vectorized Detailed Result

| Case              | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error |
| ----------------- | :---: | -------: | -------: | -------: | ----------: | -----------: | --------: |
| `256^3`           | PASS  |   0.0225 |   0.0251 |   0.0307 |   1335.4937 |    1489.4545 |    0.0000 |
| `512^3`           | PASS  |   0.0594 |   0.0633 |   0.0819 |   4243.2712 |    4519.7240 |    0.0000 |
| `1024^3`          | PASS  |   0.3154 |   0.3349 |   0.4536 |   6412.4137 |    6808.9353 |    0.0001 |
| `2048^3`          | PASS  |   2.3040 |   2.3563 |   2.7545 |   7290.9726 |    7456.5408 |    0.0002 |
| `4096^3`          | PASS  |  16.4045 |  17.6152 |  20.0703 |   7802.2829 |    8378.1353 |    0.0004 |
| `4096x256x4096`   | PASS  |   1.2493 |   1.2771 |   1.4950 |   6726.1882 |    6875.9083 |    0.0004 |
| `256x4096x4096`   | PASS  |   1.3015 |   1.3323 |   1.5400 |   6447.6574 |    6600.0062 |    0.0004 |

## Shared-memory Padding Detailed Result

As/Bs 的物理行步长分别为 `BM + 4` 和 `BN + 4`，合计使用 4352 bytes
static shared memory。默认 `max_check_dim=1024`，所以较大用例的 `SKIP` 是
跳过 CPU reference，并非 correctness failure。

| Case              | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error |
| ----------------- | :---: | -------: | -------: | -------: | ----------: | -----------: | --------: |
| `256^3`           | PASS  |   0.0205 |   0.0283 |   0.0563 |   1185.0726 |    1638.4000 |    0.0000 |
| `512^3`           | PASS  |   0.0502 |   0.0878 |   0.2990 |   3058.6335 |    5349.8777 |    0.0000 |
| `1024^3`          | PASS  |   0.2796 |   0.2890 |   0.3123 |   7429.6795 |    7681.8751 |    0.0000 |
| `2048^3`          | SKIP  |   1.9925 |   2.0280 |   2.6010 |   8471.1701 |    8622.2162 |    0.0000 |
| `4096^3`          | SKIP  |  14.6954 |  16.0155 |  18.5119 |   8581.5961 |    9352.5000 |    0.0000 |
| `4096x256x4096`   | SKIP  |   1.1262 |   1.2481 |   1.5729 |   6882.4758 |    7627.3077 |    0.0000 |
| `256x4096x4096`   | SKIP  |   0.9810 |   1.0653 |   1.4336 |   8063.4148 |    8756.3756 |    0.0000 |

单独复测 `4096^3` 得到 `15.2979 ms`、`8984.1773 GFLOPS` 的平均结果。
为了保持逐尺寸表的同轮测试口径，Stage Summary 仍采用完整 suite 中的
`8581.5961 GFLOPS`。

## Stage Summary

| Stage              | Main Change                                 | `4096^3` Avg Result |
| ------------------ | ------------------------------------------- | ------------------: |
| Naive              | direct global-memory implementation         |    `0.123 TFLOPS` |
| Coalesced          | coalesced global-memory access              |    `0.871 TFLOPS` |
| SMEM               | shared-memory tile reuse                    |    `0.958 TFLOPS` |
| 1D Register Tiling | each thread computes `TM` results           |    `3.660 TFLOPS` |
| 2D Register Tiling | each thread computes a `TM x TN` micro tile |    `4.506 TFLOPS` |
| Vectorized         | `float4` load/store and transposed A tile   |    `7.802 TFLOPS` |
| SMEM Padding       | padded As/Bs physical shared-memory stride  |    `8.582 TFLOPS` |
| Autotuned C08      | `BM=128, BN=64, BK=16, TM=8, TN=8`         |    `9.665 TFLOPS` |

## Shared-memory Padding vs Vectorized Notes

- `1024^3`、`2048^3`、`4096^3` 分别提升 `15.9%`、`16.2%`、`10.0%`
- `256x4096x4096` 提升最大，为 `25.1%`
- 完整 suite 的 `4096^3` Avg latency 从 `17.6152 ms` 降到 `16.0155 ms`
- 单独复测的 `4096^3` 达到 `8984.1773 GFLOPS`，但不替换完整 suite 的主记录
- `256^3`、`512^3` 出现回退；这类极短 kernel 更容易受到 launch、频率和
  Laptop GPU 功耗状态波动影响
- 三个实际执行 CPU reference 的尺寸全部 `PASS`

Stage 6 的 Nsight Compute 报告显示 shared load/store bank conflict 是主要剩余
问题之一。本阶段依次为 Bs 和 As 调整物理布局，经过多轮 profiler 验证后，
主要 conflict 已基本消除，大尺寸吞吐也相应提高。当前先完成 padding 方法；
XOR swizzle 留作后续扩展学习。

## Stage 8 Autotuning Result

Stage 8 不改变 Stage 7 kernel 的计算逻辑，而是比较 14 组编译期 tile 参数。
下表记录完整 suite 中综合排名第一的 C08 与同轮 C00 基准；数值来自
`8.Autoing_kernel/results/autotune_full.csv`。

| Case | C00 Avg GFLOPS | C08 Avg GFLOPS | Change |
| ---- | --------------: | --------------: | -----: |
| `256^3` | 1161.39 | 1323.16 | +13.9% |
| `512^3` | 4793.33 | 4403.70 | -8.1% |
| `1024^3` | 7194.59 | 8764.70 | +21.8% |
| `2048^3` | 9776.41 | 9614.12 | -1.7% |
| `4096^3` | 9103.69 | 9664.98 | +6.2% |
| `4096x256x4096` | 7308.30 | 8865.61 | +21.3% |
| `256x4096x4096` | 8396.84 | 8942.19 | +6.5% |

综合排名使用相对 C00 的几何平均加速比：

| Rank | Config | Parameters | Geomean speedup |
| ---: | --- | --- | ---: |
| 1 | C08 | `BM=128, BN=64, BK=16, TM=8, TN=8` | `1.0805x` |
| 2 | C13 | `BM=64, BN=64, BK=16, TM=8, TN=4` | `1.0631x` |
| 3 | C11 | `BM=64, BN=64, BK=8, TM=8, TN=4` | `1.0312x` |

C08 的 `4096^3` Avg 为 `9.665 TFLOPS`。这是单轮 Laptop GPU autotune 结果，
可作为下一版固定参数的候选，不能当作跨设备或所有矩阵尺寸的全局最优。

## Stage 10 Warp Tiling Final Result

Stage 10 引入 warp-level 输出分工，但当前实现的寄存器压力和 shared-memory store
conflict 抵消了预期收益。以下是最终 suite 记录；correctness 在独立运行中验证为
`PASS`，表中数值只表示 CUDA Event 测速。

| Case | Avg GFLOPS | Best GFLOPS |
| ---- | ----------: | -----------: |
| `1024^3` | 6313.0621 | 6553.6002 |
| `2048^3` | 6578.1271 | 7760.0446 |
| `4096^3` | 6670.5097 | 6873.8835 |

独立 `4096^3` 复测为 6748.1435 Avg / 6990.1429 Best GFLOPS，不替换同轮 suite
的主记录。Nsight Compute 的 `1024^3` full report 显示 168 registers/thread、25%
理论 occupancy、22.33% achieved occupancy，并观察到 shared-store 平均 4-way
conflict。这是负优化分析，而不是 Stage 8 之后的自动加速。

## Stage 11 Double Buffering Final Result

Stage 11 使用寄存器预取配合 ping-pong shared-memory buffer。它不是 `cp.async`，
也没有继承 Stage 10 的 warp tile 参数，因此应视为另一条实验分支。correctness
同样在独立运行中验证为 `PASS`。

| Case | Avg GFLOPS | Best GFLOPS |
| ---- | ----------: | -----------: |
| `1024^3` | 7083.9711 | 7307.1498 |
| `2048^3` | 7618.0570 | 8380.2273 |
| `4096^3` | 8722.6411 | 9129.2158 |

独立 `4096^3` 复测为 8854.6274 Avg / 9012.7404 Best GFLOPS，不替换 suite
主记录。`1024^3` profiler 报告显示 128 registers/thread、33.33% 理论 occupancy、
28.93% achieved occupancy，并仍观察到 shared-load conflict。Profiler 中的
301.92 us 是 replay 采集上下文，不作为上述 benchmark latency。

## Historical cuBLAS Baseline

主基准使用 FP32 输入、输出和 `CUBLAS_COMPUTE_32F`。由于项目矩阵按 row-major
存储，wrapper 利用转置等价关系调用 column-major cuBLAS；计时仅覆盖 GEMM。

| Mode | Run | Avg GFLOPS | Best GFLOPS | Usage |
| ---- | --- | ----------: | -----------: | ----- |
| FP32 | final validation | 9429.4001 | 9553.0118 | 主基准 |
| FP32 | independent normal | 9357.0652 | 9532.5090 | 稳定性参考 |
| FP32 | `NVIDIA_TF32_OVERRIDE=0` | 9308.8622 | 9548.8918 | 环境对照 |
| TF32 | fast TF32 | 14844.7540 | 15925.2171 | Tensor Core 独立参考 |

这些行是统一脚本加入前的历史记录。合理表述是：cuBLAS FP32 多轮 Avg 约
9.3–9.4 TFLOPS，与 Stage 8 C08
的最佳自定义记录处于同一波动区间，现有不同轮次数据不足以证明自定义 kernel
稳定超过 cuBLAS。TF32 精度和执行路径不同，不能与 FP32 表直接排名。

## Historical v1.0 4096³ Summary

机器可读历史汇总见 [`results/final_4096.csv`](./results/final_4096.csv)。这些行
来自不同实验轮次，全部使用 Avg：

| Stage | FP32 Avg TFLOPS |
| ----: | ---------------: |
| 1 Naive | 0.123 |
| 2 Coalesced | 0.871 |
| 3 SMEM | 0.958 |
| 4 1D Register Tiling | 3.660 |
| 5 2D Register Tiling | 4.506 |
| 6 Vectorized | 7.802 |
| 7 Padding | 8.582 |
| 8 Autotuned C08 | 9.665 |
| 10 Warp Tiling | 6.671 |
| 11 Double Buffering | 8.723 |
| cuBLAS FP32 | 9.429 |

## Notes

- 数据用于学习和趋势观察，不代表工业级最终性能
- 当前项目不声明达到或替代 cuBLAS
- Vectorized、Padding 与旧阶段来自不同测试轮次，Laptop GPU 状态会带来一定波动
- 对 Vectorized 和 Padding kernel，不应使用不满足 tile/对齐约束的尺寸做
  correctness 结论
- Padding 表中 `SKIP` 用例没有执行 CPU reference，不能仅根据 `max error=0`
  声明其 correctness 已验证

## Source READMEs

- [Stage 1: Naive](./1.naive_kernel/README.md)
- [Stage 2: Global memory coalescing](./2.Global_Memory_Coalescing_kernel/README.md)
- [Stage 3: Shared memory tiling](./3.SMEM_kernel/README.md)
- [Stage 4: 1D register tiling](./4.1D_Blocktiling_kernel/README.md)
- [Stage 5: 2D register tiling](./5.2D_Blocktiling_kernel/README.md)
- [Stage 6: Vectorized memory access](./6.Vectorize_kernel/README.md)
- [Stage 7: Shared-memory layout padding](./7.Shared_Memory_Layout_Optimization/README.md)
- [Stage 8: Compile-time autotuning](./8.Autoing_kernel/README.md)
- [Stage 10: Warp tiling](./10.Wraptiling_kernel/README.md)
- [Stage 11: Double buffering](./11.Double_Buffering/README.md)
- [cuBLAS baseline](./cuBLAS_baseline/README.md)
