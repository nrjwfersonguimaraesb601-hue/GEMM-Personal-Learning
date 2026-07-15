# 2D Register Tiling Kernel

## Idea

2D Register Tiling 是在 1D Register Tiling 之后继续提高数据复用的一步。

核心思路：

- 一个 block 负责 C 的 `BM x BN` 大块
- shared memory 中缓存 `As[BM x BK]` 和 `Bs[BK x BN]`
- 每个 thread 负责一个 `TM x TN` 的 C 小块
- 每次 `dotIdx` 中，从 `As` 取 `regM[TM]`，从 `Bs` 取 `regN[TN]`
- `regM` 和 `regN` 做外积，更新 `threadRes[TM x TN]`

相比 1D Register Tiling 每个 thread 只计算 `TM` 个结果，2D Register Tiling 让每个 thread 计算 `TM x TN` 个结果，理论上可以提高数据复用和算术强度。

当前配置：

| Parameter  |    Value |
| ---------- | -------: |
| `BM`       |       64 |
| `BN`       |       64 |
| `BK`       |        8 |
| `TM`       |        8 |
| `TN`       |        8 |
| `blockDim` | `64 x 1` |

## Build

```bash
nvcc 2D_Blocktiling_kernel_benchmark.cu -O3 -lineinfo -o 2D_Blocktiling_bench
```

## Run Benchmark

```bash
./2D_Blocktiling_bench \
  --warmup 10 --iters 50 --bx 64 --by 1 --no-check
```

当前性能表使用 `--no-check`，主要用于 benchmark 性能记录。正确性需要单独去掉 `--no-check` 开启 CPU check 验证。

## Benchmark Result

| M    | N    | K    | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
| ---- | ---- | ---- | ----- | -------: | ---------: | ----------: |
| 256  | 256  | 256  | 64x1  |   0.0419 |   801.0145 |    862.3158 |
| 512  | 512  | 512  | 64x1  |   0.1104 |  2431.5510 |   2788.7659 |
| 1024 | 1024 | 1024 | 64x1  |   0.6324 |  3395.6509 |   3512.8174 |
| 2048 | 2048 | 2048 | 64x1  |   4.2373 |  4054.4146 |   4602.8027 |
| 4096 | 4096 | 4096 | 64x1  |  30.5040 |  4505.6055 |   4609.2836 |
| 1023 | 1023 | 1023 | 64x1  |   0.5681 |  3768.9978 |   4092.0040 |
| 4096 | 256  | 4096 | 64x1  |   2.0839 |  4121.9961 |   4247.3967 |
| 256  | 4096 | 4096 | 64x1  |   2.2001 |  3904.3388 |   4015.6697 |

## Compared with 1D Register Tiling

| Case              | 1D Avg GFLOPS | 2D Avg GFLOPS | Change |
| ----------------- | ------------: | ------------: | -----: |
| 256^3             |     1258.8250 |      801.0145 | -36.4% |
| 512^3             |     2676.5946 |     2431.5510 |  -9.2% |
| 1024^3            |     3397.1641 |     3395.6509 | -0.04% |
| 2048^3            |     3698.2388 |     4054.4146 |  +9.6% |
| 4096^3            |     3659.7348 |     4505.6055 | +23.1% |
| 4096 x 256 x 4096 |     3596.3530 |     4121.9961 | +14.6% |
| 256 x 4096 x 4096 |     3463.6959 |     3904.3388 | +12.7% |

观察：

- 小矩阵上 2D 不一定更快，可能受到 kernel 开销、寄存器压力、写回模式和 shared memory 访问模式影响
- `1024^3` 基本和 1D 持平
- 大矩阵上 2D 开始体现优势
- `4096^3` 从约 `3.66 TFLOPS` 提升到约 `4.51 TFLOPS`，提升约 `23.1%`
- 当前提升低于参考文章中接近 2 倍的提升，说明基础 2D 思路已经生效，但实现仍有明显优化空间

## Nsight Compute Findings

Full report:

```bash
ncu -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name regex:.*sgemm_2d.* \
  --launch-skip 1 \
  --launch-count 1 \
  -o 2D_Blocktiling_full \
  ./2D_Blocktiling_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check
```

InstructionStats:

```bash
ncu -f \
  --kernel-name-base demangled \
  --kernel-name regex:.*sgemm_2d.* \
  --section InstructionStats \
  --launch-skip 1 \
  --launch-count 1 \
  -o 2D_Blocktiling_InstructionStats \
  ./2D_Blocktiling_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --bx 64 --by 1 --no-check
```

如果 kernel 名称不是 `sgemm_2d_register_tiling`，而是 `calculate_Matrix`，需要把 regex 改成：

```bash
--kernel-name regex:.*calculate_Matrix.*
```

当前 Nsight 关键指标：

| Metric                   |       Value |
| ------------------------ | ----------: |
| Compute Throughput       |      42.63% |
| Memory Throughput        |      84.73% |
| L1/TEX Cache Throughput  |      91.95% |
| L2 Cache Throughput      |      11.85% |
| DRAM Throughput          |       5.56% |
| Registers/thread         |         116 |
| Block Size               |          64 |
| SM Busy                  |      35.99% |
| Issue Slots Busy         |      35.99% |
| FMA pipeline utilization | about 28.4% |

解释：

- `DRAM Throughput` 很低，说明主要瓶颈不是显存带宽
- `L1/TEX Cache Throughput` 很高，说明压力集中在片上 memory path
- `Registers/thread = 116`，说明 2D register tiling 带来较大寄存器压力
- `SM Busy` 和 `Issue Slots Busy` 不高，说明计算管线没有完全打满
- 当前 2D kernel 的收益被 shared memory 访问问题、global store pattern 和 register pressure 抵消了一部分

Nsight Summary 提示的主要问题：

- `L1TEX Global Store Access Pattern`: C 写回访问模式不理想，平均每个 sector 只有约 `4.0 / 32 bytes` 被有效利用
- `Shared Load Bank Conflicts`: 平均约 `4.8-way` bank conflict，bank conflicts 约 `16,777,216`，约占 overall wavefronts 的 `66.67%`
- `Uncoalesced Shared Accesses`: shared memory access 不够合并，产生 excessive wavefronts

这些问题可能对应：

```cpp
C[(threadRow * TM + i) * N + threadCol * TN + j] = threadRes[i * TN + j];

regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
```

## Current Limitations

- 当前性能表主要来自 `--no-check` 模式，正确性需要单独开启 CPU check
- 2D 在大矩阵上相对 1D 有提升，但没有达到参考文章中的接近 2 倍
- 当前存在 shared memory bank conflict
- 当前 C 写回 global memory 的 store pattern 不理想
- `Registers/thread` 较高，可能限制 occupancy 和 warp 调度能力
- `1023` 这类非整除尺寸不能单独作为 correctness 依据；如果某个实验 kernel 没有完整边界处理，非整除尺寸可能存在越界或漏算风险

## Future Work

当前阶段只记录方向，不改代码：

- shared memory padding，例如 `As[BM * (BK + 1)]`、`Bs[BK * (BN + 1)]`，尝试减少 bank conflict
- shared memory layout transpose，减少 shared memory 读取冲突
- vectorized memory access，例如 `float4` load/store
- 继续调参 `BM / BN / BK / TM / TN`
- 尝试 `BM=128, BN=128, BK=8, TM=8, TN=8`，对应 `blockDim.x=256`
- 尝试 `TM=8, TN=4`，降低寄存器压力
- 对比 1D 和 2D Nsight 报告中的 `FFMA`、`LDS`、`LDG`、`STG`、`Registers/thread`、`Occupancy`、`Stall MIO Throttle`
