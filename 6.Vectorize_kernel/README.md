# Vectorized GEMM Kernel

这一阶段在 2D Register Tiling 的基础上加入 `float4` 向量化访存，目标是减少
global-memory load/store 指令数量，并改善 shared-memory 中 A tile 的读取方式。

## 技术思路

- 一个 block 负责 `64 x 64` 的 C tile，共使用 `64` 个线程
- 每个线程在寄存器中累加一个 `8 x 8` micro tile
- A、B 从 global memory 以 `float4` 为单位加载
- A 在写入 shared memory 时转置为 `As[k][row]`，计算阶段可以按行广播读取
- B 保持 `Bs[k][col]` 布局
- C 每次用一个 `float4` 写回连续的 4 个结果

当前 SASS 中可以观察到 `LDG.E.128` 和 `STG.E.128`，说明 128-bit
load/store 已实际生成，而不只是源码层面的类型转换。

| Parameter  |    Value |
| ---------- | -------: |
| `BM`       |       64 |
| `BN`       |       64 |
| `BK`       |        8 |
| `TM`       |        8 |
| `TN`       |        8 |
| `blockDim` | `64 x 1` |

当前 kernel 没有处理边界，输入必须满足：

```text
M % 64 == 0, N % 64 == 0, K % 8 == 0
```

## 编译与运行

```bash
nvcc -O3 -lineinfo -arch=sm_89 \
  Vectorize_kernel_benchmark.cu \
  -o Vectorize_bench

./Vectorize_bench \
  --warmup 10 --iters 50 --bx 64 --by 1
```

不传位置参数 `M N K` 时会运行内置的多尺寸 benchmark。上述命令保留 CPU
reference check；校验位于 GPU event 计时之外，不计入 kernel latency。

## 实测结果

测试环境：RTX 4060 Laptop GPU，Compute Capability 8.9，24 SM，8 GiB。
全部用例均通过正确性校验。

| M    | N    | K    | Check | Avg (ms) | Avg GFLOPS | Best GFLOPS | Max Error |
| ---- | ---- | ---- | :---: | -------: | ----------: | -----------: | --------: |
| 256  | 256  | 256  | PASS  |   0.0251 |   1335.4937 |    1489.4545 |    0.0000 |
| 512  | 512  | 512  | PASS  |   0.0633 |   4243.2712 |    4519.7240 |    0.0000 |
| 1024 | 1024 | 1024 | PASS  |   0.3349 |   6412.4137 |    6808.9353 |    0.0001 |
| 2048 | 2048 | 2048 | PASS  |   2.3563 |   7290.9726 |    7456.5408 |    0.0002 |
| 4096 | 4096 | 4096 | PASS  |  17.6152 |   7802.2829 |    8378.1353 |    0.0004 |
| 4096 | 256  | 4096 | PASS  |   1.2771 |   6726.1882 |    6875.9083 |    0.0004 |
| 256  | 4096 | 4096 | PASS  |   1.3323 |   6447.6574 |    6600.0062 |    0.0004 |

## 与 2D Register Tiling 对比

以下统一比较 Avg GFLOPS：

| Case              | 2D Tiling | Vectorized | Change |
| ----------------- | --------: | ---------: | -----: |
| `256^3`           |  801.0145 |  1335.4937 | +66.7% |
| `512^3`           | 2431.5510 |  4243.2712 | +74.5% |
| `1024^3`          | 3395.6509 |  6412.4137 | +88.8% |
| `2048^3`          | 4054.4146 |  7290.9726 | +79.8% |
| `4096^3`          | 4505.6055 |  7802.2829 | +73.2% |
| `4096x256x4096`   | 4121.9961 |  6726.1882 | +63.2% |
| `256x4096x4096`   | 3904.3388 |  6447.6574 | +65.1% |

向量化版本在所有共同合法尺寸上都有明显提升。`4096^3` 的平均吞吐从
`4.51 TFLOPS` 提升到 `7.80 TFLOPS`，平均延迟从 `30.5040 ms` 降到
`17.6152 ms`。这说明 A/B/C 的访存方式和 A tile 的 shared-memory 布局，
确实是上一阶段的重要限制之一。

## Nsight Compute 结论

- Compute throughput `54.20%`，高于 2D 版本的 `42.63%`
- global store 的 sector 有效字节从 `4/32 B` 提升到 `16/32 B`
- `117 registers/thread`，与 2D 版本的 `116` 基本相同
- shared load 仍有约 `5.0-way` bank conflict，shared store 约 `2.4-way`
- MIO Throttle 从 `2.98` 降到 `0.86 cycles/instruction`，但仍存在 scoreboard 和 barrier stall

完整截图与逐项说明见 [profiling/README.md](./profiling/README.md)。

## 当前限制与后续工作

- 改进 A 的协作式 global-load 映射；当前每个线程读取 `float4`，但一个 warp 的 A 地址仍不完全连续
- 调整 C 的线程映射，继续提高 global-store sector 利用率
- 改善 shared-memory layout，降低 load/store bank conflict
- 增加非整除尺寸的边界路径，或为尾部 tile 提供单独 fallback kernel
- 分别关闭 A/B load、A transpose 和 C store 的向量化，做 ablation 对比
- 继续测试 `BM/BN/BK/TM/TN`，平衡吞吐、寄存器压力和 occupancy
- 增加 cuBLAS baseline，并继续尝试 warp tiling / double buffering

Nsight Compute 的编译、采集与打开命令见
[profiling/capture_commands.md](./profiling/capture_commands.md)。
