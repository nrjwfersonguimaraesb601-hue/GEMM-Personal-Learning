# Shared-memory Layout Optimization with Padding

这一阶段延续 Stage 6 的 vectorized GEMM，主要目标是根据 Nsight Compute 的
分析结果，降低 shared-memory bank conflict。当前先掌握最直接、容易验证的
padding 方法；XOR swizzle 保留为以后深入 shared-memory layout 时的扩展内容，
本阶段不实现。

## 核心思路

Stage 6 已经使用 `float4` 完成 A/B 的 global load 和 C 的 global store，但
Nsight Compute 仍显示 shared-memory load/store 存在明显 bank conflict。
本阶段不改变 block tile、thread tile 和计算流程，只改变 shared-memory 的
物理布局：

- A 的逻辑布局仍是转置后的 `As[BK][BM]`，物理行步长改为 `BM + 4`
- B 的逻辑布局仍是 `Bs[BK][BN]`，物理行步长改为 `BN + 4`
- B 的 4 个 padding 元素插在一行中间，使两半数据在 bank 上错开
- As/Bs 的写入和读取统一使用对应的物理 stride
- global-memory 访问、register tiling 和最终 C 写回方式保持不变

当前布局为：

```text
As: [BK][BM + 4] = [8][68]
Bs: [BK][BN + 4] = [8][68]
```

两个 tile 共占用：

```text
8 * (68 + 68) * sizeof(float) = 4352 bytes
```

相比无 padding 的 4096 bytes，只增加 256 bytes static shared memory。经过多轮
Nsight Compute 分析和布局调整，原先主要的 shared-memory bank conflict 已经
基本消除，同时大尺寸 GEMM 性能继续提升。

## 文件说明

- `Shared_Memory_Layout_Padding_kernel.cu`：本阶段完成的 As/Bs padding kernel
- `Shared_Memory_Layout_Padding_benchmark.cu`：正确性检查和多尺寸性能测试
- `Shared_Memory_Layout_XOR_Swizzle_kernel.cu`：仅预留文件，暂未实现

## Kernel 参数与限制

| Parameter | Value |
| --------- | ----: |
| `BM` | 64 |
| `BN` | 64 |
| `BK` | 8 |
| `TM` | 8 |
| `TN` | 8 |
| `blockDim` | `64 x 1` |
| Static shared memory | `4352 bytes` |

当前 kernel 没有边界分支，输入尺寸必须满足：

```text
M % 64 == 0, N % 64 == 0, K % 8 == 0
```

## 编译与运行

测试 GPU 为 RTX 4060 Laptop GPU，Compute Capability 8.9：

```bash
cd /home/fish/GEMM_For_Myself/7.Shared_Memory_Layout_Optimization

nvcc -O3 -lineinfo -arch=sm_89 \
  Shared_Memory_Layout_Padding_benchmark.cu \
  -o Shared_Memory_Layout_Padding_bench
```

先进行正确性检查：

```bash
./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 10 --iters 50
```

运行内置多尺寸 benchmark：

```bash
./Shared_Memory_Layout_Padding_bench \
  --warmup 10 --iters 50
```

`max_check_dim` 默认为 1024，所以 `256³`、`512³`、`1024³` 会进行 CPU
reference check；更大的用例显示 `SKIP`，表示为了避免 CPU 校验时间过长而
跳过，并不表示计算失败。纯测速时可以添加 `--no-check`。

## 实测结果

测试环境：RTX 4060 Laptop GPU，warmup 10 次，正式迭代 50 次。

| Case | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error |
| ---- | :---: | -------: | -------: | -------: | ----------: | -----------: | --------: |
| `256³` | PASS | 0.0205 | 0.0283 | 0.0563 | 1185.0726 | 1638.4000 | 0.0000 |
| `512³` | PASS | 0.0502 | 0.0878 | 0.2990 | 3058.6335 | 5349.8777 | 0.0000 |
| `1024³` | PASS | 0.2796 | 0.2890 | 0.3123 | 7429.6795 | 7681.8751 | 0.0000 |
| `2048³` | SKIP | 1.9925 | 2.0280 | 2.6010 | 8471.1701 | 8622.2162 | 0.0000 |
| `4096³` | SKIP | 14.6954 | 16.0155 | 18.5119 | 8581.5961 | 9352.5000 | 0.0000 |
| `4096x256x4096` | SKIP | 1.1262 | 1.2481 | 1.5729 | 6882.4758 | 7627.3077 | 0.0000 |
| `256x4096x4096` | SKIP | 0.9810 | 1.0653 | 1.4336 | 8063.4148 | 8756.3756 | 0.0000 |

另一次只运行 `4096³` 的复测结果为：

| Case | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS |
| ---- | -------: | -------: | -------: | ----------: | -----------: |
| `4096³` | 14.7843 | 15.2979 | 18.1441 | 8984.1773 | 9296.3050 |

统一性能汇总使用完整 suite 中的 `8581.5961 GFLOPS`；单独复测说明 Laptop GPU
状态较好时，当前 kernel 的平均吞吐可以接近 `9.0 TFLOPS`。

## 与 Stage 6 Vectorized 对比

以下统一比较 Avg GFLOPS：

| Case | Vectorized | Padding | Change |
| ---- | ---------: | ------: | -----: |
| `256³` | 1335.4937 | 1185.0726 | -11.3% |
| `512³` | 4243.2712 | 3058.6335 | -27.9% |
| `1024³` | 6412.4137 | 7429.6795 | +15.9% |
| `2048³` | 7290.9726 | 8471.1701 | +16.2% |
| `4096³` | 7802.2829 | 8581.5961 | +10.0% |
| `4096x256x4096` | 6726.1882 | 6882.4758 | +2.3% |
| `256x4096x4096` | 6447.6574 | 8063.4148 | +25.1% |

`1024³` 及更大的主要用例都有提升。完整 suite 中 `4096³` 从
`7.802 TFLOPS` 提升到 `8.582 TFLOPS`；单独复测达到 `8.984 TFLOPS`。
`256³`、`512³` 的 kernel 时间很短，更容易受到 launch 开销、GPU 频率和
Laptop 功耗状态影响，因此本阶段主要依据大尺寸结果和 Nsight 指标判断优化效果。

## Nsight Compute 分析结论

- Stage 6 暴露出的 shared-memory bank conflict 是本阶段的直接优化目标
- 先对 Bs 添加 padding，再调整 As 的物理 stride，并反复检查相关 wavefront
  和 bank-conflict 指标
- 当前 As/Bs padding 后，主要 conflict 已经基本消除
- 大尺寸吞吐随之提升，说明 shared-memory layout 确实是上一阶段的性能限制之一
- padding 逻辑简单、地址映射直观，适合作为现阶段掌握 bank conflict 的方法

## Nsight Compute 报告命令

Instruction Statistics：

```bash
ncu -f \
  --section InstructionStats \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Shared_Memory_Layout_Padding_instr \
  ./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --no-check
```

Full report：

```bash
ncu -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-skip 1 \
  --launch-count 1 \
  -o Shared_Memory_Layout_Padding_full \
  ./Shared_Memory_Layout_Padding_bench 1024 1024 1024 \
  --warmup 1 --iters 3 --no-check
```

`ncu` 会 replay 被采集的 kernel，因此在 profiler 下打印的 latency 不用于
性能对比。正式性能数据应来自不带 `ncu` 的普通 benchmark 运行。

截图分类、逐项指标解释与 Stage 6 对比见
[profiling/README.md](./profiling/README.md)。

## 当前结论与后续方向

本阶段已经完成预定目标：理解 shared-memory bank conflict，学会根据 Nsight
Compute 定位问题，并使用 padding 调整 As/Bs 的物理布局。XOR swizzle 暂时不学，
以后需要进一步研究 shared-memory layout 时再作为扩展实验。

后续主线将转向：

- 改善 A 的 warp-level global-memory 加载映射
- 支持非整除尺寸和边界 tile
- 尝试 warp tiling、double buffering 和 `cp.async`
- 增加统一的 cuBLAS baseline
