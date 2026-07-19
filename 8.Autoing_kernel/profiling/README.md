# Stage 8 Profiling

本目录记录 autotuning 阶段的代表性 Nsight Compute 结果。报告对比的是：

- `C00`: `BM=64, BN=64, BK=8, TM=8, TN=8`
- `C08`: `BM=128, BN=64, BK=16, TM=8, TN=8`

两者都使用 Stage 7 的 As/Bs padding 布局；Stage 8 只改变编译期 tile 参数。
正式性能数据仍以普通 benchmark 的 CUDA event 结果为准，Nsight Compute replay
后的 latency 不用于性能排名。

## 报告和截图

| Category | Screenshot | What it records |
| --- | --- | --- |
| Speed of Light | [02_speed_of_light.png](./full_report/figures/02_speed_of_light.png) | SM、memory 和 DRAM 吞吐占理论峰值的比例 |
| Memory Workload | [03_memory_workload.png](./full_report/figures/03_memory_workload.png) | global store 利用率与 shared-store bank conflict 提示 |
| Warp State | [05_warp_state.png](./full_report/figures/05_warp_state.png) | warp 未被调度、dispatch 和 scoreboard 等 stall 状态 |
| Launch Statistics | [06_launch_statistics.png](./full_report/figures/06_launch_statistics.png) | block size、register/thread、shared memory 和 occupancy 相关数据 |

原始报告：

- [`ncu_C00_4096.ncu-rep`](../ncu_C00_4096.ncu-rep)
- [`ncu_C08_4096.ncu-rep`](../ncu_C08_4096.ncu-rep)

截图文件按指标分类并采用与其他 kernel 阶段相同的英文命名，避免保留来自聊天
工具的临时文件名。

## 解读

当前截图显示 C08 的计算和内存吞吐都没有达到理论峰值，shared-memory store
仍有可继续分析的 bank-conflict 提示；因此本阶段结论是“tile 搜索找到更好的
候选”，而不是“所有访存瓶颈已经解决”。后续可围绕 warp-level load mapping、
double buffering 和 `cp.async` 做更有针对性的实验。

Nsight Compute 采集应针对单个模板实例，并使用与普通 benchmark 相同的架构：

```bash
cd /home/fish/GEMM_For_Myself/8.Autoing_kernel

nvcc -O3 -std=c++17 -lineinfo -arch=sm_89 \
  --ptxas-options=-v autotune_padding_benchmark.cu \
  -o autotune_padding_bench

# 根据目标模板实例的 demangled kernel name 调整 --kernel-name。
ncu -f --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-count 1 \
  -o ncu_stage8 \
  ./autotune_padding_bench --suite quick --warmup 0 --iters 1 \
  --no-verify --csv /tmp/stage8_ncu.csv
```

`--kernel-name` 过宽时会匹配多个模板实例；需要复现 C00 或 C08 报告时，应将
正则收窄到对应的 demangled 模板参数。
