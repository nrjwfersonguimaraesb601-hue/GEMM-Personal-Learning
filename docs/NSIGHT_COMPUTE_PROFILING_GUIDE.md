# Nsight Compute Profiling Guide

本项目把正式 benchmark 与 Nsight Compute profiling 分开：前者回答“有多快”，
后者回答“为什么”。`ncu` 会 replay kernel，因此采集期间 benchmark 打印的延迟
不能进入性能表。

## 准备

需要 `nvcc`、`ncu` 和可选的 `ncu-ui`。在 WSL 中遇到
`ERR_NVGPUCTRPERM` 时，需要在 Windows NVIDIA 控制面板允许所有用户访问 GPU
性能计数器，然后执行 `wsl --shutdown`。若 `ncu-ui` 缺少 OpenGL 依赖，可安装：

```bash
sudo apt update
sudo apt install libopengl0 libgl1 libegl1 libglu1-mesa libxkbcommon-x11-0 libxcb-cursor0
```

## 统一采集入口

从项目根目录执行：

```bash
cd /home/fish/GEMM_For_Myself
./scripts/profile_stage.sh <stage> full
./scripts/profile_stage.sh <stage> instr
```

支持的 `<stage>`：

```text
naive coalesced smem 1d 2d vectorized padding warp double-buffering
```

脚本会先调用对应 Makefile target，然后对 `1024^3` 工作负载采集第一次正式
launch。报告输出到对应阶段的 `profiling/raw/`。每个阶段的确切文件名与重点指标
见其 `profiling/capture_commands.md`。

## 打开报告

例如：

```bash
ncu-ui 3.SMEM_kernel/profiling/raw/smem_full.ncu-rep
ncu-ui 3.SMEM_kernel/profiling/raw/smem_instr.ncu-rep
```

如果 WSL 图形界面不稳定，可从 Windows 侧打开 `.ncu-rep`，或在目标目录运行：

```bash
explorer.exe .
```

## 阅读顺序

1. `Summary` / `Speed Of Light`：判断计算与内存管线利用率。
2. `Occupancy`：结合 registers/thread 和 shared memory 判断资源约束。
3. `Memory Workload Analysis`：检查 global sector 利用率、L1/L2/DRAM 与 shared
   memory conflict。
4. `Scheduler Statistics` / `Warp State Statistics`：观察 eligible warp、issued warp
   和 stall。
5. `Instruction Statistics` / `Source`：核对向量指令与源码热点。

不要只凭单个百分比下结论；应把资源使用、访存、scheduler 和正式 benchmark
结果放在一起解释。

## 常见问题

- `No kernels were profiled`：先正常运行 binary，再检查 kernel regex；模板 kernel
  建议使用 `--kernel-name-base demangled`。
- `File ... already exists`：统一脚本已经传入 `-f` 覆盖同名生成物。
- binary 不存在：运行 `make <target>`；所有生成物统一位于 `build/`。
- profiler latency 很大：这是 replay 的预期现象，不代表正常运行性能回退。

正式数据口径见 [BENCHMARK_METHODOLOGY.md](./BENCHMARK_METHODOLOGY.md)。
