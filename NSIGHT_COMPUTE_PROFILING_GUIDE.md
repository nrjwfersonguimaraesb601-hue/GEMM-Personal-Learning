# Nsight Compute Profiling Guide

这是一份精简版速查手册，记录在 WSL 下分析 `My_SMEM_kernel_benchmark.cu` 的最小流程。

## 1. 用到的工具

- `nvcc`：编译 `.cu`
- `ncu`：生成性能分析报告
- `ncu-ui`：打开可视化报告

## 2. WSL 分析前准备

如果 `ncu` 报：

```text
ERR_NVGPUCTRPERM
```

去 Windows 侧把 `管理 GPU 性能计数器` 改成 `所有用户`，然后执行：

```bash
wsl --shutdown
```

如果 `ncu-ui` 报 `libOpenGL.so.0`，装依赖：

```bash
sudo apt update
sudo apt install libopengl0 libgl1 libegl1 libglu1-mesa libxkbcommon-x11-0 libxcb-cursor0
```

## 3. 编译

```bash
cd /home/fish/GEMM_For_Myself/SMEM_kernel
nvcc My_SMEM_kernel_benchmark.cu -O3 -lineinfo -o smem_bench
```

## 4. 先正常跑一下

```bash
./smem_bench 1024 1024 1024 --warmup 10 --iters 50 --bx 32 --by 32 --no-check
```

## 5. 生成指令报告

这份报告主要看 `Instruction Statistics` 和 `Executed Instruction Mix`。

```bash
ncu -f \
  --section InstructionStats \
  --kernel-name-base demangled \
  --kernel-name regex:.*calculate_Matrix.* \
  --launch-skip 1 \
  --launch-count 1 \
  -o smem_instr \
  ./smem_bench 1024 1024 1024 --warmup 1 --iters 3 --bx 32 --by 32 --no-check
```

生成文件：

```text
/home/fish/GEMM_For_Myself/SMEM_kernel/smem_instr.ncu-rep
```

## 6. 生成完整报告

这份报告主要看：

- `Occupancy`
- `Warp State Statistics`
- `Memory Workload Analysis`
- `Scheduler Statistics`
- `Speed Of Light`

```bash
ncu -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name regex:.*calculate_Matrix.* \
  --launch-skip 1 \
  --launch-count 1 \
  -o smem_full \
  ./smem_bench 1024 1024 1024 --warmup 1 --iters 3 --bx 32 --by 32 --no-check
```

生成文件：

```text
/home/fish/GEMM_For_Myself/SMEM_kernel/smem_full.ncu-rep
```

## 7. 打开可视化界面

```bash
ncu-ui /home/fish/GEMM_For_Myself/SMEM_kernel/smem_instr.ncu-rep
```

或者：

```bash
ncu-ui /home/fish/GEMM_For_Myself/SMEM_kernel/smem_full.ncu-rep
```

如果 WSL 图形界面不稳定，就在 Windows 侧直接打开 `.ncu-rep` 文件。  
想从 WSL 打开当前目录：

```bash
explorer.exe .
```

## 8. 在界面里怎么看

1. 打开报告后先在 `Summary` 里双击 `calculate_Matrix`
2. 进入 `Details`
3. 指令报告重点看 `Instruction Statistics`
4. 完整报告重点看：
   - `Occupancy`
   - `Warp State Statistics`
   - `Memory Workload Analysis`
   - `Scheduler Statistics`
   - `Speed Of Light`

## 9. 常见问题

`No kernels were profiled`

- 通常是过滤条件太严
- 这份文档里的命令已经是今天验证过的可用版本

`File xxx.ncu-rep already exists`

- 加 `-f`

`./smem_bench does not exist`

- 重新编译：

```bash
nvcc -O3 -lineinfo My_SMEM_kernel_benchmark.cu -o smem_bench
```

## 10. 一句话总结

- `smem_instr.ncu-rep`：看指令构成
- `smem_full.ncu-rep`：看完整性能瓶颈
