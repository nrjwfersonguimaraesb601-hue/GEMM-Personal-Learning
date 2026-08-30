# 2D Block Tiling Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make 2d
./scripts/profile_stage.sh 2d full
./scripts/profile_stage.sh 2d instr
```

报告写入：

```text
5.2D_Blocktiling_kernel/profiling/raw/2D_Blocktiling_full.ncu-rep
5.2D_Blocktiling_kernel/profiling/raw/2D_Blocktiling_InstructionStats.ncu-rep
```

```bash
ncu-ui 5.2D_Blocktiling_kernel/profiling/raw/2D_Blocktiling_full.ncu-rep
ncu-ui 5.2D_Blocktiling_kernel/profiling/raw/2D_Blocktiling_InstructionStats.ncu-rep
```

重点观察二维 thread tile 带来的 register reuse、occupancy 和 scheduler 变化。
Profiler latency 不计入正式 benchmark。
