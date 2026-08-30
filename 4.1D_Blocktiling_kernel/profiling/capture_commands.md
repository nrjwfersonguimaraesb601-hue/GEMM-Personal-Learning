# 1D Block Tiling Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make 1d
./scripts/profile_stage.sh 1d full
./scripts/profile_stage.sh 1d instr
```

报告写入：

```text
4.1D_Blocktiling_kernel/profiling/raw/1D_Blocktiling_full.ncu-rep
4.1D_Blocktiling_kernel/profiling/raw/1D_Blocktiling_instr.ncu-rep
```

```bash
ncu-ui 4.1D_Blocktiling_kernel/profiling/raw/1D_Blocktiling_full.ncu-rep
ncu-ui 4.1D_Blocktiling_kernel/profiling/raw/1D_Blocktiling_instr.ncu-rep
```

重点观察 register reuse、occupancy、scheduler 和指令吞吐。Nsight Compute 的
kernel replay 会放大程序内计时，不能用该 latency 与正常 benchmark 比较。
