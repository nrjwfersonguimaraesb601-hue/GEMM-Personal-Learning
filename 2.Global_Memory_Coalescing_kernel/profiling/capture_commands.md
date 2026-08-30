# Coalesced Kernel Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make coalesced
./scripts/profile_stage.sh coalesced full
./scripts/profile_stage.sh coalesced instr
```

报告写入：

```text
2.Global_Memory_Coalescing_kernel/profiling/raw/gmemc_full.ncu-rep
2.Global_Memory_Coalescing_kernel/profiling/raw/gmemc_instr.ncu-rep
```

```bash
ncu-ui 2.Global_Memory_Coalescing_kernel/profiling/raw/gmemc_full.ncu-rep
ncu-ui 2.Global_Memory_Coalescing_kernel/profiling/raw/gmemc_instr.ncu-rep
```

重点比较 global-memory coalescing、sector 利用率、指令数和 scheduler 状态。
Nsight Compute replay 下的 latency 不用于正式性能排名。
