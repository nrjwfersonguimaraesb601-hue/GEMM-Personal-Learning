# Shared-memory Kernel Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make smem
./scripts/profile_stage.sh smem full
./scripts/profile_stage.sh smem instr
```

报告写入：

```text
3.SMEM_kernel/profiling/raw/smem_full.ncu-rep
3.SMEM_kernel/profiling/raw/smem_instr.ncu-rep
```

```bash
ncu-ui 3.SMEM_kernel/profiling/raw/smem_full.ncu-rep
ncu-ui 3.SMEM_kernel/profiling/raw/smem_instr.ncu-rep
```

重点查看 shared-memory reuse、occupancy、Memory Workload Analysis、Scheduler
Statistics 和指令构成。Profiler latency 只作采集上下文，不是 benchmark 数据。
