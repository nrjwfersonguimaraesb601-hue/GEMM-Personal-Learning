# Naive Kernel Capture Commands

从项目根目录统一构建并采集报告：

```bash
cd /home/fish/GEMM_For_Myself
make naive
./scripts/profile_stage.sh naive full
./scripts/profile_stage.sh naive instr
```

报告写入：

```text
1.naive_kernel/profiling/raw/naive_full.ncu-rep
1.naive_kernel/profiling/raw/naive_instr.ncu-rep
```

```bash
ncu-ui 1.naive_kernel/profiling/raw/naive_full.ncu-rep
ncu-ui 1.naive_kernel/profiling/raw/naive_instr.ncu-rep
```

重点查看 Speed Of Light、Instruction Statistics、Memory Workload Analysis、
Scheduler Statistics 和 Source。Nsight Compute 会 replay kernel，报告采集期间
程序打印的 latency 不作为正式 benchmark 数据。
