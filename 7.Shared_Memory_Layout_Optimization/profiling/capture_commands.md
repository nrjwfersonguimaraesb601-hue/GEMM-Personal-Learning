# Shared-memory Padding GEMM Capture Commands

当前 kernel 参数为 `BM=64`、`BN=64`、`BK=8`、`TM=8`、`TN=8`；输入需满足
`M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`。

```bash
cd /home/fish/GEMM_For_Myself
make padding
./scripts/profile_stage.sh padding full
./scripts/profile_stage.sh padding instr
```

报告写入：

```text
7.Shared_Memory_Layout_Optimization/profiling/raw/Shared_Memory_Layout_Padding_full.ncu-rep
7.Shared_Memory_Layout_Optimization/profiling/raw/Shared_Memory_Layout_Padding_instr.ncu-rep
```

```bash
ncu-ui 7.Shared_Memory_Layout_Optimization/profiling/raw/Shared_Memory_Layout_Padding_full.ncu-rep
ncu-ui 7.Shared_Memory_Layout_Optimization/profiling/raw/Shared_Memory_Layout_Padding_instr.ncu-rep
```

重点对比 padding 前后的 shared-memory conflict、wavefront、occupancy 和指令开销。
Profiler replay 下的 latency 不用于正式 benchmark。
