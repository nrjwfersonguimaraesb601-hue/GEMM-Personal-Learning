# Vectorized GEMM Capture Commands

当前 kernel 参数为 `BM=64`、`BN=64`、`BK=8`、`TM=8`、`TN=8`；输入需满足
`M % 64 == 0`、`N % 64 == 0`、`K % 8 == 0`。

```bash
cd /home/fish/GEMM_For_Myself
make vectorized
./scripts/profile_stage.sh vectorized full
./scripts/profile_stage.sh vectorized instr
```

报告写入：

```text
6.Vectorize_kernel/profiling/raw/Vectorize_full.ncu-rep
6.Vectorize_kernel/profiling/raw/Vectorize_instr.ncu-rep
```

```bash
ncu-ui 6.Vectorize_kernel/profiling/raw/Vectorize_full.ncu-rep
ncu-ui 6.Vectorize_kernel/profiling/raw/Vectorize_instr.ncu-rep
```

重点确认 vectorized global load/store、shared-memory bank conflict、registers/thread、
occupancy 与 scheduler stall。`ncu` 会 replay kernel，因此其间程序打印的 latency
不能用于正式性能排名。
