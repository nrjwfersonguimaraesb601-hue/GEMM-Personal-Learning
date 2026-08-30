# Warp Tiling Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make warp

./scripts/profile_stage.sh warp full
./scripts/profile_stage.sh warp instr

ncu-ui 10.Wraptiling_kernel/profiling/raw/Wraptiling_full.ncu-rep
ncu-ui 10.Wraptiling_kernel/profiling/raw/Wraptiling_instr.ncu-rep
```

脚本使用 `1024³`、warmup 1、`--launch-skip 1`、`--launch-count 1`，只采集
第一个正式 launch。`ncu` replay latency 不用于正式性能比较。
