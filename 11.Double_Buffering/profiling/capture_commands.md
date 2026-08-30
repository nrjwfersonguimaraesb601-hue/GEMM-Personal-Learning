# Double Buffering Capture Commands

```bash
cd /home/fish/GEMM_For_Myself
make double-buffering

./scripts/profile_stage.sh double-buffering full
./scripts/profile_stage.sh double-buffering instr

ncu-ui 11.Double_Buffering/profiling/raw/Double_Buffering_full.ncu-rep
ncu-ui 11.Double_Buffering/profiling/raw/Double_Buffering_instr.ncu-rep
```

采集使用 `1024³`、warmup 1，并跳过 warmup launch。Nsight replay latency 不用于
正式 CUDA Event 性能表。
