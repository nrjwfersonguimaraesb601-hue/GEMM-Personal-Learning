# Stage 8 Capture Commands

Stage 8 一个可执行文件包含多个模板实例，`--kernel-name` 必须收窄到目标配置。

```bash
cd /home/fish/GEMM_For_Myself
make autotuning

ncu -f --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemm_shared_memory_layout_padding.*' \
  --launch-count 1 \
  -o 8.Autoing_kernel/profiling/raw/ncu_stage8 \
  ./build/autotuning_bench --suite quick --warmup 0 --iters 1 \
  --no-verify --csv /tmp/stage8_ncu.csv
```

复现 C00/C08 时先用 `ncu --list-kernels` 或 `ncu-ui` 确认 demangled 模板参数，
避免宽正则采集错误实例。Nsight replay latency 不参与 autotuning 排名。
