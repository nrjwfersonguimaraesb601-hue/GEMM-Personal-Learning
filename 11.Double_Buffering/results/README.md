# Double Buffering Results

`benchmark.csv` 保存用户提供的正式 CUDA Event 结果。完整 suite 使用 warmup 10、
iterations 50、`--no-check`，主值为 `4096³` Avg `8722.6411 GFLOPS`。

correctness 已在独立 CPU reference run 中得到 PASS；CSV 中正式性能行的 `SKIP`
只表示该次纯测速没有运行 CPU reference。
