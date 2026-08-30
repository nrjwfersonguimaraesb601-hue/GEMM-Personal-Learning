# Warp Tiling Results

`benchmark.csv` 保存用户提供的正式 CUDA Event 记录。正式 suite 使用 warmup 10、
iterations 50、`--no-check`；表中的 `SKIP` 不是 correctness PASS。该 kernel 已在
独立 CPU reference run 中得到 PASS。

主汇总使用完整 suite 的 `4096³` Avg `6670.5097 GFLOPS`。独立 run 只作为波动
记录，不覆盖主值。
