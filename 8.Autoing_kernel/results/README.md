# Stage 8 Autotuning Results

- `autotune_quick.csv` / `autotune_full.csv`：原始候选结果。
- `quick_output.md` / `full_output.md`：历史 console 输出快照，内部旧路径按原样保留。
- `logs/compile_*.log`：ptxas 编译资源记录。
- `logs/autotune_*.log`：quick/full suite console 日志。

`run_autotune.sh` 的新运行会继续写入本目录。历史输出中的旧绝对路径只是当时的
console 记录，不代表当前文件位置。

正式综合候选是 C08：`BM=128, BN=64, BK=16, TM=8, TN=8`。完整 suite 的
`4096³` Avg/Best 为 `9664.98/9777.64 GFLOPS`。C08 是七个 case 的几何平均
候选，不是每个 shape 的单点最优。
