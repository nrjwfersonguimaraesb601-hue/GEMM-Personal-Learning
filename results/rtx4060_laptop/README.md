# RTX 4060 Laptop GPU Results

`comparison_4096.csv` 是 2026-08-30 由 `scripts/run_comparison.sh` 生成的同轮
FP32 主比较：4096³、warmup 10、iterations 50、CUDA Event timing。

CSV 中的 `PASS@256` 表示脚本先对同一实现执行并通过 256³ CPU reference；4096³
纯性能循环没有运行 CPU reference。所有阶段按脚本顺序运行，GPU clock、power 和
temperature 未锁定，因此几个百分点的差距不能解释为稳定排名。

重跑：

```bash
./scripts/run_comparison.sh
python3 scripts/plot_results.py
```
