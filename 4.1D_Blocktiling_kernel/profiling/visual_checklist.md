# 1D Blocktiling Visual Checklist

这一版最值得放的图，重点不是“图很多”，而是要把新的瓶颈形态讲清楚。

## 指令报告值得放的图

- `Instruction Statistics` 整页合并图
  这一版最值得看的就是 `FFMA`、`LDS`、`STS`、`BAR` 的组合关系。

## 完整报告值得放的图

- `Summary`
  用来交代 block size、寄存器数、throughput 和优化提示。
- `Speed Of Light`
  看 compute / memory 是否已经一起进入高利用区间。
- `Memory Chart`
  看 shared-memory traffic、L1/TEX hit rate、L2 hit rate。
- `Scheduler Statistics`
  看 eligible warp 和发射状态。
- `Warp State`
  看 stall 是不是已经转成 `MIO Throttle` / `Long Scoreboard` 主导。

## 这一版特别值得讲的观察

- `FFMA` 已经是第一大项，说明算术强度上来了。
- `LDS` 很重，shared-memory 复用已经真正参与主路径。
- `Registers/thread = 56`，occupancy 已经开始受寄存器限制。
- `MIO Throttle` 主导 stall，说明后面该重点看 shared-memory / pipeline 压力，而不只是 global-memory access。
