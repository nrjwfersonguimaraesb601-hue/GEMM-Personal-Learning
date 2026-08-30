# GEMM Concepts Index

- `M/N/K`：`A[M,K] × B[K,N] = C[M,N]` 的问题尺寸。
- `BM/BN/BK`：一个 thread block 在 M、N、K 方向处理的 tile 尺寸。
- `TM/TN`：一个 thread 在寄存器中负责的 C micro tile。
- `WM/WN`：一个 warp 负责的 C tile。
- Coalescing：让一个 warp 的线程访问尽量连续的 global-memory 地址。
- Shared-memory reuse：协作加载 A/B tile，并在 block 内重复使用。
- Register reuse：把一个 thread 的多个累加结果保存在寄存器中。
- Arithmetic intensity：单位数据移动对应的计算量。
- Bank conflict：warp 线程访问同一 shared-memory bank 的不同地址造成串行化。
- Vectorized access：使用 `float4` 等宽访问减少指令并提高 transaction 利用率。
- Occupancy：一个 SM 可同时驻留的 warps/blocks 比例；受寄存器和 shared memory
  等资源限制。
- Long Scoreboard：warp 等待较长延迟数据依赖的 stall。
- Barrier：warp 等待 block 级同步的 stall。
- MIO Throttle：memory input/output pipeline 压力导致的 stall。
- Double buffering：两个 shared-memory stage 交替读写，尝试隐藏下一 tile 的加载
  延迟。本项目 Stage 11 使用寄存器预取，不是 `cp.async`。

每个概念的实际效果应结合对应 Stage README、benchmark 和 Nsight Compute 报告
理解；采用某项技术不保证性能单调提升。
