# Global Memory Coalescing Kernel

这一版是在 `naive_kernel` 的基础上做的第一步访存优化，改动不大，重点只有一个：

- 把线程映射改回 `threadIdx.x -> col`、`threadIdx.y -> row`
- 让 warp 内线程沿列方向展开
- 这样 `B[k * N + col]` 的读取和 `C[row * N + col]` 的写回都更接近连续访问

所以这份 README 不展开讲实现细节，主要记录性能结果。

## 测试环境

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute capability: `8.9`
- Global memory: `8.00 GiB`
- SM count: `24`
- Max threads per block: `1024`
- Warmup: `10`
- Iterations: `100`
- Block: `16 x 16`
- CPU check: `enabled`, `max check dim = 2048`

## 实测结果

| M | N | K | Block | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 256 | 256 | 256 | 16x16 | PASS | 0.0440 | 0.0758 | 0.2468 | 442.9100 | 762.0465 | 0.0000 | |
| 512 | 512 | 512 | 16x16 | PASS | 0.3082 | 0.3183 | 0.3400 | 843.4147 | 870.9103 | 0.0000 | |
| 1024 | 1024 | 1024 | 16x16 | PASS | 2.1011 | 2.4286 | 2.8713 | 884.2309 | 1022.0972 | 0.0001 | |
| 2048 | 2048 | 2048 | 16x16 | PASS | 16.6113 | 16.8336 | 19.3791 | 1020.5712 | 1034.2261 | 0.0002 | |
| 4096 | 4096 | 4096 | 16x16 | SKIP | 138.0146 | 140.9154 | 161.6220 | 975.3294 | 995.8287 | 0.0000 | CPU check skipped |
| 1023 | 1023 | 1023 | 16x16 | PASS | 2.0839 | 2.1741 | 2.6276 | 984.8579 | 1027.5095 | 0.0001 | |
| 4096 | 256 | 4096 | 16x16 | SKIP | 8.3240 | 9.0087 | 9.8364 | 953.5179 | 1031.9440 | 0.0000 | CPU check skipped |
| 256 | 4096 | 4096 | 16x16 | SKIP | 8.6700 | 9.3156 | 10.1560 | 922.1050 | 990.7671 | 0.0000 | CPU check skipped |

## 相对 naive 的提升

对比 [`naive_kernel`](../naive_kernel/README.md) 当前那版 non-coalesced baseline：

| Case | Naive Avg GFLOPS | Coalesced Avg GFLOPS | 提升 |
|---|---:|---:|---:|
| 256 x 256 x 256 | 183.3508 | 442.9100 | 2.42x |
| 512 x 512 x 512 | 192.2429 | 843.4147 | 4.39x |
| 1024 x 1024 x 1024 | 230.0084 | 884.2309 | 3.84x |
| 2048 x 2048 x 2048 | 241.4446 | 1020.5712 | 4.23x |
| 4096 x 4096 x 4096 | 238.6743 | 975.3294 | 4.09x |
| 1023 x 1023 x 1023 | 661.0822 | 984.8579 | 1.49x |

## 结论

- 对 square case，这一步优化很值，提升大约在 `2.4x ~ 4.4x`
- `1024` 到 `4096` 这一段已经基本站上 `~900-1020 GFLOPS`
- 这说明只是把 global memory access 调整到更 coalesced，就已经能拿到很明显的收益

也要记一笔真实情况：

- `4096 x 256 x 4096` 和 `256 x 4096 x 4096` 这两组 shape 并没有比当前 naive baseline 更快
- 所以这一步优化的收益主要体现在常见 square case 上，而不是所有 shape 都统一提升

这一版可以看成后续 shared memory / tiling 之前的一个中间 checkpoint。
