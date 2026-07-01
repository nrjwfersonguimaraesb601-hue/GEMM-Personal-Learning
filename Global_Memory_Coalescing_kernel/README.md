# Global Memory Coalescing Kernel

这一版是在 naive baseline 上做的第一步有效优化。

核心变化很简单：

- 把线程映射改成更利于 global memory 合并访问的方式
- 让 warp 内线程更自然地沿列方向展开
- 不改算法本质，只先把访问模式理顺

这版的意义很直接：它告诉我，哪怕只是先把 global memory access 调整对，收益也已经很大。

## 编译

```bash
nvcc -O3 My_Global_Memory_Coalescing_kernel.cu -o gmemc
nvcc -O3 My_Global_Memory_Coalescing_kernel_benchmarker.cu -o gmemc_bench
```

## 使用方式

```bash
./gmemc_bench
./gmemc_bench 1024 1024 1024
./gmemc_bench 4096 4096 4096 --no-check
```

## 这次 benchmark 设置

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Warmup: `10`
- Iterations: `50`
- Block: `32 x 32`
- CPU check: `disabled`

## 实测结果

| M | N | K | Block | Avg (ms) | Avg GFLOPS | Best GFLOPS |
|---|---|---|---|---:|---:|---:|
| 256 | 256 | 256 | 32x32 | 0.0544 | 616.2947 | 642.5098 |
| 512 | 512 | 512 | 32x32 | 0.3233 | 830.3416 | 856.6797 |
| 1024 | 1024 | 1024 | 32x32 | 2.4238 | 886.0027 | 892.0255 |
| 2048 | 2048 | 2048 | 32x32 | 16.7439 | 1026.0384 | 1031.0482 |
| 4096 | 4096 | 4096 | 32x32 | 157.8177 | 870.8718 | 891.6107 |
| 1023 | 1023 | 1023 | 32x32 | 2.2740 | 941.6104 | 1011.1287 |
| 4096 | 256 | 4096 | 32x32 | 8.4779 | 1013.2102 | 1021.0437 |
| 256 | 4096 | 4096 | 32x32 | 9.7675 | 879.4382 | 884.9676 |

## 相对 naive 的提升

| Case | Naive Avg GFLOPS | Coalesced Avg GFLOPS | Speedup |
|---|---:|---:|---:|
| 256 x 256 x 256 | 92.2776 | 616.2947 | 6.68x |
| 512 x 512 x 512 | 109.5825 | 830.3416 | 7.58x |
| 1024 x 1024 x 1024 | 123.4315 | 886.0027 | 7.18x |
| 2048 x 2048 x 2048 | 123.5758 | 1026.0384 | 8.30x |
| 4096 x 4096 x 4096 | 123.0080 | 870.8718 | 7.08x |

## 这组结果怎么看

- 这一步优化是非常值的，square case 直接从 `~123 GFLOPS` 提到 `~616-1026 GFLOPS`
- `2048` 这一档已经稳定站上 `1 TFLOPS`
- 说明当前项目里，第一波最大的收益确实来自把 global memory access 做得更合理

经验上看，这版最该记住的不是某一个单点数字，而是这条经验：

- “先把访存方向改对” 的收益，往往比很多后面更复杂的技巧还更直接

## 当前结论

这版已经是一个很扎实的中间 checkpoint：

- 比 naive 明显快很多
- 结构上仍然简单
- 很适合作为 SMEM 版之前的参照
