# Performance Summary

这份文件把当前四个版本放在一起，方便横向对比。

## Benchmark 口径

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Warmup: `10`
- Iterations: `50`
- CPU check: `disabled`
- 表格为 pure kernel time

## 总览

| Version | Main Change | Representative Square Range | Best Square Avg GFLOPS |
|---|---|---|---:|
| `naive_kernel` | non-coalesced baseline | `~92-123 GFLOPS` | 123.5758 |
| `Global_Memory_Coalescing_kernel` | coalesced global-memory access | `~616-1026 GFLOPS` | 1026.0384 |
| `SMEM_kernel` | shared-memory tile reuse | `~641-1039 GFLOPS` | 1039.4091 |
| `1D_Blocktiling_kernel` | register blocking + 1D block tiling | `~1259-3698 GFLOPS` | 3698.2388 |

## Square Case Comparison

```text
Square Case Avg GFLOPS

Case             Naive      Coalesced      SMEM      1D Blocktiling
256 x 256 x 256   92.2776     616.2947   640.7902        1258.8250
512 x 512 x 512  109.5825     830.3416   836.6070        2676.5946
1024 x 1024 x 1024
                 123.4315     886.0027   904.6960        3397.1641
2048 x 2048 x 2048
                 123.5758    1026.0384  1039.4091        3698.2388
4096 x 4096 x 4096
                 123.0080     870.8718   958.3544        3659.7348
```

## Speedup Over Naive

```text
Speedup Over Naive

Case             Coalesced/Naive   SMEM/Naive   1D Blocktiling/Naive
256 x 256 x 256        6.68x          6.94x              13.64x
512 x 512 x 512        7.58x          7.63x              24.42x
1024 x 1024 x 1024
                       7.18x          7.33x              27.52x
2048 x 2048 x 2048
                       8.30x          8.41x              29.93x
4096 x 4096 x 4096
                       7.08x          7.79x              29.75x
```

## SMEM vs Coalesced

```text
SMEM vs Coalesced

Case             Coalesced GFLOPS   SMEM GFLOPS   SMEM/Coalesced
256 x 256 x 256       616.2947       640.7902          1.04x
512 x 512 x 512       830.3416       836.6070          1.01x
1024 x 1024 x 1024
                      886.0027       904.6960          1.02x
2048 x 2048 x 2048
                     1026.0384      1039.4091          1.01x
4096 x 4096 x 4096
                      870.8718       958.3544          1.10x
```

## 1D Blocktiling vs SMEM

```text
1D Blocktiling vs SMEM

Case             SMEM GFLOPS   1D Blocktiling GFLOPS   1D/SMEM
256 x 256 x 256    640.7902          1258.8250          1.96x
512 x 512 x 512    836.6070          2676.5946          3.20x
1024 x 1024 x 1024
                   904.6960          3397.1641          3.75x
2048 x 2048 x 2048
                  1039.4091          3698.2388          3.56x
4096 x 4096 x 4096
                   958.3544          3659.7348          3.82x
```

## 简短结论

- `naive` 现在可以稳定看成 `~123 GFLOPS` baseline
- `coalesced` 是第一波最明显的收益来源，直接把 square case 推到 `~0.6-1.0 TFLOPS`
- `SMEM` 在这次新数据里已经整体略强于 `coalesced`
- `1D blocktiling` 是当前项目里最强的一版，已经把主力 square case 推到 `3 TFLOPS+`

## Source

- [naive_kernel/README.md](./naive_kernel/README.md)
- [Global_Memory_Coalescing_kernel/README.md](./Global_Memory_Coalescing_kernel/README.md)
- [SMEM_kernel/README.md](./SMEM_kernel/README.md)
- [1D_Blocktiling_kernel/README.md](./1D_Blocktiling_kernel/README.md)
