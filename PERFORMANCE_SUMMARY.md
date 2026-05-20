# Performance Summary

这份文件把各个版本的 benchmark 结果放到一个地方，方便横向对比。

## 总览

| Version | Main Change | Best Square Avg GFLOPS | Representative Square Range | Notes |
|---|---|---:|---|---|
| `naive_kernel` | non-coalesced baseline | 241.4446 | `~183-241 GFLOPS` | 作为最原始对照组 |
| `Global_Memory_Coalescing_kernel` | coalesced global-memory access | 1020.5712 | `~843-1021 GFLOPS` | square case 提升最明显 |

## Square Case Comparison

| Case | Naive Avg GFLOPS | Coalesced Avg GFLOPS | Speedup |
|---|---:|---:|---:|
| 256 x 256 x 256 | 183.3508 | 442.9100 | 2.42x |
| 512 x 512 x 512 | 192.2429 | 843.4147 | 4.39x |
| 1024 x 1024 x 1024 | 230.0084 | 884.2309 | 3.84x |
| 2048 x 2048 x 2048 | 241.4446 | 1020.5712 | 4.23x |
| 4096 x 4096 x 4096 | 238.6743 | 975.3294 | 4.09x |

## Quick Notes

- 常见 square case 下，coalescing 版本相对 naive 提升大约 `2.4x ~ 4.4x`
- `1024` 到 `4096` 这一段，coalescing 版本基本稳定在 `~900-1020 GFLOPS`
- 两组窄矩形 case 里，coalescing 版本反而更慢，所以不能默认“更 coalesced 就对所有 shape 都更快”

## Test Setup

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute capability: `8.9`
- Global memory: `8.00 GiB`
- SM count: `24`
- Max threads per block: `1024`
- Warmup: `10`
- Iterations: `100`
- Block: `16 x 16`
- CPU check: `enabled`, `max check dim = 2048`

## Detailed Data

### naive_kernel

| M | N | K | Avg ms | Avg GFLOPS | Best GFLOPS | Check | Note |
|---|---|---|---:|---:|---:|---|---|
| 256 | 256 | 256 | 0.1830 | 183.3508 | 191.6257 | PASS | |
| 512 | 512 | 512 | 1.3963 | 192.2429 | 236.5920 | PASS | |
| 1024 | 1024 | 1024 | 9.3365 | 230.0084 | 242.8665 | PASS | |
| 2048 | 2048 | 2048 | 71.1545 | 241.4446 | 243.8088 | PASS | |
| 4096 | 4096 | 4096 | 575.8431 | 238.6743 | 245.3101 | SKIP | CPU check skipped |
| 1023 | 1023 | 1023 | 3.2389 | 661.0822 | 693.7671 | PASS | boundary case |
| 4096 | 256 | 4096 | 2.5852 | 3322.7325 | 3750.0415 | SKIP | CPU check skipped |
| 256 | 4096 | 4096 | 2.5929 | 3312.8442 | 3462.0750 | SKIP | CPU check skipped |

### Global_Memory_Coalescing_kernel

| M | N | K | Avg ms | Avg GFLOPS | Best GFLOPS | Check | Note |
|---|---|---|---:|---:|---:|---|---|
| 256 | 256 | 256 | 0.0758 | 442.9100 | 762.0465 | PASS | |
| 512 | 512 | 512 | 0.3183 | 843.4147 | 870.9103 | PASS | |
| 1024 | 1024 | 1024 | 2.4286 | 884.2309 | 1022.0972 | PASS | |
| 2048 | 2048 | 2048 | 16.8336 | 1020.5712 | 1034.2261 | PASS | |
| 4096 | 4096 | 4096 | 140.9154 | 975.3294 | 995.8287 | SKIP | CPU check skipped |
| 1023 | 1023 | 1023 | 2.1741 | 984.8579 | 1027.5095 | PASS | |
| 4096 | 256 | 4096 | 9.0087 | 953.5179 | 1031.9440 | SKIP | CPU check skipped |
| 256 | 4096 | 4096 | 9.3156 | 922.1050 | 990.7671 | SKIP | CPU check skipped |

## Source

- [naive_kernel/README.md](./naive_kernel/README.md)
- [Global_Memory_Coalescing_kernel/README.md](./Global_Memory_Coalescing_kernel/README.md)
