# Shared Memory Kernel

这一版是在 `Global_Memory_Coalescing_kernel` 的基础上继续往前走的一步：把 `A` 和 `B` 的一个 tile 先搬进 shared memory，再在 block 内复用。

这一步的核心目标不是改线程映射，而是解决前一版里仍然存在的一个问题：

- 即使 global memory 访问已经更 coalesced 了
- 同一个 block 内的很多线程还是会重复从 global memory 读取相同的 `A` 和 `B` 元素
- 这些重复读取会继续消耗带宽，也会限制后续吞吐提升

所以这一版做的事情可以概括成一句话：

- 用 `32 x 32` 的 tile 把 `A` 和 `B` 的子块缓存到 shared memory
- 让同一个 block 里的线程重复使用这份 tile，而不是每次乘加都直接回 global memory 取数

## 文件说明

- `My_SMEM_kernel.cu`: correctness-first 的 shared-memory GEMM 版本
- `My_SMEM_kernel_benchmark.cu`: benchmark 版本，支持命令行参数和性能统计
- `teacher.md`: 对参考代码片段的逐行解释

## 这版 kernel 做了什么

和前一版 coalesced kernel 相比，这版新增了下面几件事：

- 在 kernel 内声明 `As` 和 `Bs` 两块 shared memory
- 每个 thread 先各自搬运一个 `A` 元素和一个 `B` 元素进 shared memory
- 用 `__syncthreads()` 保证整块 tile 都装载完成
- 在 shared memory 上做当前 tile 的局部 dot-product
- 循环推进到下一个 `K` 方向 tile，直到累加完成

同时，这版 benchmark / correctness 版本还补了边界保护：

- `row < M`、`col < N` 的输出边界判断
- `bkIdx + tx < K`、`bkIdx + ty < K` 的尾 tile 读取保护

所以它不只适用于 `4096 x 4096 x 4096` 这类 `32` 的整倍数方阵，也能安全处理 `1023 x 1023 x 1023` 这种非整倍数 case。

## 编译

```bash
nvcc -O3 My_SMEM_kernel.cu -o My_SMEM_kernel
nvcc -O3 My_SMEM_kernel_benchmark.cu -o My_SMEM_kernel_benchmark
```

## 使用方式

运行默认 benchmark suite：

```bash
./My_SMEM_kernel_benchmark
```

运行单个指定 case：

```bash
./My_SMEM_kernel_benchmark 4096 4096 4096
```

关闭 CPU check，专注测 GPU kernel 时间：

```bash
./My_SMEM_kernel_benchmark 4096 4096 4096 --no-check
```

放宽 CPU check 阈值，强制对 `4096` 做 CPU 参考校验：

```bash
./My_SMEM_kernel_benchmark 4096 4096 4096 --max-check-dim 4096
```

## 测试环境

下面这组 benchmark 数据来自你这次实际跑出的结果：

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Compute capability: `8.9`
- Global memory: `8.00 GiB`
- SM count: `24`
- Max threads per block: `1024`
- Warmup: `10`
- Iterations: `100`
- Block: `32 x 32`
- CPU check: `enabled`, `max check dim = 2048`

## 实测结果

| M | N | K | Block | Check | Min (ms) | Avg (ms) | Max (ms) | Avg GFLOPS | Best GFLOPS | Max Error | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 256 | 256 | 256 | 32x32 | PASS | 0.0451 | 0.0516 | 0.0737 | 650.0541 | 744.7273 | 0.0000 | |
| 512 | 512 | 512 | 32x32 | PASS | 0.3082 | 0.3170 | 0.3318 | 846.8679 | 870.9103 | 0.0000 | |
| 1024 | 1024 | 1024 | 32x32 | PASS | 2.0490 | 2.5527 | 3.0730 | 841.2554 | 1048.0519 | 0.0001 | |
| 2048 | 2048 | 2048 | 32x32 | PASS | 16.3397 | 16.4871 | 18.8476 | 1042.0211 | 1051.4160 | 0.0002 | |
| 4096 | 4096 | 4096 | 32x32 | SKIP | 140.5974 | 144.8331 | 163.3137 | 948.9473 | 977.5359 | 0.0000 | CPU check skipped |
| 1023 | 1023 | 1023 | 32x32 | PASS | 2.0582 | 2.1747 | 2.7574 | 984.6031 | 1040.3055 | 0.0001 | |
| 4096 | 256 | 4096 | 32x32 | SKIP | 8.9231 | 9.7429 | 11.2876 | 881.6572 | 962.6621 | 0.0000 | CPU check skipped |
| 256 | 4096 | 4096 | 32x32 | SKIP | 8.8606 | 9.5496 | 11.7463 | 899.5054 | 969.4556 | 0.0000 | CPU check skipped |

## 和前两版的对比

### 相对 naive baseline

| Case | Naive Avg GFLOPS | SMEM Avg GFLOPS | 提升 |
|---|---:|---:|---:|
| 256 x 256 x 256 | 183.3508 | 650.0541 | 3.55x |
| 512 x 512 x 512 | 192.2429 | 846.8679 | 4.41x |
| 1024 x 1024 x 1024 | 230.0084 | 841.2554 | 3.66x |
| 2048 x 2048 x 2048 | 241.4446 | 1042.0211 | 4.32x |
| 4096 x 4096 x 4096 | 238.6743 | 948.9473 | 3.98x |
| 1023 x 1023 x 1023 | 661.0822 | 984.6031 | 1.49x |

### 相对 coalesced kernel

| Case | Coalesced Avg GFLOPS | SMEM Avg GFLOPS | 相对变化 |
|---|---:|---:|---:|
| 256 x 256 x 256 | 442.9100 | 650.0541 | 1.47x |
| 512 x 512 x 512 | 843.4147 | 846.8679 | 1.00x |
| 1024 x 1024 x 1024 | 884.2309 | 841.2554 | 0.95x |
| 2048 x 2048 x 2048 | 1020.5712 | 1042.0211 | 1.02x |
| 4096 x 4096 x 4096 | 975.3294 | 948.9473 | 0.97x |
| 1023 x 1023 x 1023 | 984.8579 | 984.6031 | 1.00x |

## 怎么理解这组结果

### 1. 正确性没有问题

`256`、`512`、`1024`、`2048`、`1023` 这些 case 都通过了 CPU check，`max error` 维持在 `1e-4` 左右。这说明当前 shared-memory 版本在这些测试点上是正确的。

`4096` 和两组 rectangular case 显示 `SKIP`，不是 GPU 没跑，也不是结果错误，而是 benchmark 逻辑主动跳过了 CPU 参考实现。原因只是 `4096^3` 这种 case 用三重循环 CPU 去全量对拍会非常慢。

### 2. 相对 naive，有明显提升

这一点是明确的。即使不看 coalesced 版，只看最原始的 non-coalesced baseline，这版 `SMEM` 已经把常见 square case 的吞吐从 `~183-241 GFLOPS` 提升到了 `~650-1042 GFLOPS`。

也就是说：

- shared memory 确实在起作用
- block 内对 tile 的复用，确实减少了很多重复 global memory 读取
- 这一步已经把 kernel 从“纯 baseline”推进到了“接近 1 TFLOPS”的量级

### 3. 但它没有稳定压过 coalesced 版

这次最值得认真记录的现象，不是“SMEM 很快”，而是：

- 它比 naive 快很多
- 但和前一版 `Global_Memory_Coalescing_kernel` 相比，并没有出现一边倒的全面领先

从数据上看：

- `256` 上提升很明显
- `512` 基本持平
- `1024` 反而略慢
- `2048` 略快
- `4096` 又略慢
- `1023` 基本打平

换句话说，这版 shared-memory kernel 在你这台 `RTX 4060 Laptop GPU` 上，更像是“和 coalesced 版打成同一档”，而不是“明显打开一个新台阶”。

### 4. 为什么会这样

从实现结构上，至少有几个很合理的解释：

- 当前 block 直接用了 `32 x 32 = 1024` threads，已经顶到每个 block 的线程上限
- shared memory load/store、两次 `__syncthreads()` 也带来了额外开销
- 每个 thread 仍然只算一个输出值，没有做 register blocking
- 还没有做 vectorized load/store，也没有做更深的 tile 级优化

所以现在这版更准确的定位是：

- 已经把 shared memory 这一步走通了
- correctness、benchmark、文档都补齐了
- 但还不是一个“shared memory 一加上去就稳定更快”的成熟高性能版本

### 5. 这版的价值在哪

它的价值主要有三层：

- 你已经把 shared-memory tiling 的基本写法亲手走通了
- 你已经能用 benchmark 证明“shared memory 不是魔法，它也有实现代价”
- 它为后续的 `register blocking`、`block tiling`、`vectorized access` 提供了一个非常自然的起点

也就是说，这一版很像一个典型的学习型 checkpoint：

- 不是终点
- 但它把下一步该往哪里推，已经暴露得很清楚了

## 当前结论

- 这版 `SMEM kernel` 在 RTX 4060 Laptop GPU 上已经能把 `4096 x 4096 x 4096` 跑到大约 `0.95 TFLOPS`
- 它相对 naive baseline 提升很明显
- 但相对当前 coalesced 版并没有形成稳定领先
- 这说明 shared memory 这一步已经走通，但后续还需要继续引入更深的 tiling / register reuse / vectorized access，才能把收益真正放大

## 建议的下一步

1. 系统比较 block size，例如 `16x16`、`32x8`、`32x16`
2. 尝试让一个 thread 计算多个输出值，减少同步和 shared-memory 开销摊销问题
3. 引入更明确的 block tiling / register blocking
4. 后续再补 `cuBLAS` baseline，看看和库实现还有多大差距
