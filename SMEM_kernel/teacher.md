# Teacher Snippet Notes

下面这段代码是 shared-memory GEMM 的核心骨架。它的思路是：

- 一个 block 负责输出矩阵 `C` 的一个 `BLOCKSIZE x BLOCKSIZE` 子块
- 一个 thread 负责这个子块中的一个输出元素
- 先把 `A` 和 `B` 的当前 tile 读进 shared memory
- 再在 shared memory 上做局部乘加
- 沿着 `K` 维不断推进，直到把完整的 dot-product 累加完

原始代码如下：

```cpp
// advance pointers to the starting positions
//   "把A,B,C的指针位置都移动到当前位置"    //
A += cRow * BLOCKSIZE * K;                     // row=cRow, col=0
B += cCol * BLOCKSIZE;                         // row=0, col=cCol
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

//   "初始化temp = 0"    //
float tmp = 0.0;
// the outer loop advances A along the columns and B along
// the rows until we have fully calculated the result in C.

// 循环开始
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
{
    // Have each thread load one of the elements in A & B from
    // global memory into shared memory.
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
    {
        tmp += As[threadRow * BLOCKSIZE + dotIdx] *
            Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
}
C[threadRow * N + threadCol] =
    alpha * tmp + beta * C[threadRow * N + threadCol];
```

## 先建立整体坐标感

在看逐行解释之前，先把这几个量的物理意义想清楚：

- `A` 的形状是 `M x K`
- `B` 的形状是 `K x N`
- `C` 的形状是 `M x N`
- `cRow` 表示当前 block 负责 `C` 的第几个 tile 行
- `cCol` 表示当前 block 负责 `C` 的第几个 tile 列
- `threadRow` 和 `threadCol` 表示 thread 在 block 内部的二维坐标

如果 `BLOCKSIZE = 32`，那么：

- 一个 block 负责 `C` 里的一个 `32 x 32` 子块
- block 内一共有 `32 x 32 = 1024` 个 threads
- 每个 thread 最终负责这个 `32 x 32` 子块中的一个输出元素

## 逐行解释

### 1. `A += cRow * BLOCKSIZE * K;`

这行是在移动 `A` 指针，让它指向当前 block 需要处理的 `A` 子块起点。

为什么是 `cRow * BLOCKSIZE * K`：

- `cRow * BLOCKSIZE` 表示当前 block 对应的是 `A` 的第几组 tile 行
- `A` 是 row-major 存储
- `A` 每往下跳一整行，要跨过 `K` 个元素

所以：

- 先往下跳 `cRow * BLOCKSIZE` 行
- 每一行长度是 `K`
- 总偏移量就是 `cRow * BLOCKSIZE * K`

这时 `A` 指向的是：

- 当前 block 对应的 `A` tile 的左上角
- 也就是 `row = cRow * BLOCKSIZE, col = 0`

### 2. `B += cCol * BLOCKSIZE;`

这行是在移动 `B` 指针，让它指向当前 block 需要处理的 `B` 子块起点。

为什么这里只是 `cCol * BLOCKSIZE`：

- `B` 的 tile 在水平方向上按列移动
- 当前 block 负责的是 `C` 的第 `cCol` 个 tile 列
- 因此 `B` 需要先向右偏移 `cCol * BLOCKSIZE` 列

这时 `B` 指向的是：

- 当前 block 所对应 `B` tile 的左上角
- 也就是 `row = 0, col = cCol * BLOCKSIZE`

### 3. `C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;`

这行把 `C` 指针移动到当前 block 负责写回的输出子块起点。

为什么是这个偏移：

- 先往下跳 `cRow * BLOCKSIZE` 行
- 每一行长度是 `N`
- 所以下移部分是 `cRow * BLOCKSIZE * N`
- 然后再向右跳 `cCol * BLOCKSIZE` 列

合起来就是：

- `cRow * BLOCKSIZE * N + cCol * BLOCKSIZE`

这时 `C` 指向的是：

- 当前输出 tile 的左上角
- 也就是 `row = cRow * BLOCKSIZE, col = cCol * BLOCKSIZE`

### 4. `float tmp = 0.0;`

这行定义了当前 thread 的局部累加器。

含义是：

- 每个 thread 最终要算一个输出元素
- 这个输出元素本质上是 `A` 某一行和 `B` 某一列的点积
- 点积要跨很多个 `K` 元素分批累加

所以需要一个寄存器变量：

- 一开始置零
- 每处理完一个 tile，就把这一 tile 的部分和加进去
- 最后再一次性写回到 `C`

### 5. `for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)`

这是最关键的外层循环，它表示沿着 `K` 维一段一段推进。

为什么要这样做：

- 直接把整个 `A` 行和整个 `B` 列一次性放进 shared memory 是不现实的
- shared memory 容量有限
- 所以要把完整的 dot-product 切成多个小块

每次循环处理的是：

- `A` 当前 tile 的一小段列
- `B` 当前 tile 的一小段行

如果 `BLOCKSIZE = 32`，那每次就处理 `K` 维上的 `32` 个元素。

### 6. `As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];`

这行表示当前 thread 从 global memory 读一个 `A` 元素，放到 shared memory 的 `As` 中。

左边：

- `As[...]` 是 shared memory 中保存 `A` tile 的位置
- `threadRow`、`threadCol` 决定当前 thread 在 tile 中负责哪个元素

右边：

- `A[threadRow * K + threadCol]` 表示从当前 `A` tile 的相对坐标 `(threadRow, threadCol)` 读取一个元素

这里为什么是 `threadRow * K + threadCol`：

- 当前 `A` 指针已经移动到 tile 左上角
- row-major 下，同一行长度仍然是整张矩阵的 `K`

所以每个 thread：

- 负责把 `A` tile 里的一个元素搬到 shared memory
- 整个 block 一起协作，就能把一个完整 tile 装满

### 7. `Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];`

这行和上一行完全对称，只不过对象换成了 `B`。

左边：

- `Bs[...]` 是 shared memory 中保存 `B` tile 的位置

右边：

- `B[threadRow * N + threadCol]` 表示从当前 `B` tile 的相对坐标 `(threadRow, threadCol)` 读取一个元素

这里为什么步长是 `N`：

- `B` 是 `K x N`
- row-major 下，一整行长度是 `N`

这一步完成后：

- 所有 threads 合作把 `B` 的一个 tile 也搬到了 shared memory

### 8. 为什么这里强调 `threadCol` 连续

代码注释里特别强调：

- 让 `threadCol` 作为连续下标
- 这样能让 global memory access 更 coalesced

原因是：

- 一个 warp 内通常 `threadIdx.x` 连续变化
- 如果 `threadCol` 对应横向连续地址
- 那么 warp 内线程在读 `A` 或 `B` 的时候，就更容易形成合并访存

这正是前一版 coalescing 优化和这一版 shared-memory 优化能够叠加的基础。

### 9. `__syncthreads();`

这是第一次同步，意义非常重要。

为什么必须同步：

- 虽然每个 thread 只负责搬一个元素
- 但后面做乘加时，每个 thread 都要读完整的一整行 `As` 和一整列 `Bs`
- 如果某些 thread 还没搬完，另一些 thread 就已经开始算
- 那就会读到未初始化或旧数据

所以第一次 `__syncthreads()` 的作用是：

- 等整块 tile 全部装入 shared memory
- 再允许任何 thread 开始计算

### 10. `A += BLOCKSIZE;`

这行表示：

- 当前这次 tile 已经装入 shared memory 了
- 接下来准备让 `A` 指针移动到下一块 `K` 方向 tile

为什么只是 `+ BLOCKSIZE`：

- 对 `A` 来说，tile 推进方向是列方向
- 同一行内向右移动 `BLOCKSIZE` 个元素即可

因此下一轮循环时：

- `A` 就会指向当前 block 对应的下一段 `A` tile

### 11. `B += BLOCKSIZE * N;`

这行表示：

- `B` 也要推进到下一个 tile
- 但对 `B` 来说，推进方向是行方向

为什么是 `BLOCKSIZE * N`：

- 往下跳 `BLOCKSIZE` 行
- `B` 每行有 `N` 个元素

所以总偏移量是：

- `BLOCKSIZE * N`

这意味着下一轮循环时：

- `B` 会指向当前 block 需要的下一段 `B` tile

### 12. `for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)`

这是内层乘加循环。

含义是：

- 当前 shared memory 中已经缓存好了 `A` 的一个 tile 和 `B` 的一个 tile
- 现在要把这个 tile 对当前输出元素的贡献算出来

如果把当前 thread 对应的输出元素记作：

- `C_local(threadRow, threadCol)`

那么它需要做的是：

- 取 `As` 的第 `threadRow` 行
- 取 `Bs` 的第 `threadCol` 列
- 做长度为 `BLOCKSIZE` 的点积

### 13. `tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];`

这行就是 tile 内部的乘加核心。

可以把它拆成两部分：

- `As[threadRow * BLOCKSIZE + dotIdx]`
  - 取 `A` tile 中当前 thread 对应的那一行
  - 第 `dotIdx` 个元素

- `Bs[dotIdx * BLOCKSIZE + threadCol]`
  - 取 `B` tile 中当前 thread 对应的那一列
  - 第 `dotIdx` 个元素

这两个元素相乘后：

- 就是当前 tile 在 `K` 维某一个位置上的乘积项

把 `dotIdx = 0 ... BLOCKSIZE-1` 都加起来：

- 就得到了当前 tile 对输出元素的局部贡献

然后再把这个局部贡献加到 `tmp` 里。

注意这里最重要的收益是：

- `As` 和 `Bs` 都在 shared memory
- 同一个 tile 会被 block 内很多线程反复读取
- 这比每次乘加都回到 global memory 重新取数便宜得多

### 14. 第二个 `__syncthreads();`

这次同步的作用和第一次不同。

第一次同步是为了：

- 确保 tile 已经装满，才能开始算

第二次同步是为了：

- 确保所有 thread 都已经把当前 tile 用完
- 然后才能安全地让某些更快的线程去覆盖 shared memory，装下一轮 tile

如果没有这次同步：

- 快的 thread 可能提前进入下一轮循环
- 把 `As`、`Bs` 中的数据改写掉
- 慢的 thread 还在读上一轮 tile
- 最终就会出现 data race，结果错误

所以这个同步本质上是在保护 shared memory 这块“公共缓存”。

### 15. 外层循环结束意味着什么

当 `bkIdx` 从 `0` 一直推进到 `K` 末尾时，表示：

- 当前输出元素需要的所有 `K` 段贡献都已经累加到 `tmp` 里了

也就是说，这时候：

- `tmp` 已经是完整的 dot-product 结果

### 16. `C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];`

这是最终写回。

它不是简单的：

- `C = tmp`

而是更通用的 GEMM 形式：

- `C = alpha * (A x B) + beta * C`

如果：

- `alpha = 1`
- `beta = 0`

那它就退化成最普通的 GEMM：

- `C = A x B`

为什么这里写成这种形式：

- 因为很多高性能 GEMM / BLAS 接口本来就是这个通用表达式
- 这样内核结构更接近真实 GEMM 库的接口风格

## 把整段代码串起来看

如果把这段代码压缩成一句“它到底在干什么”，可以这样理解：

1. block 先定位自己负责的 `C` 子块
2. 计算这个 `C` 子块所需的 `A` tile 和 `B` tile 起点
3. 每轮沿 `K` 维取一个 `BLOCKSIZE` 宽度的小块
4. block 内所有 threads 协作把 `A`、`B` 当前 tile 搬到 shared memory
5. 同步
6. 每个 thread 在 shared memory 上完成这一 tile 的局部点积
7. 同步
8. 推进到下一 tile
9. 所有 tile 累加完后，把最终结果写回 `C`

## 这段代码为什么重要

shared-memory GEMM 的第一步，不是马上把性能做到很高，而是先建立这个核心心智模型：

- global memory 负责把大矩阵“分批运进来”
- shared memory 负责在 block 内做“近距离复用”
- 寄存器变量 `tmp` 负责保存 thread 私有的累加结果

一旦这个模型理解透了，后面的很多优化就会自然得多：

- 更复杂的 block tiling
- register blocking
- vectorized load/store
- warp tiling

本质上都还是在围绕这三层数据流继续深挖：

- global memory -> shared memory -> register -> compute -> writeback
