# v1.0 Reorganization Report

Date: 2026-08-10; final audit and controlled rerun: 2026-08-30

## Scope and Safety

本次整理只改变工程结构、文档、统一构建入口和资产分类，没有修改任何 GEMM
kernel 的算法实现。开始整理时保留了已有 dirty worktree：`.gitignore` 与
`6.Vectorize_kernel/question.md` 的用户改动没有被回滚；尚未跟踪的 Stage 10、
Stage 11 和 cuBLAS 目录也全部在原工作区内继续整理。

没有执行 `git reset`、`git clean`、强制 checkout 或 push。历史 benchmark、CSV、
console output、日志、截图、`.ncu-rep` 和学习笔记均被保留。

## Problems Found Before Reorganization

- 构建产物散落在各 Stage 根目录，Stage 1–7 的 ELF 甚至已被 Git 跟踪。
- `.ncu-rep`、截图、日志、CSV 和 console 快照缺少统一分类。
- 根文档只讲到 Stage 8，未纳入 Warp Tiling、Double Buffering 和 cuBLAS。
- profiling 命令引用不同 binary 名和已过时路径，部分绝对 Markdown 链接失效。
- Stage 1–5 同时存在独立 kernel 文件和 benchmark 内嵌实现，权威关系不清。
- Stage 10/11/cuBLAS 是未跟踪工作，不能用清理命令覆盖或删除。
- correctness、纯测速和 profiler replay 的数据口径容易混淆。

## Final Structure

```text
GEMM_For_Myself/
├── 1.naive_kernel/
├── 2.Global_Memory_Coalescing_kernel/
├── 3.SMEM_kernel/
├── 4.1D_Blocktiling_kernel/
├── 5.2D_Blocktiling_kernel/
├── 6.Vectorize_kernel/
├── 7.Shared_Memory_Layout_Optimization/
├── 8.Autoing_kernel/
├── 10.Wraptiling_kernel/
├── 11.Double_Buffering/
├── cuBLAS_baseline/
├── docs/
├── results/
├── scripts/
├── build/                    # ignored local artifacts
├── Makefile
├── README.md
├── PERFORMANCE_SUMMARY.md
└── TODO.md
```

仓库历史中没有可确认的 Stage 9。本次没有虚构 Stage 9，也没有重编号 Stage 10/11。

## Authority Mapping

“性能权威入口”指生成当前 benchmark 结果的 translation unit；“kernel source”指
独立 kernel 文件是否直接被该 benchmark include。

| Stage | Performance entry | Kernel authority / status |
| ----: | ----------------- | ------------------------- |
| 1 | `My_naive_kernel_benchmark.cu` | benchmark 内嵌历史 kernel；独立 `My_naive_kernel.cu` 是学习/演示版本 |
| 2 | `My_Global_Memory_Coalescing_kernel_benchmarker.cu` | benchmark 内嵌历史 kernel；独立同名 kernel 文件不参与统一构建 |
| 3 | `My_SMEM_kernel_benchmark.cu` | benchmark 内嵌历史 kernel；`My_SMEM_kernel.cu` 保留作阶段源码记录 |
| 4 | `1D_Blocktiling_kernel_benchmark.cu` | benchmark 内嵌被测速实现；另外两个 `.cu` 是历史/演示变体 |
| 5 | `2D_Blocktiling_kernel_benchmark.cu` | benchmark 内嵌被测速实现；`main_...cu` 与 `2D_...cu` 保留为历史变体 |
| 6 | `Vectorize_kernel_benchmark.cu` | 直接 include `main_Vectorize_kerenl.cu`，后者是 kernel 权威源码 |
| 7 | `Shared_Memory_Layout_Padding_benchmark.cu` | 直接 include `Shared_Memory_Layout_Padding_kernel.cu`；XOR 文件是未并入正式结果的扩展实验 |
| 8 | `autotune_padding_benchmark.cu` | 直接 include Stage 8 的 templated `Shared_Memory_Layout_Padding_kernel.cu` |
| 10 | `Wraptiling_kernel_benchmark.cu` | 直接 include `main_Wraptiling_kernel.cu` |
| 11 | `Double_Buffering_benchmark.cu` | 直接 include `DoubleBuffering_main_kernel.cu` |
| cuBLAS | `cuBLAS_benchmark.cu` | wrapper 调用 cuBLAS；没有自定义 GEMM kernel |

Stage 1–5 没有在封版时强制去重，因为其 benchmark 内嵌代码与独立源码存在历史
差异。未经逐阶段等价性证明就替换 include 关系，可能改变正式性能和正确性行为。

## Duplicate Kernel Mapping

| Stage | Duplicate / alternate source | v1.0 handling |
| ----: | ---------------------------- | ------------- |
| 1 | benchmark `calculate_Matrix` vs `My_naive_kernel.cu` | benchmark 是测速权威；两份均保留 |
| 2 | benchmark `calculate_Matrix` vs standalone coalesced file | benchmark 是测速权威；两份均保留 |
| 3 | benchmark `calculate_Matrix` vs `My_SMEM_kernel.cu` | benchmark 是测速权威；两份均保留 |
| 4 | benchmark implementation、`1D_Blocktiling_kernel.cu`、`1D_Blocktiling_main_gpu_kernel.cu` | 不合并，标为历史变体 |
| 5 | benchmark implementation、`2D_Blocktiling_kernel.cu`、`main_2D_Blocktiling_kernel.cu` | 不合并，标为历史变体 |
| 6–8, 10–11 | benchmark 直接 include 单一阶段 kernel | 保持 include 关系，避免再次复制 |

后续若去重，应先对候选源码做逐行语义比较，再以相同编译参数运行 CPU reference
和短 benchmark；不能只因为函数名相似就替换。

## Move and Rename Mapping

### Root documentation

| Before | After |
| ------ | ----- |
| `GPU_DEVICE_INFO.md` | `docs/GPU_DEVICE_INFO.md` |
| `NSIGHT_COMPUTE_PROFILING_GUIDE.md` | `docs/NSIGHT_COMPUTE_PROFILING_GUIDE.md` |
| `GEMM_CONCEPTS_NOTES.md` | `docs/GEMM_CONCEPTS_NOTES.md` |

### Learning notes

| Before | After |
| ------ | ----- |
| `3.SMEM_kernel/teacher.md` | `3.SMEM_kernel/notes/teacher.md` |
| `5.2D_Blocktiling_kernel/qustion.md` | `5.2D_Blocktiling_kernel/notes/question.md` |
| `6.Vectorize_kernel/question.md` | `6.Vectorize_kernel/notes/question.md` |

历史文件名中的 `Autoing`、`Wraptiling`、`kerenl`、`benchmarker` 没有批量重命名，
以免破坏路径、脚本和 Git 历史；新文档使用规范技术名称。

### Nsight Compute reports

Stage 1–8 的根级 `.ncu-rep` 移入各自的 `profiling/raw/`。Stage 10 的
`Wraptiling_instr.ncu-rep` 与 Stage 11 的 `Double_Buffering_full.ncu-rep`、
`Double_Buffering_instr.ncu-rep` 同样完成分类。原始报告内容没有转换或裁剪。

完整 raw-report move mapping：

```text
1.naive_kernel/{naive_full,naive_instr}.ncu-rep
  -> 1.naive_kernel/profiling/raw/
2.Global_Memory_Coalescing_kernel/{gmemc_full,gmemc_instr}.ncu-rep
  -> 2.Global_Memory_Coalescing_kernel/profiling/raw/
3.SMEM_kernel/{smem_full,smem_instr}.ncu-rep
  -> 3.SMEM_kernel/profiling/raw/
4.1D_Blocktiling_kernel/{1D_Blocktiling_full,1D_Blocktiling_instr}.ncu-rep
  -> 4.1D_Blocktiling_kernel/profiling/raw/
5.2D_Blocktiling_kernel/{2D_Blocktiling_full,2D_Blocktiling_InstructionStats}.ncu-rep
  -> 5.2D_Blocktiling_kernel/profiling/raw/
6.Vectorize_kernel/{Vectorize_full,Vectorize_instr}.ncu-rep
  -> 6.Vectorize_kernel/profiling/raw/
7.Shared_Memory_Layout_Optimization/
  {Shared_Memory_Layout_Padding_full,Shared_Memory_Layout_Padding_instr}.ncu-rep
  -> 7.Shared_Memory_Layout_Optimization/profiling/raw/
8.Autoing_kernel/{ncu_C00_4096,ncu_C08_4096}.ncu-rep
  -> 8.Autoing_kernel/profiling/raw/
10.Wraptiling_kernel/Wraptiling_instr.ncu-rep
  -> 10.Wraptiling_kernel/profiling/raw/
11.Double_Buffering/{Double_Buffering_full,Double_Buffering_instr}.ncu-rep
  -> 11.Double_Buffering/profiling/raw/
```

此外，本次使用统一 profiling 脚本实际补采了 Stage 10 的
`profiling/raw/Wraptiling_full.ncu-rep`。Stage 10 仍没有历史截图，文档明确记录
缺口，没有伪造图片；Stage 11 的六张现有截图完整保留。

### Stage 8 results

| Before | After |
| ------ | ----- |
| `8.Autoing_kernel/autotune_quick.csv` | `8.Autoing_kernel/results/autotune_quick.csv` |
| `8.Autoing_kernel/autotune_full.csv` | `8.Autoing_kernel/results/autotune_full.csv` |
| root-level run logs | `8.Autoing_kernel/results/logs/` |
| `quick_output.md`, `full_output.md` | `8.Autoing_kernel/results/` |

console 快照中的旧绝对路径作为历史输出原样保留；当前 `run_autotune.sh` 已改为写入
新的 `results/` 和 `results/logs/`。

### Executables

原先散落在 Stage 1–8、10、11 与 cuBLAS 目录中的 ELF 均移入忽略目录
`build/legacy/`，没有删除。此前被 Git 跟踪的 Stage 1–7 binary 在工作树中表现为
删除，是因为 v1.0 停止版本控制生成物；其文件仍可在 `build/legacy/stage*/` 找到。

新的 Makefile 统一生成：

```text
build/naive_bench
build/coalesced_bench
build/smem_bench
build/register_1d_bench
build/register_2d_bench
build/vectorized_bench
build/padding_bench
build/autotuning_bench
build/warp_tiling_bench
build/double_buffering_bench
build/cublas_bench
```

## Deleted or No-longer-tracked Items

没有删除源码、文档、结果或 profiling 资产。唯一停止跟踪的是七个历史 ELF：

```text
1.naive_kernel/naive_bench
2.Global_Memory_Coalescing_kernel/gmemc_bench
3.SMEM_kernel/smem_bench
4.1D_Blocktiling_kernel/1D_Blocktiling_bench
5.2D_Blocktiling_kernel/2D_Blocktiling_bench
6.Vectorize_kernel/Vectorize_bench
7.Shared_Memory_Layout_Optimization/Shared_Memory_Layout_Padding_bench
```

它们分别保留在 `build/legacy/stage1/` 到 `stage7/`。Stage 8、10、11 和 cuBLAS
原有未跟踪 ELF 也保留在对应 `build/legacy/` 子目录。新的 `build/`、`*.o`、
`*_bench`、`__pycache__/` 和 `*.pyc` 由根 `.gitignore` 排除。

## Historical Assets Preserved

- Stage 1–8 的原始 Full / InstructionStats `.ncu-rep`；
- Stage 10/11 整理前已有的 raw reports；
- Stage 1–8 与 Stage 11 的 Nsight Compute 截图和解释；
- Stage 8 quick/full CSV、compile/autotune logs 和 console output；
- Stage 3、5、6 的学习笔记，包括用户在 Stage 6 笔记中的现有改动；
- Stage 1–5 的重复/变体 kernel 源码；
- Stage 7 的 XOR swizzle 扩展实验；
- 所有旧 ELF，迁入 ignored `build/legacy/`。

## New Infrastructure

- `Makefile`：统一 `sm_89` 默认构建，支持 `CUDA_ARCH` 覆盖。
- `scripts/build_all.sh`：从任意目录调用根 Makefile。
- `scripts/smoke_test.sh`：Stage 1–8、10、11 与 cuBLAS 的 256³ CPU correctness。
- `scripts/benchmark_stage.sh`：正式 CUDA Event benchmark 入口，默认关闭 CPU
  reference；因此输出应为 `SKIP`，不能写成 `PASS`。
- `scripts/profile_stage.sh`：统一 Full / InstructionStats 报告采集。
- `scripts/run_comparison.sh` / `run_comparison.py`：先验证 correctness，再顺序运行
  统一 4096³ FP32 benchmark 并写入 CSV。
- `scripts/plot_results.py`：由同轮 comparison CSV 生成 SVG/PNG；
  `plot_performance.py` 保留为兼容入口。
- `scripts/check_markdown_links.py`：检查仓库内 Markdown 本地链接。
- `docs/BENCHMARK_METHODOLOGY.md`：分离 correctness、benchmark、profiling 与精度。
- `results/final_4096.csv`：Stage 1–8、10、11、cuBLAS FP32 与独立 TF32 参考。
- `results/rtx4060_laptop/comparison_4096.csv`：2026-08-30 的同轮 FP32 主比较。

## Performance Record Decisions

- Stage 7 主记录继续使用完整 suite 的 8581.5961 Avg GFLOPS；独立 8984.1773
  复测不覆盖它。
- Stage 8 C08 的 9664.98 Avg GFLOPS 是完整 autotune suite 记录，不表述为所有
  shape 或设备的通用最优。
- Stage 10 使用最终 suite 的 6670.5097 Avg GFLOPS，明确记录为负优化。
- Stage 11 使用最终 suite 的 8722.6411 Avg GFLOPS；它是软件 double buffering，
  不是 `cp.async`。
- cuBLAS FP32 主记录是 9429.4001 Avg GFLOPS；多轮约 9.3–9.4 TFLOPS，与最佳
  自定义记录处于同一波动区间，不声明任何一方稳定获胜。
- cuBLAS TF32 14.844754 Avg TFLOPS 是不同精度/执行路径的 Tensor Core 参考，
  不进入 FP32 排名。

## Validation Performed

在当前 RTX 4060 Laptop GPU、Compute Capability 8.9 环境中完成：

1. 强制 `make -B -j2 all`：全部 11 个 benchmark target 从源码编译成功，cuBLAS
   链接成功。审计中修正了 Stage 11 对不存在 `main_kernel.cu` 的依赖/include，
   与实际权威文件 `DoubleBuffering_main_kernel.cu` 对齐；kernel 算法未改。
2. Shell `bash -n`：统一脚本与 Stage 8 autotune 脚本通过。
3. Python `py_compile`：plot 与 Markdown link checker 通过。
4. Stage 1–7、10、11、cuBLAS FP32：256³ CPU reference 均实际显示 `PASS`。
5. Stage 8：14 个候选在 256³ correctness verification 中全部显示 `PASS`；
   smoke CSV 写入忽略的 `build/stage8_smoke.csv`，没有覆盖正式结果。
6. `benchmark_stage.sh double-buffering 256 256 256`：统一 pure-speed 入口正常，
   因传入 `--no-check` 如实显示 `SKIP`。
7. `profile_stage.sh warp full`：成功生成可打开的 Stage 10 Full report。
8. `run_comparison.sh`：完成 warmup 10、iterations 50 的 4096³ 同轮 FP32 比较；
   C08 为 8.914 Avg TFLOPS，cuBLAS FP32 为 8.709 Avg TFLOPS，不据此声明稳定胜出。
9. `plot_results.py`：由同轮 CSV 成功生成 `docs/assets/performance_4096.png` 与 SVG。
10. Stage 6–8、10、11 的 benchmark include 目标逐一检查存在。
11. `git diff --check`：通过。
12. `check_markdown_links.py`：全部本地 Markdown 链接通过。

这些 smoke run 不替换历史正式 benchmark。Nsight Compute 报告中的 replay latency
也没有写入正式性能表。

## Remaining Limitations

- Stage 10 没有分类截图，只有 raw Full / InstructionStats 报告和人工摘要。
- Stage 1–5 尚有 benchmark 内嵌 kernel 副本，需要未来逐阶段验证后再去重。
- 多数高性能 kernel 只支持满足 tile 整除和向量对齐的尺寸，没有通用 tail path。
- 已完成一次统一顺序复测，但没有锁定频率、功耗和温度，也没有报告跨轮方差或
  置信区间；历史数据仍来自不同轮次。
- Stage 10 的高寄存器压力与 shared-store conflict、Stage 11 的 shared-load conflict
  仍是后续实验重点。

建议下一版先做两件事：在固定 GPU 状态下统一复测 FP32 路线；为 Stage 1–5 的
重复 kernel 建立逐阶段等价性测试。`cp.async`、XOR swizzle 和 tail handling 应
作为新的独立实验提交，避免改写 v1.0 的历史基线。

## Git Handoff

本次没有创建 commit 或 push。建议审阅顺序：

1. `README.md`、`PERFORMANCE_SUMMARY.md`、`TODO.md`
2. `Makefile` 与 `scripts/`
3. 本报告的 authority/move mapping
4. Stage 10、Stage 11、cuBLAS README 与结果 CSV
5. `git status --short` 和 `git diff --stat`

确认后再由仓库所有者决定如何拆分 commit。
