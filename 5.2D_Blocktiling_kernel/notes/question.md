第一,整体映射,这一点是必须要理解的。完成后应该能回答 BM、BN、BK、TM、TN分别代表什么,一个 Block负责计算 C 的哪一块,一个 Thread 负责计算多少个 C 元素,以及 Grid、Block、Thread 是如何映射到整个矩阵 C 的。
第二部分是 Kernel 初始化,阅读这一部分代码的时候,想一想,为什么要算 blockRow 和 blockCol,A、B、C 指针一开始为什么要偏移,当前 block 对应的是矩阵中的哪个位置。
第三部分是 Shared Memory 加载,重点思考一个 thread 搬运几个 A 元素和 B 元素,innerRowA、innerColA 是怎么算的, strideA、strideB 存在的意义是什么,以及为什么这样可以让所有线程把整个 tile 搬完,还有就是为什么这里需要 \_\_syncthreads()
第四部分是 BK 循环,每一次 BK 循环能描述当前处理的是 K维的哪一段,Shared Memory 中的数据什么时候更新,为什么 BK 循环结束后要加载下一块。
第五部分是寄存器缓存,这里要重点理解 regM 保存什么,regN 保存什么,为什么先度到寄存器而不是一直访问 Shared Memory,每次 BK 内层循环寄存器什么时候更新。
第六部分是计算过程,要能解释 threadResults 为什么是二维,一个 Thread 为什么能算 TM 乘以 TN 个结果,一次 FMA 计算的数据来自哪里,以及相比一维 Block Tiling,多复用了什么。
第七部分是写回 Global Memory,要能看懂 ThreadResults 如何对应到 C,每个线程写哪些元素,写回地址是怎么计算的,一个 Warp 写回是否连续,是否 coalesced。
第八部分是性能分析,结合 Nsight,思考这一版主要优化了什么,与 Shared Memory 相比快在哪里,与一维 Block Tiling 相比为什么更快, Register 使用增加了多少,Shared Memory 使用增加了吗?当前瓶颈是什么。
最后一部分是最终自测,不看代码,试着回答,从 Global Memory 到 Shared Memory 的数据流是什么,从 Shared Memory 到 Register 的数据流是什么,Register 中保存了哪些数据,一个 Thread 的完整工作流程,一个 Block 的完整工作流程,以及为什么二维 Block Tiling 能提升性能,以及目前只提升 20% 可能的原因有哪些。

一个块必须最好能被n个TM*TN覆盖（完全覆盖），BM 最好能被 TM 整除，BN 最好能被 TN 整除；
同一块上的线程blockRow和blockCol是一样的，所以作初始移动时才能有效
interRowA和interColA只是为了搬运A到As，为了对应A和As，不决定计算C的哪一部分
由于TM*TN，相对于1D分块，这样的线程数变少了，所以一个线程在搬运数据到As和Bs时也需要搬运更多
