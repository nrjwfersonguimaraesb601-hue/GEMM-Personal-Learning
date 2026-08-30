#include <cuda_runtime.h>

// Most of this kernel is based on the previous
// vectorized + shared-memory-padding kernel.
//
// Double buffering:
//   current tile : SMEM[readStage] -> compute
//   next tile    : GMEM -> thread registers -> SMEM[writeStage]
//
// The two SMEM stages alternate between 0 and 1.

template <
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN>
__global__ void sgemm_double_buffering(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K)
{
    // --------------------------------------------------------
    // Compile-time checks.
    // --------------------------------------------------------
    static_assert(BK % 4 == 0,
                  "BK must be divisible by 4");

    static_assert(BN % 8 == 0,
                  "BN must be divisible by 8");

    static_assert(TN % 4 == 0,
                  "TN must be divisible by 4");

    static_assert(
        (BM * BN) % (TM * TN) == 0,
        "BM * BN must be divisible by TM * TN");

    // --------------------------------------------------------
    // Shared-memory padding.
    // Same layout as the previous padding kernel.
    // --------------------------------------------------------
    constexpr int BS_PADDING = 4;
    constexpr int BS_HALF = BN / 2;
    constexpr int BS_STRIDE = BN + BS_PADDING;

    constexpr int AS_PADDING = 4;
    constexpr int AS_STRIDE = BM + AS_PADDING;

    // --------------------------------------------------------
    // Block coordinates.
    //
    // One block still computes one BM x BN tile of C.
    // --------------------------------------------------------
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    constexpr int threadsNum =
        (BM * BN) / (TM * TN);

    const int tid = threadIdx.x;

    // --------------------------------------------------------
    // Thread -> C mapping.
    //
    // Nothing changes here because of double buffering.
    //
    // Each thread still computes one TM x TN tile of C.
    // --------------------------------------------------------
    const int threadRow =
        tid / (BN / TN);

    const int threadCol =
        tid % (BN / TN);

    // --------------------------------------------------------
    // Double-buffered shared memory.
    //
    // Previous:
    //
    //   As[BK][BM]
    //   Bs[BK][BN]
    //
    // Now:
    //
    //   As[0] / Bs[0]
    //   As[1] / Bs[1]
    //
    // One stage is used for computation while the other
    // stage is prepared for the next BK tile.
    // --------------------------------------------------------
    __shared__ float As[2][BK * AS_STRIDE];
    __shared__ float Bs[2][BK * BS_STRIDE];

    // --------------------------------------------------------
    // Move A/B/C base pointers to this block's matrix tile.
    //
    // Unlike the old kernel, A and B will NOT be advanced
    // inside the BK loop.
    //
    // bkIdx / nextBk will explicitly select the K tile.
    // --------------------------------------------------------
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N +
         blockCol * BN;

    // ========================================================
    // GMEM loading coordinates for A.
    //
    // A tile:
    //
    //      BM rows
    //      BK columns
    //
    // Every vector load contains 4 floats.
    // ========================================================
    const int interRowA =
        tid / (BK / 4);

    const int interColA =
        tid % (BK / 4);

    constexpr int strideA =
        threadsNum / (BK / 4);

    static_assert(
        threadsNum % (BK / 4) == 0,
        "A loading mapping requires exact division");

    static_assert(
        BM % strideA == 0,
        "Each thread must load an equal number of A float4 vectors");

    constexpr int A_LOADS_PER_THREAD =
        BM / strideA;

    // ========================================================
    // GMEM loading coordinates for B.
    //
    // B tile:
    //
    //      BK rows
    //      BN columns
    //
    // Every vector load contains 4 floats.
    // ========================================================
    const int interRowB =
        tid / (BN / 4);

    const int interColB =
        tid % (BN / 4);

    constexpr int strideB =
        threadsNum / (BN / 4);

    static_assert(
        threadsNum % (BN / 4) == 0,
        "B loading mapping requires exact division");

    static_assert(
        BK % strideB == 0,
        "Each thread must load an equal number of B float4 vectors");

    constexpr int B_LOADS_PER_THREAD =
        BK / strideB;

    // --------------------------------------------------------
    // Result registers.
    //
    // Each thread still owns TM x TN C values.
    // --------------------------------------------------------
    float threadRes[TM * TN] = {0.0f};

    // Current dot-product operands.
    float regM[TM];
    float regN[TN];

    // --------------------------------------------------------
    // Prefetch registers.
    //
    // These registers hold the NEXT BK tile.
    //
    // They are different from regM/regN:
    //
    // nextA / nextB:
    //     GMEM -> registers -> next SMEM stage
    //
    // regM / regN:
    //     current SMEM stage -> registers -> FMA
    // --------------------------------------------------------
    float4 nextA[A_LOADS_PER_THREAD];
    float4 nextB[B_LOADS_PER_THREAD];

    // ========================================================
    //
    // PROLOGUE
    //
    // Load the first BK tile into SMEM stage 0.
    //
    // Before starting the pipeline, we must already have
    // one tile available for computation.
    //
    // ========================================================

    // --------------------------------------------------------
    // Initial A tile:
    //
    // A[:, 0:BK]
    //
    // GMEM -> As[0]
    //
    // A is still transposed during loading:
    //
    // A[row][k]
    //       ->
    // As[0][k][row]
    // --------------------------------------------------------
#pragma unroll
    for (int loadOffset = 0;
         loadOffset < BM;
         loadOffset += strideA)
    {
        const int rowA =
            interRowA + loadOffset;

        const int colA =
            interColA * 4;

        const float4 tmp =
            reinterpret_cast<const float4 *>(
                &A[rowA * K +
                   colA])[0];

        As[0][(colA + 0) * AS_STRIDE +
              rowA] = tmp.x;

        As[0][(colA + 1) * AS_STRIDE +
              rowA] = tmp.y;

        As[0][(colA + 2) * AS_STRIDE +
              rowA] = tmp.z;

        As[0][(colA + 3) * AS_STRIDE +
              rowA] = tmp.w;
    }

    // --------------------------------------------------------
    // Initial B tile:
    //
    // B[0:BK, :]
    //
    // GMEM -> Bs[0]
    //
    // Padding mapping is unchanged.
    // --------------------------------------------------------
#pragma unroll
    for (int loadOffset = 0;
         loadOffset < BK;
         loadOffset += strideB)
    {
        const int rowB =
            interRowB + loadOffset;

        const int logicalColB =
            interColB * 4;

        const float4 tmp =
            reinterpret_cast<const float4 *>(
                &B[rowB * N +
                   logicalColB])[0];

        const int physicalColB =
            logicalColB +
            (logicalColB >= BS_HALF
                 ? BS_PADDING
                 : 0);

        reinterpret_cast<float4 *>(
            &Bs[0][rowB * BS_STRIDE +
                   physicalColB])[0] = tmp;
    }

    // Everybody must finish loading tile 0
    // before any thread starts computing it.
    __syncthreads();

    // --------------------------------------------------------
    // Stage 0 contains tile 0.
    // --------------------------------------------------------
    int readStage = 0;

    // ========================================================
    //
    // MAIN PIPELINE
    //
    // ========================================================
    for (int bkIdx = 0;
         bkIdx < K;
         bkIdx += BK)
    {
        // ----------------------------------------------------
        // readStage:
        //
        // Shared-memory tile currently used for computation.
        //
        // writeStage:
        //
        // Shared-memory tile that will contain the next tile.
        //
        // 0 -> 1
        // 1 -> 0
        // ----------------------------------------------------
        const int writeStage =
            readStage ^ 1;

        const bool hasNext =
            (bkIdx + BK < K);

        // ====================================================
        //
        // STEP 1:
        //
        // Prefetch NEXT tile:
        //
        // GMEM -> thread registers
        //
        // This happens BEFORE computing the current tile.
        //
        // ====================================================
        if (hasNext)
        {
            const int nextBk =
                bkIdx + BK;

            // -----------------------------------------------
            // Prefetch next A tile.
            //
            // Current:
            //
            //   A[:, bkIdx : bkIdx+BK]
            //
            // Next:
            //
            //   A[:, nextBk : nextBk+BK]
            //
            // -----------------------------------------------
            int slotA = 0;

#pragma unroll
            for (int loadOffset = 0;
                 loadOffset < BM;
                 loadOffset += strideA)
            {
                const int rowA =
                    interRowA + loadOffset;

                const int colA =
                    interColA * 4;

                nextA[slotA] =
                    reinterpret_cast<const float4 *>(
                        &A[rowA * K +
                           nextBk +
                           colA])[0];

                ++slotA;
            }

            // -----------------------------------------------
            // Prefetch next B tile.
            //
            // Next:
            //
            //   B[nextBk : nextBk+BK, :]
            //
            // -----------------------------------------------
            int slotB = 0;

#pragma unroll
            for (int loadOffset = 0;
                 loadOffset < BK;
                 loadOffset += strideB)
            {
                const int rowB =
                    interRowB + loadOffset;

                const int logicalColB =
                    interColB * 4;

                nextB[slotB] =
                    reinterpret_cast<const float4 *>(
                        &B[(nextBk + rowB) * N +
                           logicalColB])[0];

                ++slotB;
            }
        }

        // ====================================================
        //
        // STEP 2:
        //
        // Compute CURRENT tile:
        //
        // As[readStage]
        // Bs[readStage]
        //
        //      ->
        //
        // regM / regN
        //
        //      ->
        //
        // threadRes
        //
        // ====================================================

#pragma unroll
        for (int dotIdx = 0;
             dotIdx < BK;
             ++dotIdx)
        {
            // -----------------------------------------------
            // Current A values:
            //
            // As[readStage][dotIdx][thread rows]
            //
            // Each thread reads TM A values.
            // -----------------------------------------------
#pragma unroll
            for (int i = 0;
                 i < TM;
                 ++i)
            {
                regM[i] =
                    As[readStage][dotIdx * AS_STRIDE +
                                  threadRow * TM +
                                  i];
            }

            // -----------------------------------------------
            // Current B values:
            //
            // Bs[readStage][dotIdx][thread cols]
            //
            // Each thread reads TN B values.
            // -----------------------------------------------
#pragma unroll
            for (int j = 0;
                 j < TN;
                 ++j)
            {
                const int logicalColB =
                    threadCol * TN + j;

                const int physicalColB =
                    logicalColB +
                    (logicalColB >= BS_HALF
                         ? BS_PADDING
                         : 0);

                regN[j] =
                    Bs[readStage][dotIdx * BS_STRIDE +
                                  physicalColB];
            }

            // -----------------------------------------------
            // TM x TN outer product.
            //
            // Exactly the same calculation as before.
            // -----------------------------------------------
#pragma unroll
            for (int i = 0;
                 i < TM;
                 ++i)
            {
#pragma unroll
                for (int j = 0;
                     j < TN;
                     ++j)
                {
                    threadRes[i * TN + j] +=
                        regM[i] *
                        regN[j];
                }
            }
        }

        // ====================================================
        //
        // STEP 3:
        //
        // The next tile was prefetched earlier into:
        //
        // nextA[]
        // nextB[]
        //
        // Now store those registers into the OTHER
        // shared-memory stage.
        //
        // registers -> SMEM[writeStage]
        //
        // ====================================================
        if (hasNext)
        {
            // -----------------------------------------------
            // nextA registers -> As[writeStage]
            //
            // Keep the same A transpose layout.
            // -----------------------------------------------
            int slotA = 0;

#pragma unroll
            for (int loadOffset = 0;
                 loadOffset < BM;
                 loadOffset += strideA)
            {
                const int rowA =
                    interRowA + loadOffset;

                const int colA =
                    interColA * 4;

                const float4 tmp =
                    nextA[slotA];

                As[writeStage][(colA + 0) * AS_STRIDE +
                               rowA] = tmp.x;

                As[writeStage][(colA + 1) * AS_STRIDE +
                               rowA] = tmp.y;

                As[writeStage][(colA + 2) * AS_STRIDE +
                               rowA] = tmp.z;

                As[writeStage][(colA + 3) * AS_STRIDE +
                               rowA] = tmp.w;

                ++slotA;
            }

            // -----------------------------------------------
            // nextB registers -> Bs[writeStage]
            //
            // Keep exactly the same padding mapping.
            // -----------------------------------------------
            int slotB = 0;

#pragma unroll
            for (int loadOffset = 0;
                 loadOffset < BK;
                 loadOffset += strideB)
            {
                const int rowB =
                    interRowB + loadOffset;

                const int logicalColB =
                    interColB * 4;

                const int physicalColB =
                    logicalColB +
                    (logicalColB >= BS_HALF
                         ? BS_PADDING
                         : 0);

                reinterpret_cast<float4 *>(
                    &Bs[writeStage][rowB * BS_STRIDE +
                                    physicalColB])[0] =
                    nextB[slotB];

                ++slotB;
            }

            // -----------------------------------------------
            // Before the next iteration computes
            // As[writeStage]/Bs[writeStage],
            // all threads must finish filling that tile.
            // -----------------------------------------------
            __syncthreads();

            // -----------------------------------------------
            // Ping-pong:
            //
            // Current:
            //     read 0 / write 1
            //
            // Next:
            //     read 1 / write 0
            //
            // -----------------------------------------------
            readStage = writeStage;
        }
    }

    // ========================================================
    //
    // WRITE C
    //
    // Completely unchanged from the previous kernel.
    //
    // Each thread writes its TM x TN result tile.
    //
    // ========================================================

#pragma unroll
    for (int i = 0;
         i < TM;
         ++i)
    {
#pragma unroll
        for (int j = 0;
             j < TN;
             j += 4)
        {
            const int resultBase =
                i * TN + j;

            const float4 out =
                make_float4(
                    threadRes[resultBase + 0],
                    threadRes[resultBase + 1],
                    threadRes[resultBase + 2],
                    threadRes[resultBase + 3]);

            const int cIndex =
                (threadRow * TM + i) * N +
                threadCol * TN +
                j;

            reinterpret_cast<float4 *>(
                &C[cIndex])[0] =
                out;
        }
    }
}