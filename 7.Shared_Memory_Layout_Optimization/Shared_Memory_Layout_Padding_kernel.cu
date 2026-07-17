// Most of this file is copied from the 2D block tiling kernel.
// This version vectorizes GMEM/SMEM accesses with float4.

template <
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN>
__global__ void sgemm_shared_memory_layout_padding(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M,
    int N,
    int K)
{
    // float4 requires groups of four floats.
    static_assert(BK % 4 == 0,
                  "BK must be divisible by 4");
    static_assert(BN % 8 == 0,
                  "BN must be divisible by 8 so the half-row padding boundary is float4-aligned");
    static_assert(TN % 4 == 0,
                  "TN must be divisible by 4");

    // Insert one float4-sized gap halfway through each B row. The physical
    // row stride must include this gap for both the load and compute paths.
    constexpr int BS_PADDING = 4;
    constexpr int BS_HALF = BN / 2;
    constexpr int BS_STRIDE = BN + BS_PADDING;

    constexpr int AS_PADDING = 4;
    constexpr int AS_STRIDE = BM + AS_PADDING;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadsNum =
        (BM * BN) / (TM * TN);

    const int tid = threadIdx.x;

    // Current thread is responsible for one TM x TN tile of C.
    const int threadRow =
        tid / (BN / TN);

    const int threadCol =
        tid % (BN / TN);

    // As is logically stored as As[BK][BM] after transposition.
    __shared__ float As[BK * AS_STRIDE];

    // Bs is logically stored as Bs[BK][BN].
    __shared__ float Bs[BK * BS_STRIDE];

    // Move A, B and C to the tile handled by this block.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // --------------------------------------------------------
    // float4 loading coordinates for A
    // One row of A tile contains BK / 4 float4 groups.
    // --------------------------------------------------------
    const int interRowA =
        tid / (BK / 4);

    const int interColA =
        tid % (BK / 4);

    const int strideA =
        threadsNum / (BK / 4);

    // --------------------------------------------------------
    // float4 loading coordinates for B
    // One row of B tile contains BN / 4 float4 groups.
    // --------------------------------------------------------
    const int interRowB =
        tid / (BN / 4);

    const int interColB =
        tid % (BN / 4);

    const int strideB =
        threadsNum / (BN / 4);

    float threadRes[TM * TN] = {0.0f};

    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // Iterate over the K dimension in BK-sized tiles.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // ----------------------------------------------------
        // Load A from GMEM using float4.
        // A is read as A[row][k], but stored as As[k][row].
        // ----------------------------------------------------
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
                    &A[rowA * K + colA])[0];

            // Transpose during GMEM -> SMEM transfer.
            As[(colA + 0) * AS_STRIDE + rowA] = tmp.x;
            As[(colA + 1) * AS_STRIDE + rowA] = tmp.y;
            As[(colA + 2) * AS_STRIDE + rowA] = tmp.z;
            As[(colA + 3) * AS_STRIDE + rowA] = tmp.w;
        }

        // ----------------------------------------------------
        // Load B from GMEM to SMEM using float4.
        // B and Bs both retain the [k][col] layout.
        // ----------------------------------------------------
        for (int loadOffset = 0;
             loadOffset < BK;
             loadOffset += strideB)
        {
            const int rowB =
                interRowB + loadOffset;

            const int loadLogicalColB =
                interColB * 4;

            const float4 tmp =
                reinterpret_cast<const float4 *>(
                    &B[rowB * N + loadLogicalColB])[0];

            const int loadPhysicalColB =
                loadLogicalColB + (loadLogicalColB >= BS_HALF ? BS_PADDING : 0);

            reinterpret_cast<float4 *>(
                &Bs[rowB * BS_STRIDE + loadPhysicalColB])[0] = tmp;
        }

        __syncthreads();

        // Move to the next K tile.
        A += BK;
        B += BK * N;

        // ----------------------------------------------------
        // Compute the current BK tile.
        // ----------------------------------------------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // As is transposed: As[k][row].
            for (int i = 0; i < TM; ++i)
            {
                regM[i] =
                    As[dotIdx * AS_STRIDE +
                       threadRow * TM +
                       i];
            }

            // Bs remains Bs[k][col].
            for (int j = 0; j < TN; ++j)
            {
                const int computeLogicalColB =
                    threadCol * TN + j;

                const int computePhysicalColB =
                    computeLogicalColB +
                    (computeLogicalColB >= BS_HALF ? BS_PADDING : 0);

                regN[j] =
                    Bs[dotIdx * BS_STRIDE +
                       computePhysicalColB];
            }

            // TM x TN outer product.
            for (int i = 0; i < TM; ++i)
            {
                for (int j = 0; j < TN; ++j)
                {
                    threadRes[i * TN + j] +=
                        regM[i] * regN[j];
                }
            }
        }

        // Ensure no thread overwrites As/Bs before every
        // thread finishes using the current tile.
        __syncthreads();
    }

    // --------------------------------------------------------
    // Write C to GMEM four consecutive values at a time.
    // --------------------------------------------------------
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; j += 4)
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
                &C[cIndex])[0] = out;
        }
    }
}
