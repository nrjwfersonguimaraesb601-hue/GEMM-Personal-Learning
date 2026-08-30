template <
    const int BM,
    const int BN,
    const int BK,
    const int WM,
    const int WN,
    const int TM,
    const int TN,
    const int WMITER,
    const int WNITER>

__global__ void sgemm_2d_register_tiling(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{
    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);

    static_assert(WM % WMITER == 0);
    static_assert(WN % WNITER == 0);

    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    static_assert(TN % 4 == 0);

    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    static_assert(WSUBM % TM == 0);
    static_assert(WSUBN % TN == 0);
    static_assert((WSUBM / TM) * (WSUBN / TN) == 32);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int tid = threadIdx.x;

    // 当前线程属于 block 内的哪个 warp。
    const int warpId = tid / 32;

    // 当前线程是 warp 内的哪个 lane。
    const int laneId = tid % 32;

    // block 在 M、N 方向分别有多少个 warp。
    constexpr int WARPS_M = BM / WM;
    constexpr int WARPS_N = BN / WN;

    // 整个 block 一共有多少个 warp 和线程。
    constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;

    constexpr int threadsNum = WARPS_PER_BLOCK * 32;

    // 当前 warp 位于 block tile 的哪个 warp 行、warp 列。
    const int warpRow = warpId / WARPS_N;

    const int warpCol = warpId % WARPS_N;

    // 一个 warp tile 内，N 方向有多少个 thread tile。
    constexpr int THREAD_TILES_N = WSUBN / TN;

    // 当前 lane 位于 warp tile 内的哪个 thread-tile 行列。
    const int threadRowInWarp = laneId / THREAD_TILES_N;

    const int threadColInWarp = laneId % THREAD_TILES_N;

    constexpr int A_VECS = BM * BK / 4;
    constexpr int B_VECS = BK * BN / 4;

    //--shared--//
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    float threadRes[WMITER * TM * WNITER * TN] = {0.0f};

    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        for (int vecIdx = tid; vecIdx < A_VECS; vecIdx += threadsNum)
        {
            const int scalarIdx = vecIdx * 4;

            const int rowA = scalarIdx / BK;
            const int colA = scalarIdx % BK;

            const float4 value =
                reinterpret_cast<const float4 *>(
                    &A[rowA * K + colA])[0];

            // A 写入
            As[(colA + 0) * BM + rowA] = value.x;
            As[(colA + 1) * BM + rowA] = value.y;
            As[(colA + 2) * BM + rowA] = value.z;
            As[(colA + 3) * BM + rowA] = value.w;
        }
        for (int vecIdx = tid; vecIdx < B_VECS; vecIdx += threadsNum)
        {
            const int scalarIdx = vecIdx * 4;

            const int rowB = scalarIdx / BN;
            const int colB = scalarIdx % BN;

            const float4 value =
                reinterpret_cast<const float4 *>(
                    &B[rowB * N + colB])[0];

            reinterpret_cast<float4 *>(
                &Bs[rowB * BN + colB])[0] =
                value;
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
            {
                for (int i = 0; i < TM; ++i)
                {
                    const int row = warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i;
                    regM[wSubRow * TM + i] = As[dotIdx * BM + row];
                }
            }
            for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
            {
                for (int j = 0; j < TN; ++j)
                {
                    const int col = warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + j;
                    regN[wSubCol * TN + j] = Bs[dotIdx * BN + col];
                }
            }

            for (int wSubRow = 0;
                 wSubRow < WMITER;
                 ++wSubRow)
            {
                for (int wSubCol = 0;
                     wSubCol < WNITER;
                     ++wSubCol)
                {
                    for (int i = 0;
                         i < TM;
                         ++i)
                    {
                        for (int j = 0;
                             j < TN;
                             ++j)
                        {
                            const int resultRow =
                                wSubRow * TM + i;

                            const int resultCol =
                                wSubCol * TN + j;

                            const int resultIndex =
                                resultRow *
                                    (WNITER * TN) +
                                resultCol;

                            threadRes[resultIndex] +=
                                regM[wSubRow * TM + i] *
                                regN[wSubCol * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int wSubRow = 0;
         wSubRow < WMITER;
         ++wSubRow)
    {
        for (int wSubCol = 0;
             wSubCol < WNITER;
             ++wSubCol)
        {
            for (int i = 0;
                 i < TM;
                 ++i)
            {
                for (int j = 0;
                     j < TN;
                     j += 4)
                {
                    const int cRow =
                        warpRow * WM +
                        wSubRow * WSUBM +
                        threadRowInWarp * TM +
                        i;

                    const int cCol =
                        warpCol * WN +
                        wSubCol * WSUBN +
                        threadColInWarp * TN +
                        j;

                    const int resultRow =
                        wSubRow * TM + i;

                    const int resultCol =
                        wSubCol * TN + j;

                    const int resultIndex =
                        resultRow *
                            (WNITER * TN) +
                        resultCol;

                    const float4 result =
                        make_float4(
                            threadRes[resultIndex + 0],
                            threadRes[resultIndex + 1],
                            threadRes[resultIndex + 2],
                            threadRes[resultIndex + 3]);

                    reinterpret_cast<float4 *>(
                        &C[cRow * N + cCol])[0] =
                        result;
                }
            }
        }
    }
}
