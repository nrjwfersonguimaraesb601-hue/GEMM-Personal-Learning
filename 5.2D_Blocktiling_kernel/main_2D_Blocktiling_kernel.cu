template <
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN>

__global__ void sgemm_2d_register_tiling(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadsNum = (BM * BN) / (TM * TN);

    const int tid = threadIdx.x;

    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    //--shared--//
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    const int interRowA = tid / BK;
    const int interColA = tid % BK;
    const int strideA = threadsNum / BK;

    const int interRowB = tid / BN;
    const int interColB = tid % BN;
    const int strideB = threadsNum / BN;

    float threadRes[TM * TN] = {0.0f};

    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            As[(interRowA + loadOffset) * BK + interColA] =
                A[(interRowA + loadOffset) * K + interColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            Bs[(interRowB + loadOffset) * BN + interColB] =
                B[(interRowB + loadOffset) * N + interColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for (int i = 0; i < TM; ++i)
            {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int j = 0; j < TN; ++j)
            {
                regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
            }

            for (int i = 0; i < TM; ++i)
            {
                for (int j = 0; j < TN; ++j)
                {
                    threadRes[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            C[(threadRow * TM + i) * N + threadCol * TN + j] =
                threadRes[i * TN + j];
        }
    }
}