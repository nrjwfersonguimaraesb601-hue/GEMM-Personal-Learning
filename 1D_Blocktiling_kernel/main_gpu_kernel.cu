__global__ void calculate_Matrix(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    const int num_thread = BM * BN / TM;

    // shared memory

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 重新映射because of TM

    int tid = threadIdx.x;

    int threadCol = tid % BN;
    int threadRow = tid / BN;

    int interRowA = tid / BK;
    int interColA = tid % BK;

    int interRowB = tid / BN;
    int interColB = tid % BN;

    C += blockIdx.y * BM * N + blockIdx.x * BN;
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;

    float threadResult[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        As[interRowA * BK + interColA] = A[interRowA * K + interColA];
        Bs[interRowB * BN + interColB] = B[interRowB * N + interColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float Btemp = Bs[dotIdx * BN + threadCol];

            for (int resIdx = 0; resIdx < TM; ++resIdx)
            {
                int row = threadRow * TM + resIdx;

                threadResult[resIdx] += As[row * BK + dotIdx] * Btemp;
            }
        }

        __syncthreads();
    }
    for (int resIdx = 0; resIdx < TM; ++resIdx)
    {
        int row = threadRow * TM + resIdx;

        C[row * N + threadCol] = threadResult[resIdx];
    }
}