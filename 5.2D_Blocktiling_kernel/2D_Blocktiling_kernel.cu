// 2D register block-tiling CUDA GEMM correctness demo.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);

#define CHECK(call)                                                 \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(1);                                                \
        }                                                           \
    } while (0)

int parseInt(const char *value, const char *name)
{
    char *end = nullptr;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0)
    {
        cerr << name << " must be a positive integer: " << value << endl;
        exit(1);
    }
    return static_cast<int>(parsed);
}

void initialData(float *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() / static_cast<float>(RAND_MAX);
    }
}

void GEMM_cpu(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

template <
    const int BLOCK_M,
    const int BLOCK_N,
    const int BLOCK_K,
    const int THREAD_M,
    const int THREAD_N>
__global__ void sgemm_2d_register_tiling(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{
    static_assert(BLOCK_M % THREAD_M == 0, "BLOCK_M must be divisible by THREAD_M");
    static_assert(BLOCK_N % THREAD_N == 0, "BLOCK_N must be divisible by THREAD_N");
    static_assert((BLOCK_M * BLOCK_N) % (THREAD_M * THREAD_N) == 0,
                  "Thread tile must divide block tile");

    constexpr int threadsNum = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
    static_assert(threadsNum % BLOCK_K == 0, "threadsNum must be divisible by BLOCK_K");
    static_assert(threadsNum % BLOCK_N == 0, "threadsNum must be divisible by BLOCK_N");

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;

    const int threadRow = tid / (BLOCK_N / THREAD_N);
    const int threadCol = tid % (BLOCK_N / THREAD_N);

    __shared__ float As[BLOCK_M * BLOCK_K];
    __shared__ float Bs[BLOCK_K * BLOCK_N];

    const int interRowA = tid / BLOCK_K;
    const int interColA = tid % BLOCK_K;
    const int strideA = threadsNum / BLOCK_K;

    const int interRowB = tid / BLOCK_N;
    const int interColB = tid % BLOCK_N;
    const int strideB = threadsNum / BLOCK_N;

    float threadRes[THREAD_M * THREAD_N] = {0.0f};
    float regM[THREAD_M] = {0.0f};
    float regN[THREAD_N] = {0.0f};

    const bool fullBlockMN =
        (blockRow * BLOCK_M + BLOCK_M <= M) &&
        (blockCol * BLOCK_N + BLOCK_N <= N);

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K)
    {
        const bool fullKTile = (bkIdx + BLOCK_K <= K);

        for (int loadOffset = 0; loadOffset < BLOCK_M; loadOffset += strideA)
        {
            const int aTileRow = interRowA + loadOffset;
            const int aTileCol = interColA;
            const int globalRow = blockRow * BLOCK_M + aTileRow;
            const int globalCol = bkIdx + aTileCol;

            if (fullBlockMN && fullKTile)
            {
                As[aTileRow * BLOCK_K + aTileCol] =
                    A[globalRow * K + globalCol];
            }
            else if (globalRow < M && globalCol < K)
            {
                As[aTileRow * BLOCK_K + aTileCol] =
                    A[globalRow * K + globalCol];
            }
            else
            {
                As[aTileRow * BLOCK_K + aTileCol] = 0.0f;
            }
        }

        for (int loadOffset = 0; loadOffset < BLOCK_K; loadOffset += strideB)
        {
            const int bTileRow = interRowB + loadOffset;
            const int bTileCol = interColB;
            const int globalRow = bkIdx + bTileRow;
            const int globalCol = blockCol * BLOCK_N + bTileCol;

            if (fullBlockMN && fullKTile)
            {
                Bs[bTileRow * BLOCK_N + bTileCol] =
                    B[globalRow * N + globalCol];
            }
            else if (globalRow < K && globalCol < N)
            {
                Bs[bTileRow * BLOCK_N + bTileCol] =
                    B[globalRow * N + globalCol];
            }
            else
            {
                Bs[bTileRow * BLOCK_N + bTileCol] = 0.0f;
            }
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx)
        {
#pragma unroll
            for (int i = 0; i < THREAD_M; ++i)
            {
                regM[i] = As[(threadRow * THREAD_M + i) * BLOCK_K + dotIdx];
            }

#pragma unroll
            for (int j = 0; j < THREAD_N; ++j)
            {
                regN[j] = Bs[dotIdx * BLOCK_N + threadCol * THREAD_N + j];
            }

#pragma unroll
            for (int i = 0; i < THREAD_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < THREAD_N; ++j)
                {
                    threadRes[i * THREAD_N + j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < THREAD_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < THREAD_N; ++j)
        {
            const int globalRow = blockRow * BLOCK_M + threadRow * THREAD_M + i;
            const int globalCol = blockCol * BLOCK_N + threadCol * THREAD_N + j;

            if (globalRow < M && globalCol < N)
            {
                C[globalRow * N + globalCol] = threadRes[i * THREAD_N + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    const int dev = 0;
    CHECK(cudaSetDevice(dev));

    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc == 4)
    {
        M = parseInt(argv[1], "M");
        N = parseInt(argv[2], "N");
        K = parseInt(argv[3], "K");
    }
    else if (argc != 1)
    {
        cerr << "Usage: " << argv[0] << " [M N K]" << endl;
        return 1;
    }

    const size_t nByte_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t nByte_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t nByte_C = static_cast<size_t>(M) * N * sizeof(float);

    float *A_host = static_cast<float *>(malloc(nByte_A));
    float *B_host = static_cast<float *>(malloc(nByte_B));
    float *C_host = static_cast<float *>(malloc(nByte_C));
    float *C_from_gpu_host = static_cast<float *>(malloc(nByte_C));

    if (A_host == nullptr || B_host == nullptr || C_host == nullptr || C_from_gpu_host == nullptr)
    {
        cerr << "Host memory allocation failed" << endl;
        return 1;
    }

    srand(20260708u);
    initialData(A_host, M * K);
    initialData(B_host, K * N);
    memset(C_host, 0, nByte_C);
    memset(C_from_gpu_host, 0, nByte_C);

    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), nByte_A));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), nByte_B));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&C_dev), nByte_C));

    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(C_dev, 0, nByte_C));

    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_2d_register_tiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        A_dev, B_dev, C_dev, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(C_from_gpu_host, C_dev, nByte_C, cudaMemcpyDeviceToHost));
    GEMM_cpu(A_host, B_host, C_host, M, N, K);

    bool passed = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(C_host[i] - C_from_gpu_host[i]) > 1e-3f)
        {
            const int row = i / N;
            const int col = i % N;
            printf("mismatch at (%d, %d): cpu=%f gpu=%f\n",
                   row, col, C_host[i], C_from_gpu_host[i]);
            passed = false;
            break;
        }
    }

    if (passed)
    {
        printf("验证通过\n");
    }
    else
    {
        printf("验证失败\n");
    }

    CHECK(cudaFree(A_dev));
    CHECK(cudaFree(B_dev));
    CHECK(cudaFree(C_dev));

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu_host);

    return passed ? 0 : 1;
}
