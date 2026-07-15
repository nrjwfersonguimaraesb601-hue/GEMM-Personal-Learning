// 1D block-tiling CUDA GEMM demo.
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
constexpr int THREADS_PER_BLOCK = BM * BN / TM;

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

void initialData(float *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() / (float)RAND_MAX;
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

__global__ void calculate_Matrix(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int threadCol = tid % BN;
    const int threadRowGroup = tid / BN;
    const int globalCol = blockCol + threadCol;
    const int globalRowBase = blockRow + threadRowGroup * TM;

    const int aTileRow = tid / BK;
    const int aTileCol = tid % BK;
    const int bTileRow = tid / BN;
    const int bTileCol = tid % BN;

    float threadResult[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        const int globalARow = blockRow + aTileRow;
        const int globalACol = bkIdx + aTileCol;
        if (globalARow < M && globalACol < K)
        {
            As[aTileRow * BK + aTileCol] = A[globalARow * K + globalACol];
        }
        else
        {
            As[aTileRow * BK + aTileCol] = 0.0f;
        }

        const int globalBRow = bkIdx + bTileRow;
        const int globalBCol = blockCol + bTileCol;
        if (globalBRow < K && globalBCol < N)
        {
            Bs[bTileRow * BN + bTileCol] = B[globalBRow * N + globalBCol];
        }
        else
        {
            Bs[bTileRow * BN + bTileCol] = 0.0f;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            const float bValue = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx)
            {
                const int localRow = threadRowGroup * TM + resIdx;
                threadResult[resIdx] += As[localRow * BK + dotIdx] * bValue;
            }
        }

        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; ++resIdx)
    {
        const int globalRow = globalRowBase + resIdx;
        if (globalRow < M && globalCol < N)
        {
            C[globalRow * N + globalCol] = threadResult[resIdx];
        }
    }
}

int main()
{
    const int dev = 0;
    CHECK(cudaSetDevice(dev));

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t nByte_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t nByte_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t nByte_C = static_cast<size_t>(M) * N * sizeof(float);

    float *A_host = static_cast<float *>(malloc(nByte_A));
    float *B_host = static_cast<float *>(malloc(nByte_B));
    float *C_host = static_cast<float *>(malloc(nByte_C));
    float *C_from_gpu_host = static_cast<float *>(malloc(nByte_C));

    memset(C_host, 0, nByte_C);
    memset(C_from_gpu_host, 0, nByte_C);

    initialData(A_host, M * K);
    initialData(B_host, K * N);

    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), nByte_A));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), nByte_B));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&C_dev), nByte_C));

    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(C_dev, 0, nByte_C));

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    calculate_Matrix<<<gridDim, blockDim>>>(M, N, K, A_dev, B_dev, C_dev);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(C_from_gpu_host, C_dev, nByte_C, cudaMemcpyDeviceToHost));
    GEMM_cpu(A_host, B_host, C_host, M, N, K);

    bool passed = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(C_host[i] - C_from_gpu_host[i]) > 1e-3f)
        {
            printf("mismatch at %d: cpu=%f gpu=%f\n", i, C_host[i], C_from_gpu_host[i]);
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
