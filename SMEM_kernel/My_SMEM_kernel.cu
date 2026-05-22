
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

constexpr int TILE = 32;

#define CHECK(call)                                          \
    if ((call) != cudaSuccess)                               \
    {                                                        \
        printf("CUDA error at %s %d\n", __FILE__, __LINE__); \
        exit(1);                                             \
    }

void initialData(float *arr, int n)
{

    for (int i = 0; i < n; i++)
        arr[i] = rand() / (float)RAND_MAX;
}

void GEMM_cpu(float *A, float *B, float *C, int M, int N, int K)
{

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
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
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float temp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += blockDim.x)
    {
        // Shared-memory tiling needs boundary checks so the kernel can
        // safely handle shapes that are not exact multiples of the tile size.
        if (row < M && (bkIdx + tx) < K)
        {
            As[ty][tx] = A[row * K + (bkIdx + tx)];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        if (col < N && (bkIdx + ty) < K)
        {
            Bs[ty][tx] = B[(bkIdx + ty) * N + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < blockDim.x; ++dotIdx)
        {
            temp += As[ty][dotIdx] * Bs[dotIdx][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = temp;
    }
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    int M = 1024, N = 1024, K = 1024;
    int nByte_A = M * K * sizeof(float);
    int nByte_B = K * N * sizeof(float);
    int nByte_C = M * N * sizeof(float);

    float *A_host = (float *)malloc(nByte_A);
    float *B_host = (float *)malloc(nByte_B);
    float *C_host = (float *)malloc(nByte_C);
    float *C_from_gpu_host = (float *)malloc(nByte_C);

    // Initialize output buffers to zero.
    // 先清零输出缓冲区，避免未初始化数据影响对比结果。
    memset(C_host, 0, nByte_C);
    memset(C_from_gpu_host, 0, nByte_C);

    // Randomly initialize input matrices.
    initialData(A_host, M * K);
    initialData(B_host, N * K);

    // Allocate device memory on GPU side.
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;

    CHECK(cudaMalloc((void **)&A_dev, nByte_A));
    CHECK(cudaMalloc((void **)&B_dev, nByte_B));
    CHECK(cudaMalloc((void **)&C_dev, nByte_C));

    // Copy inputs from host to device.
    // 对于 C，这里也把 host 端的 0 拷贝过去，相当于初始化 device output。
    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(C_dev, C_host, nByte_C, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    // Launch the naive GEMM kernel.
    // kernel launch 本身是异步的，所以后面用 synchronize 等待执行完成。
    calculate_Matrix<<<gridDim, blockDim>>>(M, N, K, A_dev, B_dev, C_dev);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy GPU result back to host for validation.
    CHECK(cudaMemcpy(C_from_gpu_host, C_dev, nByte_C, cudaMemcpyDeviceToHost));

    // Run CPU reference GEMM.
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

    // Release GPU memory first, then CPU memory.
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu_host);
}
