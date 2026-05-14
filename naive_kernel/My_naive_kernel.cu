// implement of 1024*1024 matrix in CUDA
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    if (row < M && col < N)
    {
        for (int k = 0; k < K; ++k)
        {
            // A[row][k]*B[k][col]
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
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

    // malloc
    float *A_host = (float *)malloc(nByte_A);
    float *B_host = (float *)malloc(nByte_B);
    float *C_host = (float *)malloc(nByte_C);
    float *C_from_gpu_host = (float *)malloc(nByte_C);

    // initial
    memset(C_host, 0, nByte_C);
    memset(C_from_gpu_host, 0, nByte_C);

    initialData(A_host, M * K);
    initialData(B_host, N * K);

    // cudaMalloc
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;

    CHECK(cudaMalloc((void **)&A_dev, nByte_A));
    CHECK(cudaMalloc((void **)&B_dev, nByte_B));
    CHECK(cudaMalloc((void **)&C_dev, nByte_C));

    // cudaMemcpy
    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(C_dev, C_host, nByte_C, cudaMemcpyHostToDevice));

    // define dim3
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    // launch kernel
    calculate_Matrix<<<gridDim, blockDim>>>(M, N, K, A_dev, B_dev, C_dev);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // cudaMencpu_back
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

    // free mem
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu_host);
}