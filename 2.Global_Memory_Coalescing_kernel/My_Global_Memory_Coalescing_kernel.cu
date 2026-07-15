// Naive CUDA GEMM demo for a 1024 x 1024 matrix multiplication.
// 这是一个最基础的 CUDA GEMM 示例，用来理解：
// 1. host/device 内存分配与拷贝
// 2. thread 如何映射到输出矩阵 C 的一个元素
// 3. 最朴素 naive kernel 的计算方式
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

// Simple CUDA error-check helper.
// 用宏包一层，避免每次都手写 if (err != cudaSuccess)。
#define CHECK(call)                                          \
    if ((call) != cudaSuccess)                               \
    {                                                        \
        printf("CUDA error at %s %d\n", __FILE__, __LINE__); \
        exit(1);                                             \
    }

void initialData(float *arr, int n)
{
    // Fill the input buffer with random values in [0, 1].
    // 用随机数初始化输入矩阵，便于后面和 CPU 结果做对比。
    for (int i = 0; i < n; i++)
        arr[i] = rand() / (float)RAND_MAX;
}

void GEMM_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    // CPU reference implementation:
    // C[M][N] = A[M][K] x B[K][N]
    //
    // Row-major layout:
    // A[i][k] -> A[i * K + k]
    // B[k][j] -> B[k * N + j]
    // C[i][j] -> C[i * N + j]
    //
    // 这个函数主要作为“金标准 / golden reference”，
    // 用来验证 GPU kernel 的结果是否正确。
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
    // 2D thread mapping:
    // each thread computes exactly one output element C[row][col].
    //
    // 当前映射方式是：
    // - threadIdx.x / blockIdx.x -> 输出矩阵的行 row
    // - threadIdx.y / blockIdx.y -> 输出矩阵的列 col
    //
    // 也就是说，warp 内线程主要沿着行方向展开，而不是沿着列方向展开。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    if (row < M && col < N)
    {
        // Compute the dot product of:
        // - row "row" from A
        // - column "col" from B
        //
        // 即：
        // C[row][col] = sum_k A[row][k] * B[k][col]
        for (int k = 0; k < K; ++k)
        {
            // A[row][k] * B[k][col]
            //
            // 这是最朴素的 global memory 版本：
            // - 不使用 shared memory
            // - 不做寄存器分块
            // - 每次循环都直接从显存读取 A 和 B
            //
            // 访问特征上：
            // - A[row * K + k]：同一 warp 内 row 连续变化，地址步长约为 K
            // - B[k * N + col]：同一 warp 内 col 固定，很多线程读取同一个 B 元素
            // - C[row * N + col]：最终写回时 row 连续变化，地址步长约为 N
            //
            // 因此这个版本适合作为 non-coalesced 访问的对照组。
            sum += A[row * K + k] * B[k * N + col];
        }

        // Write the final accumulation result to C[row][col].
        C[row * N + col] = sum;
    }
}

int main()
{
    // Select GPU 0.
    // 默认选择第 0 张 GPU。
    int dev = 0;
    cudaSetDevice(dev);

    // GEMM shape:
    // A is M x K
    // B is K x N
    // C is M x N
    int M = 1024, N = 1024, K = 1024;
    int nByte_A = M * K * sizeof(float);
    int nByte_B = K * N * sizeof(float);
    int nByte_C = M * N * sizeof(float);

    // Allocate host memory on CPU side.
    // A_host / B_host: input matrices
    // C_host: CPU reference output
    // C_from_gpu_host: result copied back from GPU
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

    // Define execution configuration.
    //
    // blockDim(32, 32) means:
    // - x dimension maps to output rows
    // - y dimension maps to output columns
    // - one block contains 1024 threads
    //
    // gridDim uses ceil division so the whole M x N output region
    // is covered even when M or N is not a multiple of 32.
    dim3 blockDim(32, 32);
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

    // Compare CPU and GPU outputs element by element.
    // 这里使用一个较宽松的误差阈值 1e-3f，
    // 因为浮点加法顺序可能带来微小误差。
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
