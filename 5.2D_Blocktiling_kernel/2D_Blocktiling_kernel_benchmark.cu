#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

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

struct BenchmarkConfig
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 10;
    int iters = 50;
    int bx = THREADS_PER_BLOCK;
    int by = 1;
    int max_check_dim = 2048;
    unsigned int seed = 20260708u;
    bool run_check = true;
    bool csv = false;
    bool single_case = false;
};

struct BenchmarkResult
{
    int M = 0;
    int N = 0;
    int K = 0;
    int bx = 0;
    int by = 0;
    int warmup = 0;
    int iters = 0;
    bool check_attempted = false;
    bool passed = false;
    float max_abs_error = 0.0f;
    float min_ms = 0.0f;
    float avg_ms = 0.0f;
    float max_ms = 0.0f;
    double avg_gflops = 0.0;
    double best_gflops = 0.0;
    double device_gib = 0.0;
    double host_gib = 0.0;
    string note;
};

void printUsage(const char *prog)
{
    cout << "Usage:\n"
         << "  " << prog << "                    # run default benchmark suite\n"
         << "  " << prog << " M N K              # run one case\n"
         << "  " << prog << " [options]\n\n"
         << "Options:\n"
         << "  --m <int>            matrix rows of A and C\n"
         << "  --n <int>            matrix cols of B and C\n"
         << "  --k <int>            matrix cols of A / rows of B\n"
         << "  --warmup <int>       warmup iterations\n"
         << "  --iters <int>        benchmark iterations\n"
         << "  --bx <int>           blockDim.x (must be " << THREADS_PER_BLOCK << " for this kernel)\n"
         << "  --by <int>           blockDim.y (must be 1 for this kernel)\n"
         << "  --seed <uint>        random seed\n"
         << "  --max-check-dim <n>  skip CPU check when max(M,N,K) > n\n"
         << "  --no-check           disable CPU correctness check\n"
         << "  --csv                print CSV rows\n"
         << "  --help               show this message\n";
}

int parseInt(const char *value, const char *name)
{
    char *end = nullptr;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0')
    {
        cerr << "Invalid integer for " << name << ": " << value << endl;
        exit(1);
    }
    if (parsed <= 0)
    {
        cerr << name << " must be positive: " << value << endl;
        exit(1);
    }
    return static_cast<int>(parsed);
}

unsigned int parseUInt(const char *value, const char *name)
{
    char *end = nullptr;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0')
    {
        cerr << "Invalid unsigned integer for " << name << ": " << value << endl;
        exit(1);
    }
    return static_cast<unsigned int>(parsed);
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

float maxAbsDiff(const float *cpu, const float *gpu, int count)
{
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i)
    {
        max_err = max(max_err, fabs(cpu[i] - gpu[i]));
    }
    return max_err;
}

bool checkResult(const float *cpu, const float *gpu, int M, int N, float eps)
{
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(cpu[i] - gpu[i]) > eps)
        {
            const int row = i / N;
            const int col = i % N;
            printf("mismatch at (%d, %d): cpu=%f gpu=%f\n",
                   row, col, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

BenchmarkResult runCase(const BenchmarkConfig &cfg, int M, int N, int K)
{
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.bx = cfg.bx;
    result.by = cfg.by;
    result.warmup = cfg.warmup;
    result.iters = cfg.iters;

    const size_t nByte_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t nByte_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t nByte_C = static_cast<size_t>(M) * N * sizeof(float);

    result.device_gib = static_cast<double>(nByte_A + nByte_B + nByte_C) / (1024.0 * 1024.0 * 1024.0);
    result.host_gib = static_cast<double>(nByte_A + nByte_B + 2 * nByte_C) / (1024.0 * 1024.0 * 1024.0);

    float *A_host = static_cast<float *>(malloc(nByte_A));
    float *B_host = static_cast<float *>(malloc(nByte_B));
    float *C_cpu_host = static_cast<float *>(malloc(nByte_C));
    float *C_gpu_host = static_cast<float *>(malloc(nByte_C));

    if (A_host == nullptr || B_host == nullptr || C_cpu_host == nullptr || C_gpu_host == nullptr)
    {
        cerr << "Host memory allocation failed for M=" << M << " N=" << N << " K=" << K << endl;
        exit(1);
    }

    srand(cfg.seed + static_cast<unsigned int>(M * 31 + N * 17 + K * 13));
    initialData(A_host, M * K);
    initialData(B_host, K * N);
    memset(C_cpu_host, 0, nByte_C);
    memset(C_gpu_host, 0, nByte_C);

    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), nByte_A));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), nByte_B));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&C_dev), nByte_C));

    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(C_dev, 0, nByte_C));

    dim3 blockDim(cfg.bx, cfg.by);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < cfg.warmup; ++i)
    {
        sgemm_2d_register_tiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
            A_dev, B_dev, C_dev, M, N, K);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    result.min_ms = numeric_limits<float>::max();
    result.max_ms = 0.0f;

    for (int i = 0; i < cfg.iters; ++i)
    {
        CHECK(cudaEventRecord(start));
        sgemm_2d_register_tiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
            A_dev, B_dev, C_dev, M, N, K);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
        result.min_ms = min(result.min_ms, elapsed_ms);
        result.max_ms = max(result.max_ms, elapsed_ms);
    }

    result.avg_ms = total_ms / cfg.iters;
    result.avg_gflops = (2.0 * static_cast<double>(M) * N * K) / (result.avg_ms * 1e6);
    result.best_gflops = (2.0 * static_cast<double>(M) * N * K) / (result.min_ms * 1e6);

    if (cfg.run_check)
    {
        if (max(M, max(N, K)) <= cfg.max_check_dim)
        {
            result.check_attempted = true;
            CHECK(cudaMemcpy(C_gpu_host, C_dev, nByte_C, cudaMemcpyDeviceToHost));
            GEMM_cpu(A_host, B_host, C_cpu_host, M, N, K);
            result.max_abs_error = maxAbsDiff(C_cpu_host, C_gpu_host, M * N);
            result.passed = checkResult(C_cpu_host, C_gpu_host, M, N, 1e-3f);
        }
        else
        {
            result.note = "CPU check skipped";
        }
    }
    else
    {
        result.note = "CPU check disabled";
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(A_dev));
    CHECK(cudaFree(B_dev));
    CHECK(cudaFree(C_dev));

    free(A_host);
    free(B_host);
    free(C_cpu_host);
    free(C_gpu_host);

    return result;
}

void printDeviceInfo(int dev)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    cout << "device: " << prop.name << "\n";
    cout << "compute capability: " << prop.major << "." << prop.minor << "\n";
    cout << "global memory: " << fixed << setprecision(2)
         << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GiB\n";
    cout << "SM count: " << prop.multiProcessorCount << "\n";
    cout << "max threads per block: " << prop.maxThreadsPerBlock << "\n";
}

void printCsvHeader()
{
    cout << "M,N,K,bx,by,warmup,iters,check,status,max_abs_error,min_ms,avg_ms,max_ms,avg_gflops,best_gflops,device_gib,host_gib,note\n";
}

void printResultCsv(const BenchmarkResult &r)
{
    const char *status = !r.check_attempted ? "SKIP" : (r.passed ? "PASS" : "FAIL");
    cout << r.M << ',' << r.N << ',' << r.K << ','
         << r.bx << ',' << r.by << ','
         << r.warmup << ',' << r.iters << ','
         << (r.check_attempted ? "yes" : "no") << ','
         << status << ','
         << r.max_abs_error << ','
         << r.min_ms << ',' << r.avg_ms << ',' << r.max_ms << ','
         << r.avg_gflops << ',' << r.best_gflops << ','
         << r.device_gib << ',' << r.host_gib << ','
         << r.note << '\n';
}

void printResultTableHeader()
{
    cout << left
         << setw(8) << "M"
         << setw(8) << "N"
         << setw(8) << "K"
         << setw(12) << "block"
         << setw(10) << "check"
         << setw(12) << "min(ms)"
         << setw(12) << "avg(ms)"
         << setw(12) << "max(ms)"
         << setw(14) << "avg GFLOPS"
         << setw(14) << "best GFLOPS"
         << setw(12) << "max err"
         << "note\n";
}

void printResultTable(const BenchmarkResult &r)
{
    const string check_status = r.check_attempted ? (r.passed ? "PASS" : "FAIL") : "SKIP";
    const string block = to_string(r.bx) + "x" + to_string(r.by);
    cout << left << fixed << setprecision(4)
         << setw(8) << r.M
         << setw(8) << r.N
         << setw(8) << r.K
         << setw(12) << block
         << setw(10) << check_status
         << setw(12) << r.min_ms
         << setw(12) << r.avg_ms
         << setw(12) << r.max_ms
         << setw(14) << r.avg_gflops
         << setw(14) << r.best_gflops
         << setw(12) << r.max_abs_error
         << r.note << '\n';
}

BenchmarkConfig parseArgs(int argc, char **argv)
{
    BenchmarkConfig cfg;
    vector<int> positional;

    for (int i = 1; i < argc; ++i)
    {
        const string arg = argv[i];
        if (arg == "--help")
        {
            printUsage(argv[0]);
            exit(0);
        }
        else if (arg == "--m")
        {
            cfg.M = parseInt(argv[++i], "--m");
            cfg.single_case = true;
        }
        else if (arg == "--n")
        {
            cfg.N = parseInt(argv[++i], "--n");
            cfg.single_case = true;
        }
        else if (arg == "--k")
        {
            cfg.K = parseInt(argv[++i], "--k");
            cfg.single_case = true;
        }
        else if (arg == "--warmup")
        {
            cfg.warmup = parseInt(argv[++i], "--warmup");
        }
        else if (arg == "--iters")
        {
            cfg.iters = parseInt(argv[++i], "--iters");
        }
        else if (arg == "--bx")
        {
            cfg.bx = parseInt(argv[++i], "--bx");
        }
        else if (arg == "--by")
        {
            cfg.by = parseInt(argv[++i], "--by");
        }
        else if (arg == "--seed")
        {
            cfg.seed = parseUInt(argv[++i], "--seed");
        }
        else if (arg == "--max-check-dim")
        {
            cfg.max_check_dim = parseInt(argv[++i], "--max-check-dim");
        }
        else if (arg == "--no-check")
        {
            cfg.run_check = false;
        }
        else if (arg == "--csv")
        {
            cfg.csv = true;
        }
        else if (arg.rfind("--", 0) == 0)
        {
            cerr << "Unknown option: " << arg << endl;
            exit(1);
        }
        else
        {
            positional.push_back(parseInt(argv[i], "positional argument"));
        }
    }

    if (positional.size() == 3)
    {
        cfg.M = positional[0];
        cfg.N = positional[1];
        cfg.K = positional[2];
        cfg.single_case = true;
    }
    else if (!positional.empty())
    {
        cerr << "Expected either zero positional arguments or exactly three: M N K" << endl;
        exit(1);
    }

    if (cfg.bx * cfg.by > 1024)
    {
        cerr << "Invalid block size: bx * by must be <= 1024" << endl;
        exit(1);
    }

    if (cfg.bx != THREADS_PER_BLOCK || cfg.by != 1)
    {
        cerr << "2D blocktiling kernel expects blockDim = ("
             << THREADS_PER_BLOCK << ", 1)" << endl;
        exit(1);
    }

    return cfg;
}

int main(int argc, char **argv)
{
    BenchmarkConfig cfg = parseArgs(argc, argv);

    const int dev = 0;
    CHECK(cudaSetDevice(dev));

    vector<tuple<int, int, int>> cases;
    if (cfg.single_case)
    {
        cases.push_back(make_tuple(cfg.M, cfg.N, cfg.K));
    }
    else
    {
        cases = {
            make_tuple(256, 256, 256),
            make_tuple(512, 512, 512),
            make_tuple(1024, 1024, 1024),
            make_tuple(2048, 2048, 2048),
            make_tuple(4096, 4096, 4096),
            make_tuple(1023, 1023, 1023),
            make_tuple(4096, 256, 4096),
            make_tuple(256, 4096, 4096)};
    }

    if (!cfg.csv)
    {
        printDeviceInfo(dev);
        cout << "warmup: " << cfg.warmup << ", iterations: " << cfg.iters
             << ", block: (" << cfg.bx << ", " << cfg.by << ")\n";
        cout << "CPU check: " << (cfg.run_check ? "enabled" : "disabled")
             << ", max check dim: " << cfg.max_check_dim << "\n";
        printResultTableHeader();
    }
    else
    {
        printCsvHeader();
    }

    for (size_t i = 0; i < cases.size(); ++i)
    {
        int M = 0;
        int N = 0;
        int K = 0;
        tie(M, N, K) = cases[i];
        BenchmarkResult result = runCase(cfg, M, N, K);
        if (cfg.csv)
        {
            printResultCsv(result);
        }
        else
        {
            printResultTable(result);
        }
    }

    return 0;
}
