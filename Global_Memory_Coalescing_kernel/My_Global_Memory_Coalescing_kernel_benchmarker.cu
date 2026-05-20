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

// CUDA API call checker.
// 和基础版一样，这里把错误检查统一封装起来，benchmark 过程中更容易定位失败位置。
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
    // Matrix sizes for a single test case.
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // warmup: run several untimed iterations first
    // to reduce one-time overhead influence (context setup / cache effects).
    int warmup = 10;

    // iters: number of timed iterations used for statistics.
    int iters = 100;

    // Thread block shape.
    int bx = 16;
    int by = 16;

    // Skip CPU reference checking for very large matrices,
    // otherwise CPU verification can dominate total runtime.
    int max_check_dim = 2048;
    unsigned int seed = 20260514u;
    bool run_check = true;
    bool csv = false;

    // single_case = true means the user explicitly specified one GEMM shape.
    bool single_case = false;
};

struct BenchmarkResult
{
    // Echo back the case configuration,方便打印时直接使用。
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
    // Command line help for the benchmark executable.
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
         << "  --bx <int>           blockDim.x\n"
         << "  --by <int>           blockDim.y\n"
         << "  --seed <uint>        random seed\n"
         << "  --max-check-dim <n>  skip CPU check when max(M,N,K) > n\n"
         << "  --no-check           disable CPU correctness check\n"
         << "  --csv                print CSV rows\n"
         << "  --help               show this message\n";
}

int parseInt(const char *value, const char *name)
{
    // Robust integer parsing with validation.
    // 不接受空字符串、非数字内容和非正值。
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
    // Parse unsigned values, mainly for random seed.
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
    // Initialize a dense matrix with random floats in [0, 1].
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

void GEMM_cpu(const float *A, const float *B, float *C, int M, int N, int K)
{
    // CPU reference GEMM in row-major format.
    // This is intentionally straightforward rather than optimized,
    // because correctness is more important than speed here.
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
    // Each CUDA thread computes one output element C[row][col].
    // 一个 thread 对应 C 中的一个点。
    //
    // 这里刻意使用 non-coalesced 的线程映射方式：
    // - threadIdx.x / blockIdx.x -> 输出矩阵的行 row
    // - threadIdx.y / blockIdx.y -> 输出矩阵的列 col
    //
    // 对于常见的 blockDim(32, by)，一个 warp 内通常是 row 连续、col 固定。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        // Accumulate the dot product of A's row and B's column.
        //
        // 访问特征：
        // - A[row * K + k]：warp 内 row 连续变化，地址步长约为 K
        // - B[k * N + col]：warp 内 col 固定，很多线程读取同一个地址
        // - C[row * N + col]：写回时 row 连续变化，地址步长约为 N
        //
        // 因此 A 和 C 都不是按连续地址访问，这正是本 benchmark
        // 想要测量的 non-coalesced naive kernel。
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float maxAbsDiff(const float *cpu, const float *gpu, int count)
{
    // Return the maximum absolute difference across all elements.
    // 用于快速量化 GPU 结果和 CPU 参考结果之间的误差上界。
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i)
    {
        max_err = max(max_err, fabs(cpu[i] - gpu[i]));
    }
    return max_err;
}

bool checkResult(const float *cpu, const float *gpu, int M, int N, float eps)
{
    // Element-wise correctness check.
    // 如果某个元素误差超过 eps，打印其二维坐标，方便定位问题。
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(cpu[i] - gpu[i]) > eps)
        {
            int row = i / N;
            int col = i % N;
            printf("mismatch at (%d, %d): cpu=%f gpu=%f\n", row, col, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

BenchmarkResult runCase(const BenchmarkConfig &cfg, int M, int N, int K)
{
    // Run exactly one benchmark case:
    // 1. allocate host/device memory
    // 2. initialize inputs
    // 3. warm up kernel
    // 4. time multiple iterations with cudaEvent
    // 5. optionally verify against CPU
    // 6. collect metrics and free resources
    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.bx = cfg.bx;
    result.by = cfg.by;
    result.warmup = cfg.warmup;
    result.iters = cfg.iters;

    // Byte sizes of matrices in row-major storage.
    const size_t nByte_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t nByte_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t nByte_C = static_cast<size_t>(M) * N * sizeof(float);

    // Rough memory footprint report:
    // device needs A + B + C
    // host keeps A + B + CPU_C + GPU_C
    result.device_gib = static_cast<double>(nByte_A + nByte_B + nByte_C) / (1024.0 * 1024.0 * 1024.0);
    result.host_gib = static_cast<double>(nByte_A + nByte_B + 2 * nByte_C) / (1024.0 * 1024.0 * 1024.0);

    // Host buffers:
    // A_host / B_host -> inputs
    // C_cpu_host      -> CPU reference result
    // C_gpu_host      -> GPU result copied back for checking
    float *A_host = static_cast<float *>(malloc(nByte_A));
    float *B_host = static_cast<float *>(malloc(nByte_B));
    float *C_cpu_host = static_cast<float *>(malloc(nByte_C));
    float *C_gpu_host = static_cast<float *>(malloc(nByte_C));

    if (A_host == nullptr || B_host == nullptr || C_cpu_host == nullptr || C_gpu_host == nullptr)
    {
        cerr << "Host memory allocation failed for M=" << M << " N=" << N << " K=" << K << endl;
        exit(1);
    }

    // Derive a per-shape seed so different test cases are deterministic
    // but not all initialized with exactly the same data.
    srand(cfg.seed + static_cast<unsigned int>(M * 31 + N * 17 + K * 13));
    initialData(A_host, M * K);
    initialData(B_host, K * N);
    memset(C_cpu_host, 0, nByte_C);
    memset(C_gpu_host, 0, nByte_C);

    // Device buffers on GPU.
    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&A_dev), nByte_A));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&B_dev), nByte_B));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&C_dev), nByte_C));

    // Transfer inputs H2D and clear output buffer on device.
    CHECK(cudaMemcpy(A_dev, A_host, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(C_dev, 0, nByte_C));

    // Execution configuration.
    // ceil division guarantees enough blocks to cover the full output matrix.
    dim3 blockDim(cfg.bx, cfg.by);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Warmup phase:
    // 执行若干次不计时 kernel，减少首次启动成本对 benchmark 的污染。
    for (int i = 0; i < cfg.warmup; ++i)
    {
        calculate_Matrix<<<gridDim, blockDim>>>(M, N, K, A_dev, B_dev, C_dev);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // CUDA events provide GPU-side elapsed time in milliseconds.
    // 相比 CPU wall-clock，更适合测 kernel 执行时间。
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    result.min_ms = numeric_limits<float>::max();
    result.max_ms = 0.0f;

    // Timed loop:
    // each iteration measures one kernel launch latency/runtime.
    for (int i = 0; i < cfg.iters; ++i)
    {
        CHECK(cudaEventRecord(start));
        calculate_Matrix<<<gridDim, blockDim>>>(M, N, K, A_dev, B_dev, C_dev);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(stop));

        // elapsed_ms is the device execution time between the two events.
        float elapsed_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
        result.min_ms = min(result.min_ms, elapsed_ms);
        result.max_ms = max(result.max_ms, elapsed_ms);
    }

    // Average latency and derived throughput.
    //
    // GEMM FLOPs ~= 2 * M * N * K
    // because each multiply-accumulate is treated as 2 floating-point ops.
    result.avg_ms = total_ms / cfg.iters;
    result.avg_gflops = (2.0 * M * N * K) / (result.avg_ms * 1e6);
    result.best_gflops = (2.0 * M * N * K) / (result.min_ms * 1e6);

    if (cfg.run_check)
    {
        // Optional correctness validation.
        // 大矩阵时 CPU 参考实现会很慢，因此允许按维度阈值跳过。
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

    // Clean up all resources before returning the metrics.
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
    // Print a concise summary of the active GPU.
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
    // CSV mode is convenient for importing into Excel / pandas / plotting scripts.
    cout << "M,N,K,bx,by,warmup,iters,check,status,max_abs_error,min_ms,avg_ms,max_ms,avg_gflops,best_gflops,device_gib,host_gib,note\n";
}

void printResultCsv(const BenchmarkResult &r)
{
    // One benchmark result as a single CSV row.
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
    // Human-readable table header for terminal output.
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
    // Human-readable table row.
    string check_status = r.check_attempted ? (r.passed ? "PASS" : "FAIL") : "SKIP";
    string block = to_string(r.bx) + "x" + to_string(r.by);
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
    // Parse command-line arguments into BenchmarkConfig.
    // 支持：
    // 1. 位置参数 M N K
    // 2. --m/--n/--k 形式
    // 3. benchmark 控制选项
    BenchmarkConfig cfg;
    vector<int> positional;

    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
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

    // CUDA hardware limit: one block cannot exceed 1024 threads on most NVIDIA GPUs.
    if (cfg.bx * cfg.by > 1024)
    {
        cerr << "Invalid block size: bx * by must be <= 1024" << endl;
        exit(1);
    }

    return cfg;
}

int main(int argc, char **argv)
{
    // Overall flow of benchmark main:
    // 1. parse config
    // 2. choose GPU
    // 3. build test case list
    // 4. print header
    // 5. run each case and print results
    BenchmarkConfig cfg = parseArgs(argc, argv);

    const int dev = 0;
    CHECK(cudaSetDevice(dev));

    vector<tuple<int, int, int>> cases;
    if (cfg.single_case)
    {
        // User-specified single case.
        cases.push_back(make_tuple(cfg.M, cfg.N, cfg.K));
    }
    else
    {
        // Default benchmark suite:
        // includes square sizes, odd size, and a few rectangular shapes.
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

    // Run all selected benchmark cases one by one.
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
