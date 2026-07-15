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

#include "main_Vectorize_kerenl.cu"

using namespace std;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);
constexpr float CHECK_ABS_TOLERANCE = 1.0e-3f;
constexpr float CHECK_REL_TOLERANCE = 1.0e-6f;

#define CHECK(call)                                                 \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
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
    int max_check_dim = 4096;
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
         << "  --m <int>            matrix rows of A and C (multiple of " << BM << ")\n"
         << "  --n <int>            matrix cols of B and C (multiple of " << BN << ")\n"
         << "  --k <int>            matrix cols of A / rows of B (multiple of " << BK << ")\n"
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

const char *requireValue(int &index, int argc, char **argv, const char *option)
{
    if (index + 1 >= argc)
    {
        cerr << "Missing value for " << option << endl;
        exit(EXIT_FAILURE);
    }
    return argv[++index];
}

int parseInt(const char *value, const char *name)
{
    char *end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0')
    {
        cerr << "Invalid integer for " << name << ": " << value << endl;
        exit(EXIT_FAILURE);
    }
    if (parsed <= 0 || parsed > numeric_limits<int>::max())
    {
        cerr << name << " must be in the range [1, "
             << numeric_limits<int>::max() << "]: " << value << endl;
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(parsed);
}

unsigned int parseUInt(const char *value, const char *name)
{
    char *end = nullptr;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed > numeric_limits<unsigned int>::max())
    {
        cerr << "Invalid unsigned integer for " << name << ": " << value << endl;
        exit(EXIT_FAILURE);
    }
    return static_cast<unsigned int>(parsed);
}

void validateProblemSize(int M, int N, int K)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        cerr << "Vectorized kernel requires M % " << BM << " == 0, N % "
             << BN << " == 0, and K % " << BK << " == 0; got M="
             << M << ", N=" << N << ", K=" << K << endl;
        exit(EXIT_FAILURE);
    }
}

void initialData(float *arr, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        arr[i] = rand() / static_cast<float>(RAND_MAX);
    }
}

void GEMM_cpu(const float *__restrict__ A,
              const float *__restrict__ B,
              float *__restrict__ C,
              int M, int N, int K)
{
    fill(C, C + static_cast<size_t>(M) * N, 0.0f);

    for (int i = 0; i < M; ++i)
    {
        float *const c_row = C + static_cast<size_t>(i) * N;
        const float *const a_row = A + static_cast<size_t>(i) * K;

        for (int k = 0; k < K; ++k)
        {
            const float a = a_row[k];
            const float *const b_row = B + static_cast<size_t>(k) * N;

            for (int j = 0; j < N; ++j)
            {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

float maxAbsDiff(const float *cpu, const float *gpu, size_t count)
{
    float max_error = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        if (!isfinite(cpu[i]) || !isfinite(gpu[i]))
        {
            return numeric_limits<float>::infinity();
        }
        max_error = max(max_error, fabs(cpu[i] - gpu[i]));
    }
    return max_error;
}

bool checkResult(const float *cpu, const float *gpu, int M, int N,
                 float abs_tolerance, float rel_tolerance)
{
    const size_t count = static_cast<size_t>(M) * N;
    for (size_t i = 0; i < count; ++i)
    {
        const float error = fabs(cpu[i] - gpu[i]);
        const float tolerance =
            abs_tolerance + rel_tolerance * fabs(cpu[i]);
        if (!isfinite(cpu[i]) || !isfinite(gpu[i]) || error > tolerance)
        {
            const size_t row = i / static_cast<size_t>(N);
            const size_t col = i % static_cast<size_t>(N);
            cerr << "mismatch at (" << row << ", " << col << "): cpu="
                 << cpu[i] << " gpu=" << gpu[i] << " abs_error=" << error
                 << " tolerance=" << tolerance << endl;
            return false;
        }
    }
    return true;
}

BenchmarkResult runCase(const BenchmarkConfig &cfg, int M, int N, int K)
{
    validateProblemSize(M, N, K);

    BenchmarkResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.bx = cfg.bx;
    result.by = cfg.by;
    result.warmup = cfg.warmup;
    result.iters = cfg.iters;

    const size_t count_A = static_cast<size_t>(M) * K;
    const size_t count_B = static_cast<size_t>(K) * N;
    const size_t count_C = static_cast<size_t>(M) * N;
    const size_t nByte_A = count_A * sizeof(float);
    const size_t nByte_B = count_B * sizeof(float);
    const size_t nByte_C = count_C * sizeof(float);

    result.device_gib = static_cast<double>(nByte_A + nByte_B + nByte_C) /
                        (1024.0 * 1024.0 * 1024.0);
    result.host_gib = static_cast<double>(nByte_A + nByte_B + 2 * nByte_C) /
                      (1024.0 * 1024.0 * 1024.0);

    float *A_host = static_cast<float *>(malloc(nByte_A));
    float *B_host = static_cast<float *>(malloc(nByte_B));
    float *C_cpu_host = static_cast<float *>(malloc(nByte_C));
    float *C_gpu_host = static_cast<float *>(malloc(nByte_C));

    if (A_host == nullptr || B_host == nullptr ||
        C_cpu_host == nullptr || C_gpu_host == nullptr)
    {
        cerr << "Host memory allocation failed for M=" << M
             << " N=" << N << " K=" << K << endl;
        exit(EXIT_FAILURE);
    }

    srand(cfg.seed + static_cast<unsigned int>(M * 31 + N * 17 + K * 13));
    initialData(A_host, count_A);
    initialData(B_host, count_B);
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

    const dim3 blockDim(cfg.bx, cfg.by);
    const dim3 gridDim(N / BN, M / BM);

    for (int i = 0; i < cfg.warmup; ++i)
    {
        sgemm_vectorize_GEMM_SMEM<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(A_dev, B_dev, C_dev, M, N, K);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    result.min_ms = numeric_limits<float>::max();
    result.max_ms = 0.0f;

    for (int i = 0; i < cfg.iters; ++i)
    {
        CHECK(cudaEventRecord(start));
        sgemm_vectorize_GEMM_SMEM<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(A_dev, B_dev, C_dev, M, N, K);
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
    result.avg_gflops = (2.0 * static_cast<double>(M) * N * K) /
                        (result.avg_ms * 1e6);
    result.best_gflops = (2.0 * static_cast<double>(M) * N * K) /
                         (result.min_ms * 1e6);

    if (cfg.run_check)
    {
        if (max(M, max(N, K)) <= cfg.max_check_dim)
        {
            result.check_attempted = true;
            CHECK(cudaMemcpy(C_gpu_host, C_dev, nByte_C, cudaMemcpyDeviceToHost));
            GEMM_cpu(A_host, B_host, C_cpu_host, M, N, K);
            result.max_abs_error = maxAbsDiff(C_cpu_host, C_gpu_host, count_C);
            result.passed =
                checkResult(C_cpu_host, C_gpu_host, M, N,
                            CHECK_ABS_TOLERANCE, CHECK_REL_TOLERANCE);
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
         << static_cast<double>(prop.totalGlobalMem) /
                (1024.0 * 1024.0 * 1024.0)
         << " GiB\n";
    cout << "SM count: " << prop.multiProcessorCount << "\n";
    cout << "max threads per block: " << prop.maxThreadsPerBlock << "\n";
}

void printCsvHeader()
{
    cout << "M,N,K,bx,by,warmup,iters,check,status,max_abs_error,min_ms,avg_ms,max_ms,avg_gflops,best_gflops,device_gib,host_gib,note\n";
}

void printResultCsv(const BenchmarkResult &r)
{
    const char *status = !r.check_attempted ? "SKIP" :
                         (r.passed ? "PASS" : "FAIL");
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
    const string check_status = r.check_attempted ?
                                    (r.passed ? "PASS" : "FAIL") :
                                    "SKIP";
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
            exit(EXIT_SUCCESS);
        }
        else if (arg == "--m")
        {
            cfg.M = parseInt(requireValue(i, argc, argv, "--m"), "--m");
            cfg.single_case = true;
        }
        else if (arg == "--n")
        {
            cfg.N = parseInt(requireValue(i, argc, argv, "--n"), "--n");
            cfg.single_case = true;
        }
        else if (arg == "--k")
        {
            cfg.K = parseInt(requireValue(i, argc, argv, "--k"), "--k");
            cfg.single_case = true;
        }
        else if (arg == "--warmup")
        {
            cfg.warmup = parseInt(requireValue(i, argc, argv, "--warmup"),
                                  "--warmup");
        }
        else if (arg == "--iters")
        {
            cfg.iters = parseInt(requireValue(i, argc, argv, "--iters"),
                                 "--iters");
        }
        else if (arg == "--bx")
        {
            cfg.bx = parseInt(requireValue(i, argc, argv, "--bx"), "--bx");
        }
        else if (arg == "--by")
        {
            cfg.by = parseInt(requireValue(i, argc, argv, "--by"), "--by");
        }
        else if (arg == "--seed")
        {
            cfg.seed = parseUInt(requireValue(i, argc, argv, "--seed"),
                                 "--seed");
        }
        else if (arg == "--max-check-dim")
        {
            cfg.max_check_dim =
                parseInt(requireValue(i, argc, argv, "--max-check-dim"),
                         "--max-check-dim");
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
            exit(EXIT_FAILURE);
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
        cerr << "Expected either zero positional arguments or exactly three: M N K"
             << endl;
        exit(EXIT_FAILURE);
    }

    if (cfg.bx * cfg.by > 1024)
    {
        cerr << "Invalid block size: bx * by must be <= 1024" << endl;
        exit(EXIT_FAILURE);
    }

    if (cfg.bx != THREADS_PER_BLOCK || cfg.by != 1)
    {
        cerr << "Vectorized kernel expects blockDim = ("
             << THREADS_PER_BLOCK << ", 1)" << endl;
        exit(EXIT_FAILURE);
    }

    if (cfg.single_case)
    {
        validateProblemSize(cfg.M, cfg.N, cfg.K);
    }

    return cfg;
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parseArgs(argc, argv);

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

    bool all_checks_passed = true;
    for (size_t i = 0; i < cases.size(); ++i)
    {
        int M = 0;
        int N = 0;
        int K = 0;
        tie(M, N, K) = cases[i];
        const BenchmarkResult result = runCase(cfg, M, N, K);

        if (cfg.csv)
        {
            printResultCsv(result);
        }
        else
        {
            printResultTable(result);
        }

        if (result.check_attempted && !result.passed)
        {
            all_checks_passed = false;
        }
    }

    return all_checks_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
