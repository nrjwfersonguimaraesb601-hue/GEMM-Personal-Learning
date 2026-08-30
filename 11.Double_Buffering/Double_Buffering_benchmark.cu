#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "DoubleBuffering_main_kernel.cu"

namespace
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int AS_PADDING = 4;
    constexpr int BS_PADDING = 4;
    constexpr int AS_STRIDE = BM + AS_PADDING;
    constexpr int BS_STRIDE = BN + BS_PADDING;
    constexpr int THREADS_PER_BLOCK = BM * BN / (TM * TN);
    constexpr size_t SHARED_MEMORY_BYTES =
        static_cast<size_t>(2 * BK) * (AS_STRIDE + BS_STRIDE) * sizeof(float);

#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        const cudaError_t error = (call);                                \
        if (error != cudaSuccess)                                        \
        {                                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(error)); \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (false)

    struct Config
    {
        int M = 1024;
        int N = 1024;
        int K = 1024;
        int warmup = 10;
        int iterations = 50;
        int max_check_dim = 512;
        unsigned int seed = 20260810u;
        bool single_case = false;
        bool check = true;
        bool csv = false;
    };

    struct Result
    {
        int M;
        int N;
        int K;
        float min_ms;
        float avg_ms;
        float max_ms;
        double avg_gflops;
        double best_gflops;
        float max_abs_error;
        bool check_attempted;
        bool passed;
    };

    void usage(const char *program)
    {
        std::cout
            << "Usage:\n"
            << "  " << program << "                         # default size suite\n"
            << "  " << program << " M N K                   # one size\n"
            << "  " << program << " [options]\n\n"
            << "Options:\n"
            << "  --m/--n/--k <int>       select one size\n"
            << "  --warmup <int>          warm-up launches (default 10)\n"
            << "  --iters <int>           measured launches (default 50)\n"
            << "  --max-check-dim <int>   CPU-check size limit (default 512)\n"
            << "  --seed <uint>           input seed\n"
            << "  --no-check              disable CPU correctness check\n"
            << "  --csv                   print machine-readable output\n"
            << "  --help                  show this help\n\n"
            << "Kernel variant: double-buffered As/Bs shared memory\n"
            << "Constraints: M % " << BM << " == 0, N % " << BN
            << " == 0, K % " << BK << " == 0.\n";
    }

    const char *nextValue(int &i, int argc, char **argv, const char *option)
    {
        if (++i >= argc)
        {
            std::cerr << "Missing value for " << option << '\n';
            std::exit(EXIT_FAILURE);
        }
        return argv[i];
    }

    int parsePositiveInt(const char *text, const char *name)
    {
        char *end = nullptr;
        const long value = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max())
        {
            std::cerr << "Invalid positive integer for " << name << ": " << text << '\n';
            std::exit(EXIT_FAILURE);
        }
        return static_cast<int>(value);
    }

    unsigned int parseUnsignedInt(const char *text, const char *name)
    {
        char *end = nullptr;
        const unsigned long value = std::strtoul(text, &end, 10);
        if (end == text || *end != '\0' ||
            value > std::numeric_limits<unsigned int>::max())
        {
            std::cerr << "Invalid unsigned integer for " << name << ": " << text << '\n';
            std::exit(EXIT_FAILURE);
        }
        return static_cast<unsigned int>(value);
    }

    void validateSize(int M, int N, int K)
    {
        if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        {
            std::cerr << "Unsupported size M=" << M << ", N=" << N << ", K=" << K
                      << "; require M % " << BM << " == 0, N % " << BN
                      << " == 0, K % " << BK << " == 0.\n";
            std::exit(EXIT_FAILURE);
        }
    }

    Config parseArgs(int argc, char **argv)
    {
        Config config;
        std::vector<int> positional;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg(argv[i]);
            if (arg == "--help")
            {
                usage(argv[0]);
                std::exit(EXIT_SUCCESS);
            }
            else if (arg == "--m")
            {
                config.M = parsePositiveInt(nextValue(i, argc, argv, "--m"), "--m");
                config.single_case = true;
            }
            else if (arg == "--n")
            {
                config.N = parsePositiveInt(nextValue(i, argc, argv, "--n"), "--n");
                config.single_case = true;
            }
            else if (arg == "--k")
            {
                config.K = parsePositiveInt(nextValue(i, argc, argv, "--k"), "--k");
                config.single_case = true;
            }
            else if (arg == "--warmup")
            {
                config.warmup = parsePositiveInt(nextValue(i, argc, argv, "--warmup"), "--warmup");
            }
            else if (arg == "--iters")
            {
                config.iterations = parsePositiveInt(nextValue(i, argc, argv, "--iters"), "--iters");
            }
            else if (arg == "--max-check-dim")
            {
                config.max_check_dim = parsePositiveInt(
                    nextValue(i, argc, argv, "--max-check-dim"), "--max-check-dim");
            }
            else if (arg == "--seed")
            {
                config.seed = parseUnsignedInt(nextValue(i, argc, argv, "--seed"), "--seed");
            }
            else if (arg == "--no-check")
            {
                config.check = false;
            }
            else if (arg == "--csv")
            {
                config.csv = true;
            }
            else if (arg.rfind("--", 0) == 0)
            {
                std::cerr << "Unknown option: " << arg << '\n';
                std::exit(EXIT_FAILURE);
            }
            else
            {
                positional.push_back(parsePositiveInt(argv[i], "M/N/K"));
            }
        }

        if (!positional.empty())
        {
            if (positional.size() != 3)
            {
                std::cerr << "Expected exactly three positional arguments: M N K\n";
                std::exit(EXIT_FAILURE);
            }
            config.M = positional[0];
            config.N = positional[1];
            config.K = positional[2];
            config.single_case = true;
        }

        if (config.single_case)
        {
            validateSize(config.M, config.N, config.K);
        }
        return config;
    }

    void cpuGemm(const float *A, const float *B, float *C, int M, int N, int K)
    {
        std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
        for (int m = 0; m < M; ++m)
        {
            for (int k = 0; k < K; ++k)
            {
                const float a = A[static_cast<size_t>(m) * K + k];
                for (int n = 0; n < N; ++n)
                {
                    C[static_cast<size_t>(m) * N + n] +=
                        a * B[static_cast<size_t>(k) * N + n];
                }
            }
        }
    }

    bool compare(const std::vector<float> &reference,
                 const std::vector<float> &actual,
                 int N,
                 float &max_abs_error)
    {
        constexpr float abs_tolerance = 1.0e-3f;
        constexpr float rel_tolerance = 1.0e-6f;
        max_abs_error = 0.0f;

        for (size_t i = 0; i < reference.size(); ++i)
        {
            const float error = std::fabs(reference[i] - actual[i]);
            const float tolerance = abs_tolerance + rel_tolerance * std::fabs(reference[i]);
            max_abs_error = std::max(max_abs_error, error);
            if (!std::isfinite(actual[i]) || error > tolerance)
            {
                std::cerr << "Mismatch at (" << i / N << ", " << i % N
                          << "): CPU=" << reference[i] << ", GPU=" << actual[i]
                          << ", abs_error=" << error << ", tolerance=" << tolerance << '\n';
                return false;
            }
        }
        return true;
    }

    Result runCase(const Config &config, int M, int N, int K)
    {
        validateSize(M, N, K);

        const size_t count_A = static_cast<size_t>(M) * K;
        const size_t count_B = static_cast<size_t>(K) * N;
        const size_t count_C = static_cast<size_t>(M) * N;
        const size_t bytes_A = count_A * sizeof(float);
        const size_t bytes_B = count_B * sizeof(float);
        const size_t bytes_C = count_C * sizeof(float);

        std::vector<float> host_A(count_A);
        std::vector<float> host_B(count_B);
        std::mt19937 generator(config.seed + M * 31u + N * 17u + K * 13u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::generate(host_A.begin(), host_A.end(), [&]
                      { return distribution(generator); });
        std::generate(host_B.begin(), host_B.end(), [&]
                      { return distribution(generator); });

        float *device_A = nullptr;
        float *device_B = nullptr;
        float *device_C = nullptr;
        CUDA_CHECK(cudaMalloc(&device_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&device_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&device_C, bytes_C));
        CUDA_CHECK(cudaMemcpy(device_A, host_A.data(), bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_B, host_B.data(), bytes_B, cudaMemcpyHostToDevice));

        const dim3 block(THREADS_PER_BLOCK, 1, 1);
        const dim3 grid(N / BN, M / BM, 1);
        for (int i = 0; i < config.warmup; ++i)
        {
            sgemm_double_buffering<BM, BN, BK, TM, TN>
                <<<grid, block>>>(device_A, device_B, device_C, M, N, K);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float min_ms = std::numeric_limits<float>::max();
        float max_ms = 0.0f;
        float total_ms = 0.0f;

        for (int i = 0; i < config.iterations; ++i)
        {
            CUDA_CHECK(cudaEventRecord(start));
            sgemm_double_buffering<BM, BN, BK, TM, TN>
                <<<grid, block>>>(device_A, device_B, device_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            min_ms = std::min(min_ms, elapsed_ms);
            max_ms = std::max(max_ms, elapsed_ms);
            total_ms += elapsed_ms;
        }

        Result result{M, N, K, min_ms, total_ms / config.iterations, max_ms,
                      0.0, 0.0, 0.0f, false, false};
        const double operations = 2.0 * static_cast<double>(M) * N * K;
        result.avg_gflops = operations / (result.avg_ms * 1.0e6);
        result.best_gflops = operations / (result.min_ms * 1.0e6);

        if (config.check && std::max(M, std::max(N, K)) <= config.max_check_dim)
        {
            result.check_attempted = true;
            std::vector<float> host_C(count_C);
            std::vector<float> reference_C(count_C);
            CUDA_CHECK(cudaMemcpy(host_C.data(), device_C, bytes_C, cudaMemcpyDeviceToHost));
            cpuGemm(host_A.data(), host_B.data(), reference_C.data(), M, N, K);
            result.passed = compare(reference_C, host_C, N, result.max_abs_error);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(device_A));
        CUDA_CHECK(cudaFree(device_B));
        CUDA_CHECK(cudaFree(device_C));
        return result;
    }

    void printResult(const Result &result, bool csv)
    {
        const char *status = result.check_attempted ? (result.passed ? "PASS" : "FAIL") : "SKIP";
        if (csv)
        {
            std::cout << result.M << ',' << result.N << ',' << result.K << ',' << status << ','
                      << result.min_ms << ',' << result.avg_ms << ',' << result.max_ms << ','
                      << result.avg_gflops << ',' << result.best_gflops << ','
                      << result.max_abs_error << '\n';
            return;
        }

        std::cout << std::left << std::fixed << std::setprecision(4)
                  << std::setw(8) << result.M
                  << std::setw(8) << result.N
                  << std::setw(8) << result.K
                  << std::setw(10) << status
                  << std::setw(12) << result.min_ms
                  << std::setw(12) << result.avg_ms
                  << std::setw(12) << result.max_ms
                  << std::setw(14) << result.avg_gflops
                  << std::setw(14) << result.best_gflops
                  << result.max_abs_error << '\n';
    }
} // namespace

int main(int argc, char **argv)
{
    const Config config = parseArgs(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (!config.csv)
    {
        std::cout << "Kernel: sgemm_double_buffering\n"
                  << "Layout: As[2][BK][BM+" << AS_PADDING << "], Bs[2][BK][BN+"
                  << BS_PADDING << "], static shared memory="
                  << SHARED_MEMORY_BYTES << " bytes\n"
                  << "Device: " << properties.name << " (SM " << properties.major
                  << properties.minor << ")\n"
                  << "Tile: BM=" << BM << ", BN=" << BN << ", BK=" << BK
                  << ", TM=" << TM << ", TN=" << TN
                  << ", block=" << THREADS_PER_BLOCK << "x1\n"
                  << "Warmup=" << config.warmup << ", iterations=" << config.iterations
                  << ", CPU check limit=" << config.max_check_dim << "\n\n"
                  << std::left << std::setw(8) << "M" << std::setw(8) << "N"
                  << std::setw(8) << "K" << std::setw(10) << "check"
                  << std::setw(12) << "min(ms)" << std::setw(12) << "avg(ms)"
                  << std::setw(12) << "max(ms)" << std::setw(14) << "avg GFLOPS"
                  << std::setw(14) << "best GFLOPS" << "max error\n";
    }
    else
    {
        std::cout << "M,N,K,check,min_ms,avg_ms,max_ms,avg_gflops,best_gflops,max_abs_error\n";
    }

    std::vector<std::tuple<int, int, int>> cases;
    if (config.single_case)
    {
        cases.emplace_back(config.M, config.N, config.K);
    }
    else
    {
        cases = {{256, 256, 256},
                 {512, 512, 512},
                 {1024, 1024, 1024},
                 {2048, 2048, 2048},
                 {4096, 4096, 4096},
                 {4096, 256, 4096},
                 {256, 4096, 4096}};
    }

    bool all_passed = true;
    for (const auto &problem : cases)
    {
        int M;
        int N;
        int K;
        std::tie(M, N, K) = problem;
        const Result result = runCase(config, M, N, K);
        printResult(result, config.csv);
        all_passed = all_passed && (!result.check_attempted || result.passed);
    }
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
