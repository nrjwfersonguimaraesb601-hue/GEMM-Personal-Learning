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

#include "main_Wraptiling_kernel.cu"

namespace
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int TM = 8;
    constexpr int TN = 4;
    constexpr int WMITER = 2;
    constexpr int WNITER = 2;

    constexpr int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    constexpr size_t SHARED_MEMORY_BYTES =
        static_cast<size_t>(BM * BK + BK * BN) * sizeof(float);
    constexpr float ABS_TOLERANCE = 1.0e-3f;
    constexpr float REL_TOLERANCE = 1.0e-6f;

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
        unsigned int seed = 20260731u;
        bool single_case = false;
        bool check = true;
        bool csv = false;
    };

    struct Result
    {
        int M = 0;
        int N = 0;
        int K = 0;
        float min_ms = 0.0f;
        float avg_ms = 0.0f;
        float max_ms = 0.0f;
        double avg_gflops = 0.0;
        double best_gflops = 0.0;
        float max_abs_error = 0.0f;
        bool check_attempted = false;
        bool passed = false;
        std::string note;
    };

    void printUsage(const char *program)
    {
        std::cout
            << "Usage:\n"
            << "  " << program << "                         # run default size suite\n"
            << "  " << program << " M N K                   # run one size\n"
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
            << "Kernel configuration:\n"
            << "  BM=" << BM << ", BN=" << BN << ", BK=" << BK
            << ", WM=" << WM << ", WN=" << WN
            << ", TM=" << TM << ", TN=" << TN
            << ", WMITER=" << WMITER << ", WNITER=" << WNITER << '\n'
            << "Constraints: M % " << BM << " == 0, N % " << BN
            << " == 0, K % " << BK << " == 0.\n";
    }

    const char *nextValue(int &index, int argc, char **argv, const char *option)
    {
        if (++index >= argc)
        {
            std::cerr << "Missing value for " << option << '\n';
            std::exit(EXIT_FAILURE);
        }
        return argv[index];
    }

    int parsePositiveInt(const char *text, const char *name)
    {
        char *end = nullptr;
        const long value = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max())
        {
            std::cerr << "Invalid positive integer for " << name << ": "
                      << text << '\n';
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
            std::cerr << "Invalid unsigned integer for " << name << ": "
                      << text << '\n';
            std::exit(EXIT_FAILURE);
        }
        return static_cast<unsigned int>(value);
    }

    void validateSize(int M, int N, int K)
    {
        if (M % BM != 0 || N % BN != 0 || K % BK != 0)
        {
            std::cerr << "Unsupported size M=" << M << ", N=" << N
                      << ", K=" << K << "; require M % " << BM
                      << " == 0, N % " << BN << " == 0, K % " << BK
                      << " == 0.\n";
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
                printUsage(argv[0]);
                std::exit(EXIT_SUCCESS);
            }
            else if (arg == "--m")
            {
                config.M = parsePositiveInt(
                    nextValue(i, argc, argv, "--m"), "--m");
                config.single_case = true;
            }
            else if (arg == "--n")
            {
                config.N = parsePositiveInt(
                    nextValue(i, argc, argv, "--n"), "--n");
                config.single_case = true;
            }
            else if (arg == "--k")
            {
                config.K = parsePositiveInt(
                    nextValue(i, argc, argv, "--k"), "--k");
                config.single_case = true;
            }
            else if (arg == "--warmup")
            {
                config.warmup = parsePositiveInt(
                    nextValue(i, argc, argv, "--warmup"), "--warmup");
            }
            else if (arg == "--iters")
            {
                config.iterations = parsePositiveInt(
                    nextValue(i, argc, argv, "--iters"), "--iters");
            }
            else if (arg == "--max-check-dim")
            {
                config.max_check_dim = parsePositiveInt(
                    nextValue(i, argc, argv, "--max-check-dim"),
                    "--max-check-dim");
            }
            else if (arg == "--seed")
            {
                config.seed = parseUnsignedInt(
                    nextValue(i, argc, argv, "--seed"), "--seed");
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
                std::cerr
                    << "Expected exactly three positional arguments: M N K\n";
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

    void fillRandom(std::vector<float> &values, std::mt19937 &generator)
    {
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float &value : values)
        {
            value = distribution(generator);
        }
    }

    void cpuGemm(const std::vector<float> &A,
                 const std::vector<float> &B,
                 std::vector<float> &C,
                 int M, int N, int K)
    {
        std::fill(C.begin(), C.end(), 0.0f);
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

    bool compareResults(const std::vector<float> &reference,
                        const std::vector<float> &actual,
                        int N,
                        float &max_abs_error)
    {
        max_abs_error = 0.0f;
        size_t first_bad = reference.size();

        for (size_t i = 0; i < reference.size(); ++i)
        {
            const float error = std::fabs(reference[i] - actual[i]);
            const float tolerance =
                ABS_TOLERANCE + REL_TOLERANCE * std::fabs(reference[i]);
            max_abs_error = std::max(max_abs_error, error);

            if (first_bad == reference.size() &&
                (!std::isfinite(actual[i]) || error > tolerance))
            {
                first_bad = i;
            }
        }

        if (first_bad != reference.size())
        {
            const float error =
                std::fabs(reference[first_bad] - actual[first_bad]);
            const float tolerance =
                ABS_TOLERANCE +
                REL_TOLERANCE * std::fabs(reference[first_bad]);
            std::cerr << "Mismatch at (" << first_bad / N << ", "
                      << first_bad % N << "): CPU=" << reference[first_bad]
                      << ", GPU=" << actual[first_bad]
                      << ", abs_error=" << error
                      << ", tolerance=" << tolerance << '\n';
            return false;
        }
        return true;
    }

    Result runCase(const Config &config, int M, int N, int K)
    {
        validateSize(M, N, K);

        Result result;
        result.M = M;
        result.N = N;
        result.K = K;

        const size_t count_A = static_cast<size_t>(M) * K;
        const size_t count_B = static_cast<size_t>(K) * N;
        const size_t count_C = static_cast<size_t>(M) * N;
        const size_t bytes_A = count_A * sizeof(float);
        const size_t bytes_B = count_B * sizeof(float);
        const size_t bytes_C = count_C * sizeof(float);

        std::vector<float> A(count_A);
        std::vector<float> B(count_B);
        const bool should_check =
            config.check && std::max({M, N, K}) <= config.max_check_dim;
        std::vector<float> C_reference(should_check ? count_C : 0);
        std::vector<float> C_actual(should_check ? count_C : 0);

        std::mt19937 generator(
            config.seed + static_cast<unsigned int>(M * 31 + N * 17 + K * 13));
        fillRandom(A, generator);
        fillRandom(B, generator);

        float *device_A = nullptr;
        float *device_B = nullptr;
        float *device_C = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_A), bytes_A));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_B), bytes_B));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_C), bytes_C));
        CUDA_CHECK(cudaMemcpy(
            device_A, A.data(), bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_B, B.data(), bytes_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_C, 0, bytes_C));

        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid(N / BN, M / BM);

        for (int i = 0; i < config.warmup; ++i)
        {
            sgemm_2d_register_tiling<
                BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER>
                <<<grid, block>>>(device_A, device_B, device_C, M, N, K);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        result.min_ms = std::numeric_limits<float>::max();
        float total_ms = 0.0f;

        for (int i = 0; i < config.iterations; ++i)
        {
            CUDA_CHECK(cudaEventRecord(start));
            sgemm_2d_register_tiling<
                BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER>
                <<<grid, block>>>(device_A, device_B, device_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            total_ms += elapsed_ms;
            result.min_ms = std::min(result.min_ms, elapsed_ms);
            result.max_ms = std::max(result.max_ms, elapsed_ms);
        }

        result.avg_ms = total_ms / config.iterations;
        const double operations =
            2.0 * static_cast<double>(M) * N * K;
        result.avg_gflops = operations / (result.avg_ms * 1.0e6);
        result.best_gflops = operations / (result.min_ms * 1.0e6);

        if (should_check)
        {
            result.check_attempted = true;
            CUDA_CHECK(cudaMemcpy(
                C_actual.data(), device_C, bytes_C, cudaMemcpyDeviceToHost));
            cpuGemm(A, B, C_reference, M, N, K);
            result.passed = compareResults(
                C_reference, C_actual, N, result.max_abs_error);
        }
        else if (config.check)
        {
            result.note = "CPU check skipped";
        }
        else
        {
            result.note = "CPU check disabled";
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(device_A));
        CUDA_CHECK(cudaFree(device_B));
        CUDA_CHECK(cudaFree(device_C));

        return result;
    }

    void printDeviceInfo()
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::cout << "device: " << properties.name << '\n'
                  << "compute capability: " << properties.major << '.'
                  << properties.minor << '\n'
                  << "SM count: " << properties.multiProcessorCount << '\n'
                  << "kernel: BM=" << BM << ", BN=" << BN << ", BK=" << BK
                  << ", WM=" << WM << ", WN=" << WN
                  << ", TM=" << TM << ", TN=" << TN
                  << ", WMITER=" << WMITER << ", WNITER=" << WNITER << '\n'
                  << "block: " << THREADS_PER_BLOCK << " threads ("
                  << WARPS_PER_BLOCK << " warps), shared memory: "
                  << SHARED_MEMORY_BYTES << " bytes\n";
    }

    void printTableHeader()
    {
        std::cout << std::left
                  << std::setw(8) << "M"
                  << std::setw(8) << "N"
                  << std::setw(8) << "K"
                  << std::setw(10) << "check"
                  << std::setw(12) << "min(ms)"
                  << std::setw(12) << "avg(ms)"
                  << std::setw(12) << "max(ms)"
                  << std::setw(14) << "avg GFLOPS"
                  << std::setw(14) << "best GFLOPS"
                  << std::setw(12) << "max err"
                  << "note\n";
    }

    void printTableRow(const Result &result)
    {
        const std::string status =
            result.check_attempted ? (result.passed ? "PASS" : "FAIL") : "SKIP";
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
                  << std::setw(12) << result.max_abs_error
                  << result.note << '\n';
    }

    void printCsvHeader()
    {
        std::cout
            << "M,N,K,threads,warps,warmup,iters,check,status,max_abs_error,"
               "min_ms,avg_ms,max_ms,avg_gflops,best_gflops,note\n";
    }

    void printCsvRow(const Result &result, const Config &config)
    {
        const char *status =
            !result.check_attempted ? "SKIP" : (result.passed ? "PASS" : "FAIL");
        std::cout << result.M << ',' << result.N << ',' << result.K << ','
                  << THREADS_PER_BLOCK << ',' << WARPS_PER_BLOCK << ','
                  << config.warmup << ',' << config.iterations << ','
                  << (result.check_attempted ? "yes" : "no") << ','
                  << status << ',' << result.max_abs_error << ','
                  << result.min_ms << ',' << result.avg_ms << ','
                  << result.max_ms << ',' << result.avg_gflops << ','
                  << result.best_gflops << ',' << result.note << '\n';
    }
} // namespace

int main(int argc, char **argv)
{
    const Config config = parseArgs(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));

    std::vector<std::tuple<int, int, int>> cases;
    if (config.single_case)
    {
        cases.emplace_back(config.M, config.N, config.K);
    }
    else
    {
        cases = {
            {256, 256, 256},
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096},
            {4096, 256, 4096},
            {256, 4096, 4096},
        };
    }

    if (config.csv)
    {
        printCsvHeader();
    }
    else
    {
        printDeviceInfo();
        std::cout << "warmup: " << config.warmup
                  << ", iterations: " << config.iterations << '\n'
                  << "CPU check: "
                  << (config.check ? "enabled" : "disabled")
                  << ", max check dim: " << config.max_check_dim << '\n';
        printTableHeader();
    }

    bool all_checks_passed = true;
    for (const auto &[M, N, K] : cases)
    {
        const Result result = runCase(config, M, N, K);
        if (config.csv)
        {
            printCsvRow(result, config);
        }
        else
        {
            printTableRow(result);
        }

        if (result.check_attempted && !result.passed)
        {
            all_checks_passed = false;
        }
    }

    return all_checks_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
