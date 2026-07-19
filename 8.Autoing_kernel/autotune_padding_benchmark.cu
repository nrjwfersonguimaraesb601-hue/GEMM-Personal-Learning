#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Keep the Stage 7 kernel in a separate file so the autotuner does not
// silently change its compute or memory-access implementation.
#include "Shared_Memory_Layout_Padding_kernel.cu"

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess)                                              \
        {                                                                       \
            std::ostringstream oss__;                                            \
            oss__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "   \
                  << cudaGetErrorString(error__);                                \
            throw std::runtime_error(oss__.str());                               \
        }                                                                       \
    } while (0)

struct Options
{
    int warmup = 5;
    int iterations = 20;
    int verifySize = 256;
    float tolerance = 1.0e-2f;
    bool verify = true;
    std::string suite = "quick";
    std::string csvPath = "autotune_results.csv";
};

struct TestCase
{
    std::string name;
    int M;
    int N;
    int K;
};

struct ResourceInfo
{
    int registersPerThread = 0;
    int staticSharedBytes = 0;
    int maxThreadsPerBlock = 0;
};

using LaunchFn = void (*)(const float *, const float *, float *, int, int, int,
                          cudaStream_t);
using ResourceFn = ResourceInfo (*)();

struct Config
{
    std::string name;
    int BM;
    int BN;
    int BK;
    int TM;
    int TN;
    int threads;
    int declaredSharedBytes;
    LaunchFn launch;
    ResourceFn queryResources;
};

struct VerificationResult
{
    bool pass = false;
    float maxError = std::numeric_limits<float>::infinity();
};

struct BenchmarkResult
{
    int configIndex = -1;
    std::string caseName;
    int M = 0;
    int N = 0;
    int K = 0;
    float minMs = 0.0f;
    float avgMs = 0.0f;
    float maxMs = 0.0f;
    double avgGflops = 0.0;
    double bestGflops = 0.0;
};

template <int BM, int BN, int BK, int TM, int TN>
void launchKernel(const float *A, const float *B, float *C,
                  int M, int N, int K, cudaStream_t stream)
{
    static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0,
                  "All tile sizes must be positive");
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert((BM * BN) % (TM * TN) == 0,
                  "Thread tiles must exactly cover the block tile");

    constexpr int threads = (BM * BN) / (TM * TN);
    static_assert(threads <= 1024, "Too many threads per block");
    static_assert(threads % 32 == 0,
                  "Use a whole number of warps for this experiment");

    static_assert(BK % 4 == 0, "The current float4 A load requires BK % 4 == 0");
    static_assert(BN % 8 == 0,
                  "The current Bs midpoint padding requires BN % 8 == 0");
    static_assert(TN % 4 == 0,
                  "The current float4 C store requires TN % 4 == 0");

    // These constraints come from the existing Stage 7 cooperative-load
    // mapping. They do not optimize or redesign the kernel; they only reject
    // parameter sets that the current mapping cannot represent safely.
    static_assert(threads >= BK / 4,
                  "Not enough threads for the current A float4 load mapping");
    static_assert(threads % (BK / 4) == 0,
                  "threads must be divisible by BK / 4");
    static_assert(threads / (BK / 4) <= BM,
                  "Current A load mapping would create rowA >= BM");

    static_assert(threads >= BN / 4,
                  "Not enough threads for the current B float4 load mapping");
    static_assert(threads % (BN / 4) == 0,
                  "threads must be divisible by BN / 4");
    static_assert(threads / (BN / 4) <= BK,
                  "Current B load mapping would create rowB >= BK");

    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
    {
        throw std::runtime_error("Matrix dimensions are not divisible by the selected tile");
    }

    dim3 block(threads, 1, 1);
    dim3 grid(N / BN, M / BM, 1);

    sgemm_shared_memory_layout_padding<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

template <int BM, int BN, int BK, int TM, int TN>
ResourceInfo queryKernelResources()
{
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        sgemm_shared_memory_layout_padding<BM, BN, BK, TM, TN>));

    ResourceInfo info;
    info.registersPerThread = attr.numRegs;
    info.staticSharedBytes = static_cast<int>(attr.sharedSizeBytes);
    info.maxThreadsPerBlock = attr.maxThreadsPerBlock;
    return info;
}

template <int BM, int BN, int BK, int TM, int TN>
Config makeConfig(const std::string &name)
{
    constexpr int threads = (BM * BN) / (TM * TN);
    constexpr int pad = 4;
    constexpr int sharedBytes =
        (BK * (BM + pad) + BK * (BN + pad)) * static_cast<int>(sizeof(float));

    return Config{name,
                  BM,
                  BN,
                  BK,
                  TM,
                  TN,
                  threads,
                  sharedBytes,
                  &launchKernel<BM, BN, BK, TM, TN>,
                  &queryKernelResources<BM, BN, BK, TM, TN>};
}

std::vector<Config> buildConfigs()
{
    // C00 is the unchanged Stage 7 configuration and is used as the baseline.
    // The other entries only change compile-time tile parameters.
    return {
        makeConfig<64, 64, 8, 8, 8>("C00_baseline_64x64x8_8x8"),
        makeConfig<32, 64, 8, 8, 8>("C01_32x64x8_8x8"),
        makeConfig<64, 32, 8, 8, 8>("C02_64x32x8_8x8"),
        makeConfig<64, 128, 8, 8, 8>("C03_64x128x8_8x8"),
        makeConfig<128, 64, 8, 8, 8>("C04_128x64x8_8x8"),
        makeConfig<128, 128, 8, 8, 8>("C05_128x128x8_8x8"),
        makeConfig<64, 64, 16, 8, 8>("C06_64x64x16_8x8"),
        makeConfig<64, 128, 16, 8, 8>("C07_64x128x16_8x8"),
        makeConfig<128, 64, 16, 8, 8>("C08_128x64x16_8x8"),
        makeConfig<128, 128, 16, 8, 8>("C09_128x128x16_8x8"),
        makeConfig<64, 64, 8, 4, 8>("C10_64x64x8_4x8"),
        makeConfig<64, 64, 8, 8, 4>("C11_64x64x8_8x4"),
        makeConfig<64, 64, 16, 4, 8>("C12_64x64x16_4x8"),
        makeConfig<64, 64, 16, 8, 4>("C13_64x64x16_8x4"),
    };
}

Options parseOptions(int argc, char **argv)
{
    Options options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto requireValue = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("Missing value after ") + name);
            }
            return argv[++i];
        };

        if (arg == "--warmup")
        {
            options.warmup = std::stoi(requireValue("--warmup"));
        }
        else if (arg == "--iters")
        {
            options.iterations = std::stoi(requireValue("--iters"));
        }
        else if (arg == "--verify-size")
        {
            options.verifySize = std::stoi(requireValue("--verify-size"));
        }
        else if (arg == "--tolerance")
        {
            options.tolerance = std::stof(requireValue("--tolerance"));
        }
        else if (arg == "--suite")
        {
            options.suite = requireValue("--suite");
        }
        else if (arg == "--csv")
        {
            options.csvPath = requireValue("--csv");
        }
        else if (arg == "--no-verify")
        {
            options.verify = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: ./autotune_padding_bench [options]\n"
                << "  --suite quick|full      quick: 1024^3 and 4096^3\n"
                << "                          full: historical square/rectangular cases\n"
                << "  --warmup N              warmup launches per configuration\n"
                << "  --iters N               measured launches per configuration\n"
                << "  --verify-size N         CPU verification matrix size (default 256)\n"
                << "  --tolerance X           maximum absolute error (default 1e-2)\n"
                << "  --csv PATH              CSV output path\n"
                << "  --no-verify             skip CPU correctness verification\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.warmup < 0 || options.iterations <= 0 || options.verifySize <= 0)
    {
        throw std::runtime_error("warmup >= 0, iters > 0 and verify-size > 0 are required");
    }
    if (options.suite != "quick" && options.suite != "full")
    {
        throw std::runtime_error("--suite must be quick or full");
    }

    return options;
}

std::vector<TestCase> buildTestCases(const std::string &suite)
{
    if (suite == "quick")
    {
        return {
            {"square_1024", 1024, 1024, 1024},
            {"square_4096", 4096, 4096, 4096},
        };
    }

    return {
        {"square_256", 256, 256, 256},
        {"square_512", 512, 512, 512},
        {"square_1024", 1024, 1024, 1024},
        {"square_2048", 2048, 2048, 2048},
        {"square_4096", 4096, 4096, 4096},
        {"m4096_n256_k4096", 4096, 256, 4096},
        {"m256_n4096_k4096", 256, 4096, 4096},
    };
}

void fillDeterministic(std::vector<float> &data, std::uint32_t seed)
{
    std::uint32_t state = seed;
    for (float &value : data)
    {
        state = state * 1664525u + 1013904223u;
        const int centered = static_cast<int>((state >> 8) % 2001u) - 1000;
        value = static_cast<float>(centered) / 2000.0f;
    }
}

void cpuReference(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C,
                  int M, int N, int K)
{
    std::fill(C.begin(), C.end(), 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            const float a = A[static_cast<std::size_t>(m) * K + k];
            const float *bRow = &B[static_cast<std::size_t>(k) * N];
            float *cRow = &C[static_cast<std::size_t>(m) * N];
            for (int n = 0; n < N; ++n)
            {
                cRow[n] += a * bRow[n];
            }
        }
    }
}

VerificationResult verifyConfig(const Config &config,
                                int size,
                                float tolerance,
                                const std::vector<float> &hA,
                                const std::vector<float> &hB,
                                const std::vector<float> &reference,
                                float *dA,
                                float *dB,
                                float *dC)
{
    config.launch(dA, dB, dC, size, size, size, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(static_cast<std::size_t>(size) * size);
    CUDA_CHECK(cudaMemcpy(output.data(), dC,
                          output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        maxError = std::max(maxError, std::fabs(output[i] - reference[i]));
    }

    return {maxError <= tolerance, maxError};
}

BenchmarkResult benchmarkConfig(const Config &config,
                                int configIndex,
                                const TestCase &test,
                                const Options &options,
                                float *dA,
                                float *dB,
                                float *dC)
{
    for (int i = 0; i < options.warmup; ++i)
    {
        config.launch(dA, dB, dC, test.M, test.N, test.K, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times;
    times.reserve(options.iterations);

    for (int i = 0; i < options.iterations; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        config.launch(dA, dB, dC, test.M, test.N, test.K, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        times.push_back(elapsedMs);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const float minMs = *std::min_element(times.begin(), times.end());
    const float maxMs = *std::max_element(times.begin(), times.end());
    const float avgMs = std::accumulate(times.begin(), times.end(), 0.0f) /
                        static_cast<float>(times.size());

    const double operations =
        2.0 * static_cast<double>(test.M) *
        static_cast<double>(test.N) *
        static_cast<double>(test.K);

    BenchmarkResult result;
    result.configIndex = configIndex;
    result.caseName = test.name;
    result.M = test.M;
    result.N = test.N;
    result.K = test.K;
    result.minMs = minMs;
    result.avgMs = avgMs;
    result.maxMs = maxMs;
    result.avgGflops = operations / (static_cast<double>(avgMs) * 1.0e6);
    result.bestGflops = operations / (static_cast<double>(minMs) * 1.0e6);
    return result;
}

void printConfigTable(const std::vector<Config> &configs,
                      const std::vector<ResourceInfo> &resources)
{
    std::cout << "\nConfigurations\n";
    std::cout << std::left << std::setw(30) << "name"
              << std::right << std::setw(5) << "BM"
              << std::setw(5) << "BN"
              << std::setw(5) << "BK"
              << std::setw(5) << "TM"
              << std::setw(5) << "TN"
              << std::setw(9) << "threads"
              << std::setw(9) << "regs/th"
              << std::setw(11) << "smem(B)" << '\n';

    for (std::size_t i = 0; i < configs.size(); ++i)
    {
        const auto &c = configs[i];
        const auto &r = resources[i];
        std::cout << std::left << std::setw(30) << c.name
                  << std::right << std::setw(5) << c.BM
                  << std::setw(5) << c.BN
                  << std::setw(5) << c.BK
                  << std::setw(5) << c.TM
                  << std::setw(5) << c.TN
                  << std::setw(9) << c.threads
                  << std::setw(9) << r.registersPerThread
                  << std::setw(11) << r.staticSharedBytes << '\n';
    }
}

void printPerCaseRanking(const std::vector<BenchmarkResult> &results,
                         const std::vector<Config> &configs,
                         const std::vector<TestCase> &tests)
{
    std::cout << "\nPer-case ranking (Avg GFLOPS)\n";
    for (const auto &test : tests)
    {
        std::vector<const BenchmarkResult *> rows;
        for (const auto &result : results)
        {
            if (result.caseName == test.name)
            {
                rows.push_back(&result);
            }
        }
        std::sort(rows.begin(), rows.end(), [](const auto *lhs, const auto *rhs) {
            return lhs->avgGflops > rhs->avgGflops;
        });

        std::cout << "\n[" << test.name << "]\n";
        const std::size_t count = std::min<std::size_t>(5, rows.size());
        for (std::size_t rank = 0; rank < count; ++rank)
        {
            const auto &row = *rows[rank];
            std::cout << "  " << (rank + 1) << ". "
                      << std::setw(30) << std::left
                      << configs[row.configIndex].name
                      << std::right << std::fixed << std::setprecision(2)
                      << row.avgGflops << " GFLOPS, "
                      << row.avgMs << " ms\n";
        }
    }
}

void printOverallRanking(const std::vector<BenchmarkResult> &results,
                         const std::vector<Config> &configs,
                         const std::vector<TestCase> &tests)
{
    // Overall score is the geometric mean of each configuration's speedup
    // relative to C00 on the same test case. A score above 1.0 beats the
    // current Stage 7 configuration on average.
    std::map<std::string, double> baselineByCase;
    for (const auto &result : results)
    {
        if (result.configIndex == 0)
        {
            baselineByCase[result.caseName] = result.avgGflops;
        }
    }

    struct ScoreRow
    {
        int configIndex;
        double score;
        int cases;
    };

    std::vector<ScoreRow> scores;
    for (std::size_t configIndex = 0; configIndex < configs.size(); ++configIndex)
    {
        double logSum = 0.0;
        int count = 0;
        for (const auto &result : results)
        {
            if (result.configIndex != static_cast<int>(configIndex))
            {
                continue;
            }
            const auto it = baselineByCase.find(result.caseName);
            if (it == baselineByCase.end() || it->second <= 0.0)
            {
                continue;
            }
            logSum += std::log(result.avgGflops / it->second);
            ++count;
        }
        if (count > 0)
        {
            scores.push_back({static_cast<int>(configIndex),
                              std::exp(logSum / count),
                              count});
        }
    }

    std::sort(scores.begin(), scores.end(), [](const ScoreRow &lhs, const ScoreRow &rhs) {
        return lhs.score > rhs.score;
    });

    std::cout << "\nOverall ranking\n";
    std::cout << "Score = geometric mean speedup relative to C00 baseline.\n";
    for (std::size_t rank = 0; rank < scores.size(); ++rank)
    {
        const auto &row = scores[rank];
        std::cout << "  " << (rank + 1) << ". "
                  << std::setw(30) << std::left
                  << configs[row.configIndex].name
                  << std::right << " score="
                  << std::fixed << std::setprecision(4) << row.score
                  << " (" << row.cases << " cases)\n";
    }
}

int main(int argc, char **argv)
{
    try
    {
        const Options options = parseOptions(argc, argv);
        const std::vector<Config> configs = buildConfigs();
        const std::vector<TestCase> tests = buildTestCases(options.suite);

        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "CUDA GEMM Stage 8 Autotuning\n"
                  << "Kernel: unchanged Stage 7 shared-memory padding kernel\n"
                  << "Device: " << prop.name << " (SM "
                  << prop.major << prop.minor << ")\n"
                  << "Suite: " << options.suite
                  << ", warmup=" << options.warmup
                  << ", iterations=" << options.iterations << "\n"
                  << "CSV: " << options.csvPath << "\n";

        std::vector<ResourceInfo> resources;
        resources.reserve(configs.size());
        for (const Config &config : configs)
        {
            resources.push_back(config.queryResources());
        }
        printConfigTable(configs, resources);

        std::vector<VerificationResult> verification(
            configs.size(), VerificationResult{true, 0.0f});

        if (options.verify)
        {
            const int S = options.verifySize;
            std::cout << "\nCorrectness verification: "
                      << S << '^' << 3
                      << ", tolerance=" << options.tolerance << "\n";

            std::vector<float> hA(static_cast<std::size_t>(S) * S);
            std::vector<float> hB(static_cast<std::size_t>(S) * S);
            std::vector<float> reference(static_cast<std::size_t>(S) * S);
            fillDeterministic(hA, 1u);
            fillDeterministic(hB, 2u);
            cpuReference(hA, hB, reference, S, S, S);

            float *dA = nullptr;
            float *dB = nullptr;
            float *dC = nullptr;
            const std::size_t bytes = static_cast<std::size_t>(S) * S * sizeof(float);
            CUDA_CHECK(cudaMalloc(&dA, bytes));
            CUDA_CHECK(cudaMalloc(&dB, bytes));
            CUDA_CHECK(cudaMalloc(&dC, bytes));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

            for (std::size_t i = 0; i < configs.size(); ++i)
            {
                const Config &config = configs[i];
                if (S % config.BM != 0 || S % config.BN != 0 || S % config.BK != 0)
                {
                    verification[i] = {false, std::numeric_limits<float>::infinity()};
                    std::cout << "  SKIP " << std::setw(30) << std::left << config.name
                              << " verification size is not divisible by tile\n";
                    continue;
                }

                verification[i] = verifyConfig(
                    config, S, options.tolerance,
                    hA, hB, reference, dA, dB, dC);

                std::cout << "  "
                          << (verification[i].pass ? "PASS " : "FAIL ")
                          << std::setw(30) << std::left << config.name
                          << " max_error=" << std::scientific
                          << verification[i].maxError << std::defaultfloat << '\n';
            }

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        }

        std::ofstream csv(options.csvPath);
        if (!csv)
        {
            throw std::runtime_error("Cannot open CSV output: " + options.csvPath);
        }
        csv << "config,BM,BN,BK,TM,TN,threads,registers_per_thread,"
               "static_shared_bytes,verify_pass,verify_max_error,case,M,N,K,"
               "min_ms,avg_ms,max_ms,avg_gflops,best_gflops\n";

        std::vector<BenchmarkResult> results;

        for (const TestCase &test : tests)
        {
            std::cout << "\nBenchmark case " << test.name
                      << " (M=" << test.M
                      << ", N=" << test.N
                      << ", K=" << test.K << ")\n";

            const std::size_t elementsA =
                static_cast<std::size_t>(test.M) * test.K;
            const std::size_t elementsB =
                static_cast<std::size_t>(test.K) * test.N;
            const std::size_t elementsC =
                static_cast<std::size_t>(test.M) * test.N;

            std::vector<float> hA(elementsA);
            std::vector<float> hB(elementsB);
            fillDeterministic(hA, 11u + static_cast<std::uint32_t>(test.M));
            fillDeterministic(hB, 17u + static_cast<std::uint32_t>(test.N));

            float *dA = nullptr;
            float *dB = nullptr;
            float *dC = nullptr;
            CUDA_CHECK(cudaMalloc(&dA, elementsA * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, elementsB * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, elementsC * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), elementsA * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), elementsB * sizeof(float),
                                  cudaMemcpyHostToDevice));

            for (std::size_t i = 0; i < configs.size(); ++i)
            {
                const Config &config = configs[i];
                if (!verification[i].pass)
                {
                    std::cout << "  SKIP " << config.name
                              << " because correctness verification failed\n";
                    continue;
                }
                if (test.M % config.BM != 0 ||
                    test.N % config.BN != 0 ||
                    test.K % config.BK != 0)
                {
                    std::cout << "  SKIP " << config.name
                              << " because this case is not tile-divisible\n";
                    continue;
                }

                const BenchmarkResult result = benchmarkConfig(
                    config, static_cast<int>(i), test, options, dA, dB, dC);
                results.push_back(result);

                std::cout << "  " << std::setw(30) << std::left << config.name
                          << std::right << std::fixed << std::setprecision(4)
                          << " avg=" << std::setw(9) << result.avgMs << " ms"
                          << "  avg=" << std::setw(10) << std::setprecision(2)
                          << result.avgGflops << " GFLOPS"
                          << "  best=" << std::setw(10)
                          << result.bestGflops << " GFLOPS\n";

                const ResourceInfo &resource = resources[i];
                csv << config.name << ','
                    << config.BM << ',' << config.BN << ',' << config.BK << ','
                    << config.TM << ',' << config.TN << ',' << config.threads << ','
                    << resource.registersPerThread << ','
                    << resource.staticSharedBytes << ','
                    << (verification[i].pass ? "PASS" : "FAIL") << ','
                    << verification[i].maxError << ','
                    << test.name << ',' << test.M << ',' << test.N << ',' << test.K << ','
                    << result.minMs << ',' << result.avgMs << ',' << result.maxMs << ','
                    << result.avgGflops << ',' << result.bestGflops << '\n';
                csv.flush();
            }

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        }

        printPerCaseRanking(results, configs, tests);
        printOverallRanking(results, configs, tests);

        std::cout << "\nFinished. Raw data written to "
                  << options.csvPath << "\n";
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
