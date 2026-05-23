#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "device_count = " << device_count << "\n";

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory: "
                  << prop.totalGlobalMem << " bytes\n";
        std::cout << "  MultiProcessor Count: "
                  << prop.multiProcessorCount << "\n";
        std::cout << "  Shared Memory Per Block: "
                  << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Registers Per Block: "
                  << prop.regsPerBlock << "\n";
        std::cout << "  Warp Size: "
                  << prop.warpSize << "\n";
        std::cout << "  Max Threads Per Block: "
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads Dim: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max Grid Size: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        std::cout << "  Clock Rate: "
                  << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  Memory Clock Rate: "
                  << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "  Memory Bus Width: "
                  << prop.memoryBusWidth << "-bit\n";
        std::cout << "  L2 Cache Size: "
                  << prop.l2CacheSize << " bytes\n";
    }

    return 0;
}