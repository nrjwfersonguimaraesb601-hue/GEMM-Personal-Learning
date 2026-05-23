# GPU Device Info

这份文档记录了我当前开发机器上的 GPU 设备信息，以及如何通过 `cudaGetDeviceProperties` API 自己查看这些参数。

## 我的设备信息

查询方式：

- `cudaGetDeviceProperties`：通过 CUDA Runtime API 获取设备属性
- `nvidia-smi`：补充驱动版本、PCI Bus ID 等信息

实测设备如下：

- GPU 名称（图形处理器名称）：`NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA Driver Version（CUDA 驱动版本）：`13.2`
- CUDA Runtime Version（CUDA 运行时版本）：`12.6`
- Compute Capability（计算能力）：`8.9`
- Streaming Multiprocessors (SMs)（流式多处理器数量）：`24`
- CUDA Cores（CUDA 核心数）：`3072`
- GPU Max Clock（GPU 最高主频）：`2370 MHz`
- Memory Clock（显存频率）：`8001 MHz`
- Global Memory（全局显存容量）：`8188 MiB`（`8585216000 bytes`）
- Memory Bus Width（显存位宽）：`128-bit`
- L2 Cache Size（二级缓存大小）：`33554432 bytes`（约 `32 MiB`）
- Constant Memory（常量内存大小）：`65536 bytes`
- Shared Memory Per Block（每个线程块的共享内存）：`49152 bytes`
- Registers Per Block（每个线程块可用寄存器数）：`65536`
- Warp Size（线程束大小）：`32`
- Max Threads Per SM（每个 SM 的最大线程数）：`1536`
- Max Threads Per Block（每个线程块的最大线程数）：`1024`
- Max Thread Block Dimensions（线程块维度上限）：`(1024, 1024, 64)`
- Max Grid Dimensions（网格维度上限）：`(2147483647, 65535, 65535)`
- Texture Alignment（纹理对齐字节数）：`512 bytes`
- Concurrent Copy and Kernel Execution（是否支持拷贝与内核并发执行）：`Yes`
- Kernel Timeout（内核是否有运行时限）：`Yes`
- ECC（纠错码支持）：`Disabled`
- Unified Addressing (UVA)（统一虚拟寻址）：`Yes`
- Compute Preemption（计算抢占支持）：`Yes`
- Cooperative Kernel Launch（协作式内核启动支持）：`Yes`
- Multi-Device Cooperative Launch（多设备协作式内核启动支持）：`No`
- PCI Bus ID（PCI 总线编号）：`00000000:01:00.0`
- Compute Mode（计算模式）：`Default`

## 如何自己查看

可以新建一个最小 CUDA 程序，用 `cudaGetDeviceProperties` 打印设备属性。

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "device_count = " << device_count << "\n";

    for (int i = 0; i < device_count; ++i) {
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
```

编译方式：

```bash
nvcc device_info.cu -o device_info
```

运行方式：

```bash
./device_info
```

## 这些参数对 GEMM 学习最有帮助的点

- `Compute Capability`：决定这张卡支持哪些 CUDA 特性，也会影响优化策略。
- `SM 数量`：影响理论并行规模和 occupancy 上限。
- `Shared Memory Per Block`：直接影响 block tiling / shared memory caching 的 tile 设计。
- `Registers Per Block`：会影响 register blocking 和 occupancy 之间的平衡。
- `Warp Size`：CUDA 基本执行单位，很多访存和计算优化都围绕它展开。
- `Memory Bus Width`、`Memory Clock`：帮助理解显存带宽瓶颈。
- `Max Threads Per Block`：决定 kernel 的 block 配置边界。

## 本次查询命令

本次设备信息主要来自下面两个来源：

```bash
/usr/local/cuda-12.6/extras/demo_suite/deviceQuery
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id --format=csv,noheader
```
