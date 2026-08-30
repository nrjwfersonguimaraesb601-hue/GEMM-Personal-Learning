NVCC ?= nvcc
CUDA_ARCH ?= sm_89
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(CUDA_ARCH)
BUILD_DIR ?= build

TARGETS := \
	$(BUILD_DIR)/naive_bench \
	$(BUILD_DIR)/coalesced_bench \
	$(BUILD_DIR)/smem_bench \
	$(BUILD_DIR)/register_1d_bench \
	$(BUILD_DIR)/register_2d_bench \
	$(BUILD_DIR)/vectorized_bench \
	$(BUILD_DIR)/padding_bench \
	$(BUILD_DIR)/autotuning_bench \
	$(BUILD_DIR)/warp_tiling_bench \
	$(BUILD_DIR)/double_buffering_bench \
	$(BUILD_DIR)/cublas_bench

.PHONY: all naive coalesced smem 1d 2d vectorized padding autotuning \
	warp double-buffering cublas device-info clean

all: $(TARGETS)

naive: $(BUILD_DIR)/naive_bench
coalesced: $(BUILD_DIR)/coalesced_bench
smem: $(BUILD_DIR)/smem_bench
1d: $(BUILD_DIR)/register_1d_bench
2d: $(BUILD_DIR)/register_2d_bench
vectorized: $(BUILD_DIR)/vectorized_bench
padding: $(BUILD_DIR)/padding_bench
autotuning: $(BUILD_DIR)/autotuning_bench
warp: $(BUILD_DIR)/warp_tiling_bench
double-buffering: $(BUILD_DIR)/double_buffering_bench
cublas: $(BUILD_DIR)/cublas_bench
device-info: $(BUILD_DIR)/cuda_device_info

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/naive_bench: 1.naive_kernel/My_naive_kernel_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/coalesced_bench: 2.Global_Memory_Coalescing_kernel/My_Global_Memory_Coalescing_kernel_benchmarker.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/smem_bench: 3.SMEM_kernel/My_SMEM_kernel_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/register_1d_bench: 4.1D_Blocktiling_kernel/1D_Blocktiling_kernel_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/register_2d_bench: 5.2D_Blocktiling_kernel/2D_Blocktiling_kernel_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/vectorized_bench: 6.Vectorize_kernel/Vectorize_kernel_benchmark.cu 6.Vectorize_kernel/main_Vectorize_kerenl.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/padding_bench: 7.Shared_Memory_Layout_Optimization/Shared_Memory_Layout_Padding_benchmark.cu 7.Shared_Memory_Layout_Optimization/Shared_Memory_Layout_Padding_kernel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/autotuning_bench: 8.Autoing_kernel/autotune_padding_benchmark.cu 8.Autoing_kernel/Shared_Memory_Layout_Padding_kernel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/warp_tiling_bench: 10.Wraptiling_kernel/Wraptiling_kernel_benchmark.cu 10.Wraptiling_kernel/main_Wraptiling_kernel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/double_buffering_bench: 11.Double_Buffering/Double_Buffering_benchmark.cu 11.Double_Buffering/DoubleBuffering_main_kernel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/cublas_bench: cuBLAS_baseline/cuBLAS_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -lcublas -o $@

$(BUILD_DIR)/cuda_device_info: cudaGetDeviceProperties.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	$(RM) $(TARGETS) $(BUILD_DIR)/cuda_device_info
