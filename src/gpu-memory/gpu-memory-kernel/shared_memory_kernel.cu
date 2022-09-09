// Copyright 2022.
// All rights reserved.
// @author heyanguang
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

#define CUDA_CHECK(condition)                                    \
    /* Code block avoids redefinition of cudaError_t error */    \
    do {                                                         \
        cudaError_t error = condition;                           \
        if (error != cudaSuccess) {                              \
            std::cout << cudaGetErrorString(error) << std::endl; \
        }                                                        \
    } while (0)

#define DIVUP(m, n) (((m) / (n)) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

namespace memory_test {

template <typename T = float>
__global__ void GlobalToDynamicShared(const T *input, T *output) {
    extern __shared__ float shared_memory_pool[];
    T *shared_memory_addr = reinterpret_cast<T*>(shared_memory_pool);

    int32_t block_thread_count = blockDim.x * blockDim.y;
    uint64_t shared_memory_element_count = gridDim.y * block_thread_count;
    uint64_t global_addr = blockIdx.x * shared_memory_element_count + blockIdx.y * block_thread_count
            + threadIdx.y * blockDim.x + threadIdx.x;
    uint64_t shared_addr = blockIdx.y * block_thread_count + threadIdx.y * blockDim.x + threadIdx.x;

    shared_memory_addr[shared_addr] = input[global_addr];
//    shared_memory_addr[shared_addr] = 0.5;

//    T var;
//    var = shared_memory_addr[shared_addr];
//    var = input[global_addr];
//    shared_memory_addr[shared_addr] = var;
}

template <typename T = float>
__global__ void GlobalToGlobal(const T *input, T *output) {
    int32_t block_thread_count = blockDim.x * blockDim.y;

//    uint64_t shared_memory_element_count = gridDim.y * block_thread_count;
//    uint64_t global_addr = blockIdx.x * shared_memory_element_count + blockIdx.y * block_thread_count
//            + threadIdx.y * blockDim.x + threadIdx.x;

    uint64_t shared_memory_element_count = gridDim.x * block_thread_count;
    uint64_t global_addr = blockIdx.y * shared_memory_element_count + blockIdx.x * block_thread_count
            + threadIdx.y * blockDim.x + threadIdx.x;

    output[global_addr] = input[global_addr];
    // T val = input[global_addr];
    // T val = 0.5;
    // output[global_addr] = val;

    // T var;
    // var = input[global_addr];
    // output[global_addr] = var;
}

template <typename T = float>
__global__ void GlobalToGlobalV4(const T *input, T *output) {
    int32_t block_thread_count = blockDim.x * blockDim.y;
    uint64_t shared_memory_element_count = gridDim.x * block_thread_count;
    uint64_t global_addr = blockIdx.y * shared_memory_element_count + blockIdx.x * block_thread_count
        + threadIdx.y * blockDim.x + threadIdx.x;

//    output[global_addr * 4 + 0] = input[global_addr * 4 + 0];
//    output[global_addr * 4 + 1] = input[global_addr * 4 + 1];
//    output[global_addr * 4 + 2] = input[global_addr * 4 + 2];
//    output[global_addr * 4 + 3] = input[global_addr * 4 + 3];

    asm volatile(
    "{\n\t"
    ".reg.f32 a<4>;\n\t"
    ".reg.u64 rd, wr;\n\t"
    "add.u64 rd, %0, %2;\n\t"
    "add.u64 wr, %1, %2;\n\t"
    "ld.global.v4.f32 { a0, a1, a2, a3 }, [rd];\n\t"
    "st.global.v4.f32 [wr], { a0, a1, a2, a3 };\n\t"
    "}"
    :
    : "l"(input), "l"(output), "l"(global_addr * 16)
    : "memory"
    );
}

template <typename T = float>
__global__ void GlobalToDynamicSharedToGlobal(const T *input, const int32_t sm_element_count, T *output) {
    extern __shared__ float shared_memory_pool[];
    // __shared__ float other[32 * 1024];
    T *shared_memory_addr = reinterpret_cast<T*>(shared_memory_pool);

    int32_t block_thread_count = blockDim.x * blockDim.y;
    uint64_t shared_memory_element_count = gridDim.x * block_thread_count;
    uint64_t global_addr = blockIdx.y * shared_memory_element_count + blockIdx.x * block_thread_count
            + threadIdx.y * blockDim.x + threadIdx.x;
    uint64_t shared_addr = blockIdx.x * block_thread_count + threadIdx.y * blockDim.x + threadIdx.x;

    // shared_memory_addr[shared_addr] = input[global_addr];
    // output[global_addr] = shared_memory_addr[shared_addr];

    if (shared_addr < sm_element_count) {
        // T val = input[global_addr];
        // shared_memory_addr[shared_addr] = val;
        // other[shared_addr] = T(1.0) - val;
        // output[global_addr] = other[shared_addr] + shared_memory_addr[shared_addr];

        shared_memory_addr[shared_addr] = input[global_addr];
        output[global_addr] = shared_memory_addr[shared_addr];
    } else {
        output[global_addr] = input[global_addr];
    }
}

template <typename T = float>
__global__ void SharedToShared(const T *input, T *output) {
    extern __shared__ float shared_memory_pool[];
    int32_t block_thread_count = blockDim.x * blockDim.y;
    uint64_t shared_memory_element_count = gridDim.y * block_thread_count;
    T *shared_memory_addr1 = reinterpret_cast<T*>(shared_memory_pool);
    T *shared_memory_addr2 = shared_memory_addr1 + shared_memory_element_count;

    uint64_t shared_addr = blockIdx.y * block_thread_count + threadIdx.y * blockDim.x + threadIdx.x;

    shared_memory_addr2[shared_addr] = shared_memory_addr1[shared_addr];
//    shared_memory_addr2[shared_addr] = 0.5;
}

template <typename T = float>
__global__ void SharedToSharedV4(const T *input, T *output) {
    extern __shared__ float shared_memory_pool[];
    int32_t block_thread_count = blockDim.x * blockDim.y;
    uint64_t shared_memory_element_count = gridDim.y * block_thread_count;
    T *shared_memory_addr1 = reinterpret_cast<T*>(shared_memory_pool);
    T *shared_memory_addr2 = shared_memory_addr1 + shared_memory_element_count;

    uint64_t shared_addr = blockIdx.y * block_thread_count + threadIdx.y * blockDim.x + threadIdx.x;

//    shared_memory_addr2[shared_addr] = shared_memory_addr1[shared_addr];
//    shared_memory_addr2[shared_addr] = 0.5;

//    asm volatile (
//    "{\n\t"
//    ".reg.f32 a<4>;\n\t"
//    ".reg.u32 smem_ptr32_0, smem_ptr32_1, rd, wr;\n\t"
//    ".reg.u64 smem_ptr64_0, smem_ptr64_1;\n\t"
//    "cvta.to.shared.u64 smem_ptr64_0, %0;\n\t"
//    "cvta.to.shared.u64 smem_ptr64_1, %1;\n\t"
//    "cvt.u32.u64 smem_ptr32_0, smem_ptr64_0;\n\t"
//    "cvt.u32.u64 smem_ptr32_1, smem_ptr64_1;\n\t"
//    "add.u32 smem_ptr32_0, smem_ptr32_0, %2;\n\t"
//    "add.u32 smem_ptr32_1, smem_ptr32_1, %2;\n\t"
//    "ld.shared.v4.f32 { a0, a1, a2, a3 }, [smem_ptr32_0];\n\t"
//    "st.shared.v4.f32 [smem_ptr32_1], { a0, a1, a2, a3 };\n\t"
//    "}"
//    :
//    : "l"(shared_memory_addr1), "l"(shared_memory_addr2), "r"(uint32_t(shared_addr * 16))
//    : "memory"
//    );

    asm volatile (
    "{\n\t"
    ".reg.f32 a<4>;\n\t"
    ".reg.u64 smem_ptr64_0, smem_ptr64_1;\n\t"
    "cvta.to.shared.u64 smem_ptr64_0, %0;\n\t"
    "cvta.to.shared.u64 smem_ptr64_1, %1;\n\t"
    "add.u64 smem_ptr64_0, smem_ptr64_0, %2;\n\t"
    "add.u64 smem_ptr64_1, smem_ptr64_1, %2;\n\t"
    "ld.shared.v4.f32 { a0, a1, a2, a3 }, [smem_ptr64_0];\n\t"
    "st.shared.v4.f32 [smem_ptr64_1], { a0, a1, a2, a3 };\n\t"
    "}"
    :
    : "l"(shared_memory_addr1), "l"(shared_memory_addr2), "l"(shared_addr * 16)
    : "memory"
    );

//    printf("addr1=%p, addr2=%p\n", shared_memory_addr1, shared_memory_addr2);
}

namespace functor {

template <typename T>
int32_t LaunchGlobalToDynamicShared(cudaStream_t stream, const T *input, T *output,
        const uint64_t shared_memory_size, const uint64_t cycle_count) {
    int32_t block_count = shared_memory_size / (1024 * sizeof(T));
    dim3 block(32, 32);
    dim3 grid(cycle_count, block_count);

    GlobalToDynamicShared<T><<<grid, block, shared_memory_size, stream>>>(input, output);

    return 1;
}

template <typename T>
int32_t LaunchGlobalToGlobal(cudaStream_t stream, const T *input, T *output,
    const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y) {
    int32_t sm_element_count = shared_memory_size / sizeof(T);
    int32_t grid_x = (sm_element_count + block_size - 1) / block_size;

    dim3 block(1024);
    // dim3 block(32, 32);
    // dim3 grid(grid_x, grid_y);
    dim3 grid(grid_y, grid_x);

    GlobalToGlobal<T><<<grid, block, 0, stream>>>(input, output);

    return 1;
}

template <typename T>
int32_t LaunchGlobalToGlobalV4(cudaStream_t stream, const T *input, T *output,
    const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y) {
    int32_t sm_element_count = shared_memory_size / sizeof(T);
    int32_t grid_x = (sm_element_count + block_size - 1) / block_size;

    dim3 block(1024);
    // dim3 block(32, 32);
    dim3 grid(grid_x, grid_y / 4);

    GlobalToGlobalV4<T><<<grid, block, 0, stream>>>(input, output);

    return 1;
}

template <typename T>
int32_t LaunchGlobalToDynamicSharedToGlobal(cudaStream_t stream, const T *input, T *output,
    const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y) {
    int32_t sm_element_count = shared_memory_size / sizeof(T);
    int32_t grid_x = (sm_element_count + block_size - 1) / block_size;

    dim3 block(32, 32);
    dim3 grid(grid_x, grid_y);

    if (shared_memory_size > 48 * 1024) {
        cudaFuncSetAttribute(GlobalToDynamicSharedToGlobal<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
    }
    GlobalToDynamicSharedToGlobal<T><<<grid, block, shared_memory_size, stream>>>(input, sm_element_count, output);

    return 1;
}

template <typename T>
int32_t LaunchSharedToShared(cudaStream_t stream, const T *input, T *output,
        const uint64_t shared_memory_size, const uint64_t cycle_count) {
    int32_t block_count = shared_memory_size / (2 * 1024 * sizeof(T));
    dim3 block(32, 32);
    dim3 grid(cycle_count, block_count);
//    dim3 grid(block_count, cycle_count);

    SharedToShared<T><<<grid, block, shared_memory_size, stream>>>(input, output);

    return 1;
}

template <typename T>
int32_t LaunchSharedToSharedV4(cudaStream_t stream, const T *input, T *output,
                               const uint64_t shared_memory_size, const uint64_t cycle_count) {
    int32_t block_count = shared_memory_size / (2 * 1024 * sizeof(T) * 4);
    dim3 block(128, 8);
    dim3 grid(cycle_count, block_count);
//    dim3 grid(block_count, cycle_count);

    SharedToSharedV4<T><<<grid, block, shared_memory_size, stream>>>(input, output);

    return 1;
}

template int32_t LaunchGlobalToDynamicShared(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t cycle_count);
template int32_t LaunchGlobalToGlobal(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);
template int32_t LaunchGlobalToGlobalV4(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);
template int32_t LaunchGlobalToDynamicSharedToGlobal(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);
template int32_t LaunchSharedToShared(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t cycle_count);
template int32_t LaunchSharedToSharedV4(cudaStream_t stream, const float *input, float *output,
        const uint64_t shared_memory_size, const uint64_t cycle_count);
}
}
