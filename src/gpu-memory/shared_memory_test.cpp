#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace memory_test {
    namespace functor {
        template <typename T>
        int32_t LaunchGlobalToDynamicShared(cudaStream_t stream, const T *input, T *output,
                const uint64_t shared_memory_size, const uint64_t cycle_count);

        template <typename T>
        int32_t LaunchGlobalToGlobal(cudaStream_t stream, const T *input, T *output,
            const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);
        template <typename T>
        int32_t LaunchGlobalToGlobalV4(cudaStream_t stream, const T *input, T *output,
            const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);

        template <typename T>
        int32_t LaunchGlobalToDynamicSharedToGlobal(cudaStream_t stream, const T *input, T *output,
            const uint64_t shared_memory_size, const uint64_t block_size, const uint64_t grid_y);

        template <typename T>
        int32_t LaunchSharedToShared(cudaStream_t stream, const T *input, T *output,
                const uint64_t shared_memory_size, const uint64_t cycle_count);

        template <typename T>
        int32_t LaunchSharedToSharedV4(cudaStream_t stream, const T *input, T *output,
                                       const uint64_t shared_memory_size, const uint64_t cycle_count);
    }
}

using namespace std;

#define CUDA_CHECK(condition)                                    \
/* Code block avoids redefinition of cudaError_t error */    \
do {                                                         \
cudaError_t error = condition;                           \
if (error != cudaSuccess) {                              \
std::cout << cudaGetErrorString(error) << std::endl; \
}                                                        \
} while (0)

void PrepareInput(float *input_ptr, int size) {
    std::random_device rd;
    default_random_engine e(rd());
    float min = -1.0;
    float max = 1.0;
    uniform_real_distribution<float> u(min, max);
    for (int i = 0; i < size; ++i) {
        input_ptr[i] = u(e);
    }
}

void CompareResult(const float *ptr1, const float *ptr2, int size) {
    const float eps = 1e-6;
//    for (int i = 0; i < 100; ++i) {
//        std::cout << "loc: " << i << ", val1: " << ptr1[i] << ", val2: " << ptr2[i]
//                  << ", abs(err): " << abs(ptr1[i] - ptr2[i]) << endl;
//    }
    for (int i = 0; i < size; ++i) {
        if (abs(ptr1[i] - ptr2[i]) > eps) {
            std::cout << "loc: " << i << ", val1: " << ptr1[i] << ", val2: " << ptr2[i]
                      << ", abs(err): " << abs(ptr1[i] - ptr2[i]) << endl;
            std::cout << "Compare not pass.\n";
            return;
        }
    }
    std::cout << "Compare pass.\n";

    return;
}

int main() {
    void *input_dev = nullptr;
    void *output_dev = nullptr;
    uint64_t shared_memory_size = 16 * 1024 * sizeof(float);
    uint64_t sm_element_count = shared_memory_size / sizeof(float);
    uint64_t block_size = 1024;
    uint64_t grid_x = (sm_element_count + block_size - 1) / block_size;
    uint64_t grid_y = 1024 * 32;
    uint64_t input_element_count = block_size * grid_x * grid_y;
    uint64_t input_size = input_element_count * sizeof(float);
    uint64_t output_size = input_size;
    uint64_t input_size_in_mb = input_size / 1024 / 1024;

    float *input_host = new float[input_size / sizeof(float)];
    float *output_host = new float[output_size / sizeof(float)];
    CUDA_CHECK(cudaMalloc(&input_dev, input_size));
    CUDA_CHECK(cudaMalloc(&output_dev, output_size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    PrepareInput(input_host, input_size / sizeof(float));
    cudaMemcpy(input_dev, input_host, input_size, cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMemset(output_dev, 0, output_size));
    memory_test::functor::LaunchGlobalToGlobal<float>(stream, static_cast<float*>(input_dev),
        static_cast<float*>(output_dev), shared_memory_size, block_size, grid_y);
    cudaMemcpy((void *)output_host, output_dev, output_size, cudaMemcpyDeviceToHost);
    CompareResult(input_host, output_host, output_size / sizeof(float));

    int ite = 100;

    {
        printf("LaunchGlobalToGlobalV4:\n");
        cudaEvent_t estart;
        cudaEvent_t estop;
        float usems;
        cudaEventCreate(&estart);
        cudaEventRecord(estart, stream);
        for (int i = 0; i < ite; ++i) {
            memory_test::functor::LaunchGlobalToGlobalV4<float>(stream, static_cast<float*>(input_dev),
                static_cast<float*>(output_dev), shared_memory_size, block_size, grid_y);
            // cudaMemcpy(output_dev, input_dev, input_size, cudaMemcpyDeviceToDevice);
        }
        cudaEventCreate(&estop);
        cudaEventRecord(estop, stream);
        cudaEventSynchronize(estop);
        cudaEventElapsedTime(&usems, estart, estop);

        double gb_per_sec = 1000 / double(usems / ite) * 2 * input_size_in_mb / 1024;
        printf("%.8f ms ( %d iterations)\n", usems / ite, ite);
        printf("%.1f GB/s\n", gb_per_sec);
    }

    {
        printf("LaunchGlobalToGlobal:\n");
        cudaEvent_t estart;
        cudaEvent_t estop;
        float usems;
        cudaEventCreate(&estart);
        cudaEventRecord(estart, stream);
        for (int i = 0; i < ite; ++i) {
            memory_test::functor::LaunchGlobalToGlobal<float>(stream, static_cast<float*>(input_dev),
                static_cast<float*>(output_dev), shared_memory_size, block_size, grid_y);
        }
        cudaEventCreate(&estop);
        cudaEventRecord(estop, stream);
        cudaEventSynchronize(estop);
        cudaEventElapsedTime(&usems, estart, estop);

        double gb_per_sec = 1000 / double(usems / ite) * 2 * input_size_in_mb / 1024;
        printf("%.8f ms ( %d iterations)\n", usems / ite, ite);
        printf("%.1f GB/s\n", gb_per_sec);
    }

    delete [] input_host;
    delete [] output_host;
    cudaFree(input_dev);
    cudaFree(output_dev);

    return 0;
}
