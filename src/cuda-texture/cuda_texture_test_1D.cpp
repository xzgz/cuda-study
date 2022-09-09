#include "cuda_runtime.h"
#include <stdio.h>

using namespace std;

typedef texture<float, cudaTextureType1D, cudaReadModeElementType> floatTex;

__global__ void transformKernel(float* output, int element_count);

// texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRef;
// texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRef1(0, cudaFilterModePoint, cudaAddressModeBorder);
texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRef1(0, cudaFilterModeLinear, cudaAddressModeBorder);
texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRef2(0, cudaFilterModePoint, cudaAddressModeBorder);
// floatTex tex1DRef(0, cudaFilterModePoint, cudaAddressModeBorder);
const int cycle_count = 1000;
// const int cycle_count = 1;

__global__ void transformKernel(float* output, int element_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < element_count) {
    //     output[tid] = tex1Dfetch(tex1DRef1, tid);
    //     // output[tid] = tex1D(tex1DRef, tid);
    // }

    output[tid] = tex1Dfetch(tex1DRef1, tid);
    // floatTex tex = threadIdx.x >= 32 ? tex1DRef1 : tex1DRef2;
    // output[tid] = tex1Dfetch(tex, tid);

    // if (threadIdx.x >= 32) {
    //     output[tid] = tex1Dfetch(tex1DRef1, tid);
    // } else {
    //     output[tid] = tex1Dfetch(tex1DRef2, tid);
    // }
}

int main() {
    // int total_element_count = 1024 * 1024 * 1024;
    int total_element_count = 1024 * 1024;
    size_t size = size_t(total_element_count) * sizeof(float);

	float h_data[total_element_count];

    for (int i = total_element_count - 64; i < total_element_count; ++i) {
        h_data[i] = i;
        printf("%f ", h_data[i]);
    }
    printf("\n");

    float* d_data1 = nullptr;
    float* d_data2 = nullptr;
    float* d_output_data = nullptr;
    cudaMalloc((void**)&d_data1, size);
    cudaMalloc((void**)&d_data2, size);
    cudaMalloc((void**)&d_output_data, size);
    cudaMemcpy(d_data1, h_data, size, cudaMemcpyHostToDevice);
    cudaMemset(d_data2, 233333, size);
    cudaMemset(d_output_data, 0, size);

    // tex1DRef.addressMode[0] = cudaAddressModeBorder;
    // tex1DRef.filterMode     = cudaFilterModePoint;
    // tex1DRef.normalized     = 0;
    cudaBindTexture(0, tex1DRef1, d_data1, size);
    cudaBindTexture(0, tex1DRef2, d_data2, size);

    // int block_size = 1024;
    int block_size = 64;
    // dim3 kernel_block(32, 32);
    dim3 kernel_block(block_size);
    dim3 kernel_grid((total_element_count + block_size - 1) / block_size, 1, 1);
    printf("kernel_grid.x=%d, kernel_grid.y=%d, kernel_grid.z=%d\n", kernel_grid.x, kernel_grid.y, kernel_grid.z);
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    for (int i = 0; i < cycle_count; ++i) {
        transformKernel<<<kernel_grid, kernel_block>>>(d_output_data, total_element_count);
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Processing time: %f (ms), %f (GB/s)\n", msecTotal, float(size * 1e+3 * 2 * cycle_count) / float(1024 * 1024 * 1024 * msecTotal));

    cudaMemcpy(h_data, d_output_data, size, cudaMemcpyDeviceToHost);
    for (int i = total_element_count - 64; i < total_element_count; ++i) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_output_data);

    return 0;
}
