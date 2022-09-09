#include <stdio.h>
#include <cuda_runtime.h>
#include "common/common.h"

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */
__global__ void helloFromGPU()
{
    if (threadIdx.x == 5)
        printf("Hello World from GPU thread 5!\n");
}

int main(int argc, char **argv) {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();

    cudaError_t error = cudaDeviceReset();
    const char *error_string = cudaGetErrorString(error);
    printf("%s\n", error_string);
//    CHECK(cudaDeviceReset());
    return 0;
}
