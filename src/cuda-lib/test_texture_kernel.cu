#include <cuda_runtime.h>

// texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRef(0, cudaFilterModePoint, cudaAddressModeBorder);


// __global__ void transformKernel(float* output, int element_count) {
//     // int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < element_count) {
//         output[tid] = tex1Dfetch(tex1DRef, tid);
//         // output[tid] = tex1D(tex1DRef, tid);
//     }
// }
