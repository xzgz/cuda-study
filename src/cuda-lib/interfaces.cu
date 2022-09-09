#include "matmul_kernel.h"
#include "prof.h"
#include "openblas/cblas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <iostream>

using namespace std;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void cuda_kernel_sgemm_100_tex(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta);

typedef texture<float, cudaTextureType1D, cudaReadModeElementType> floatTex;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRefA(0, cudaFilterModePoint, cudaAddressModeBorder);
texture<float, cudaTextureType1D, cudaReadModeElementType> tex1DRefB(0, cudaFilterModePoint, cudaAddressModeBorder);
#define     USE_TEXTURE     1


void gpu_sgemm(
        float *a, float *b, float *c,
        size_t N, size_t M, size_t K,
        float alpha, float beta, int kernel_type) {
    float *dev_a = NULL;
    float *dev_at = NULL;
    float *dev_b = NULL;
    float *dev_c = NULL;
    half *A = NULL;
    half *B = NULL;
    half *B_ht = NULL;
    float *C = NULL;
    float *D = NULL;
    float flop = 2 * (float)M * (float)N * (float)K;
    cublasHandle_t handle;

    int lda = K;
    int ldb = N;
    int ldc = N;
//    int lda = M;
//    int ldb = K;
//    int ldc = M;

    if (kernel_type == 'b') cublasCreate(&handle);
    if (kernel_type == 't') {
        half *B_h = (half*)b;
        B_ht = (half*)malloc(sizeof(half) * K * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < K; ++j) {
                B_ht[i * K + j] = B_h[j * N + i];
            }
        }

        checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M * K));
        checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N * K));
        checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M * N));
        checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M * N));
//        assert((int(A)) % 128 == 0);
        assert(((unsigned long long)A) % 128 == 0);
        assert(((unsigned long long)B) % 128 == 0);
        assert(((unsigned long long)C) % 128 == 0);
        assert(((unsigned long long)D) % 128 == 0);
        checkCudaErrors(cudaMemcpy(A, a, sizeof(half) * M * K, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(B, B_ht, sizeof(half) * N * K, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(C, c, sizeof(float) * M * N, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M * N));
    }

    float* at = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            at[j * M + i] = a[i * K + j];
        }
    }
    cudaMalloc((void **)&dev_a, M * K * sizeof(float));
    cudaMalloc((void **)&dev_at, M * K * sizeof(float));
    cudaMalloc((void **)&dev_b, K * N * sizeof(float));
    cudaMalloc((void **)&dev_c, M * N * sizeof(float));
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_at, at, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(0, tex1DRefA, dev_at, M * K * sizeof(float));
    cudaBindTexture(0, tex1DRefB, dev_b, K * N * sizeof(float));
    int cycle_count = 100;

//    hs_timer timer;
//    timer.tic("gpu sgemm");

    cudaError_t result;
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    switch (kernel_type)
    {
        case 0:
        {
            int grid_r = M / 32;
            int grid_c = N / 32;
            if (M % 32 != 0)
                grid_r += 1;
            if (N % 32 != 0)
                grid_c += 1;
            dim3 grid_d(grid_r, grid_c, 1);
            dim3 block_d(32, 32, 1);
            cuda_kernel_sgemm_0<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
            break;
        }
        case 1:
        {
            int grid_r = M / 32;
            int grid_c = N / 32;
            if (M % 32 != 0)
                grid_r += 1;
            if (N % 32 != 0)
                grid_c += 1;
            dim3 grid_d(grid_r, grid_c, 1);
            dim3 block_d(32, 32, 1);
            cuda_kernel_sgemm_1<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
            break;
        }
        case 2:
        {
            int grid_r = M / 32;
            int grid_c = N / 32;
            if (M % 32 != 0)
                grid_r += 1;
            if (N % 32 != 0)
                grid_c += 1;
            dim3 grid_d(grid_r, grid_c, 1);
            dim3 block_d(32, 32, 1);
            for (int n = 0; n < cycle_count; ++n) {
                cuda_kernel_sgemm_2<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
            }
            break;
        }
        case 20:
        {
            int grid_r = M / 64;
            int grid_c = N / 64;
            if (M % 64 != 0)
                grid_r += 1;
            if (N % 64 != 0)
                grid_c += 1;
            dim3 grid_d(grid_r, grid_c, 1);
            dim3 block_d(32, 32, 1);
            for (int n = 0; n < cycle_count; ++n) {
                cuda_kernel_sgemm_2_64x64<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
            }
            break;
        }
        case 'b':
        {
            for (int n = 0; n < cycle_count; ++n) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_b, N, dev_a, K, &beta, dev_c, N);
            }
            break;
        }
        case 'c':
        {
            for (int n = 0; n < cycle_count; ++n) {
                result = CutlassSgemmNN(M, N, K, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc);
            }
            if (result == cudaSuccess) {
                cout << "CutlassSgemmNN success" << endl;
            }
            break;
        }
        case 'r':
        {
            cudaError_t result;
            result = ReferenceGemm(M, N, K, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc);
            if (result == cudaSuccess) {
                cout << "ReferenceGemm success" << endl;
            }
            break;
        }
        case 't':
        {
            dim3 gridDim;
            dim3 blockDim;

            // blockDim.x must be a multple of warpSize
            // 128x4 means we have 16 warps and a block computes a 64x64 output tile
            blockDim.x = 128;
            blockDim.y = 4;

            gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
            gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

            wmma_sgemm_kernel<<<gridDim, blockDim>>>(A, B, C, D, M, N, K, alpha, beta);
        }
        case 100:
        {
            int stride_x = 64;
            int stride_y = 64;
            int grid_x = (N + stride_x - 1) / stride_x;
            int grid_y = (M + stride_y - 1) / stride_y;
            int block_x = stride_x;
            dim3 grid_d(grid_x, grid_y, 1);
            dim3 block_d(block_x, 1, 1);
            std::cout << grid_x << " " << grid_y << " " << block_x << std::endl;
            for (int n = 0; n < cycle_count; ++n) {
                cuda_kernel_sgemm_100<<<grid_d, block_d>>>(dev_at, dev_b, dev_c, M, N, K, alpha, beta);
                // cuda_kernel_sgemm_100_tex<<<grid_d, block_d>>>(dev_at, dev_b, dev_c, M, N, K, alpha, beta);
                // cuda_kernel_sgemm_100_v2<<<grid_d, block_d>>>(dev_at, dev_b, dev_c, M, N, K, alpha, beta);
            }
            break;
        }
    }
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
//    cudaDeviceSynchronize();
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, cycle_count * flop / msecTotal/ 1e+6);
//    timer.toc("gpu sgemm");

    float* ct = (float*)malloc(M * N * sizeof(float));
    if (kernel_type == 't') {
        cudaMemcpy(c, D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        if (kernel_type == 100) {
            cudaMemcpy(ct, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    c[j * N + i] = ct[i * M + j];
                }
            }
        } else {
            cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    if (kernel_type == 'b') cublasDestroy(handle);
    if (kernel_type == 't') {
        free(B_ht);
        checkCudaErrors(cudaFree((void*)A));
        checkCudaErrors(cudaFree((void*)B));
        checkCudaErrors(cudaFree((void*)C));
        checkCudaErrors(cudaFree((void*)D));
    }

    free(at);
    free(ct);
    cudaFree(dev_a);
    cudaFree(dev_at);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void gpu_warmup()
{
    float *dev_p = 0;

    hs_timer timer;
    timer.tic("gpu warmup");

    cudaMalloc((void **)&dev_p, 16 * 32 * sizeof(float));

    cuda_kernel_warmup<<<16, 32>>>(dev_p);

    cudaDeviceSynchronize();

    cudaFree(dev_p);

    timer.toc("gpu warmup");
}

//void cpu_kernel_sgemm_0(float *a, float *b, float *c, size_t N, size_t M, size_t K, float alpha, float beta) {
//    for (int m = 0; m < M; ++m) {
//        for (int n = 0; n < N; ++n) {
//            float acc = 0.0f;
//            for (int k = 0; k < K; ++k) {
//                acc += a[m * K + k] * b[k * N + n];
//            }
//            c[m * N + n] = alpha * acc + beta * c[m * N + n];
//        }
//    }
//}

void cpu_sgemm(
        float *a, float *b, float *c,
        size_t N, size_t M, size_t K,
        float alpha, float beta, int kernel_type)
{
    hs_timer timer;
    timer.tic("cpu sgemm");

    switch (kernel_type)
    {
        case 0:
        {
            cpu_kernel_sgemm_0(a, b, c, N, M, K, alpha, beta);
            break;
        }
        case 'm':
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, K, b, N, beta, c, N);
            break;
        }
    }
    timer.toc("cpu sgemm");
}

void cpu_warmup() {
    hs_timer timer;
    timer.tic("cpu warmup");

    const size_t arr_size = 1024;
    float *p = new float[arr_size];

#pragma omp parallel for simd
    for (size_t i = 0; i < arr_size; i++)
    {
        float f = (float)i;
        p[i] = f * f * f;
    }

    delete p;

    timer.toc("cpu warmup");
}


__device__ void sgemm_block_64x64_tex(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta) {

    __shared__ float a_b_shm[2 * 16 * 64];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    // float* a = pa + block_y * 64;
    // float* b = pb + block_x * 64;
    // float* c = pc + block_x * 64 * M + block_y * 64;

    // int tid = threadIdx.x & 0x3f;
    int tid = threadIdx.x;
    int ldx = tid >= 32 ? N : M;
    int tid2 = (tid >> 4) & 1;
    // int tid15 = tid & 0xf;
    int tid15 = tid & 15;
    int a_b_offset = tid >= 32 ? block_x * 64 : block_y * 64;
    int track0 = a_b_offset + tid2 * ldx + tid15 * 4;
    int track2 = track0 + 2 * ldx;
    int track4 = track0 + 4 * ldx;
    int track6 = track0 + 6 * ldx;
    int end = track0 + (K - 8) * ldx;
    int write_offset = tid2 * 64 + tid15 * 4;
    write_offset += tid >= 32 ? 512 : 0;

    int readAs = ((tid >> 1) & 7) << 2;
    int readBs = ((((tid & 0x30) >> 3) | (tid & 1)) << 2) + 512;

#if USE_TEXTURE
    floatTex tex = tid >= 32 ? tex1DRefB : tex1DRefA;
#else
    float* read_addr = tid >= 32 ? b : a;
#endif

    float cbb00=0, cbb01=0, cbb02=0, cbb03=0;
    float cbb10=0, cbb11=0, cbb12=0, cbb13=0;
    float cbb20=0, cbb21=0, cbb22=0, cbb23=0;
    float cbb30=0, cbb31=0, cbb32=0, cbb33=0;
    float cba00=0, cba01=0, cba02=0, cba03=0;
    float cba10=0, cba11=0, cba12=0, cba13=0;
    float cba20=0, cba21=0, cba22=0, cba23=0;
    float cba30=0, cba31=0, cba32=0, cba33=0;
    float cab00=0, cab01=0, cab02=0, cab03=0;
    float cab10=0, cab11=0, cab12=0, cab13=0;
    float cab20=0, cab21=0, cab22=0, cab23=0;
    float cab30=0, cab31=0, cab32=0, cab33=0;
    float caa00=0, caa01=0, caa02=0, caa03=0;
    float caa10=0, caa11=0, caa12=0, caa13=0;
    float caa20=0, caa21=0, caa22=0, caa23=0;
    float caa30=0, caa31=0, caa32=0, caa33=0;

    // float cbb00, cbb01, cbb02, cbb03;
    // float cbb10, cbb11, cbb12, cbb13;
    // float cbb20, cbb21, cbb22, cbb23;
    // float cbb30, cbb31, cbb32, cbb33;
    // float cba00, cba01, cba02, cba03;
    // float cba10, cba11, cba12, cba13;
    // float cba20, cba21, cba22, cba23;
    // float cba30, cba31, cba32, cba33;
    // float cab00, cab01, cab02, cab03;
    // float cab10, cab11, cab12, cab13;
    // float cab20, cab21, cab22, cab23;
    // float cab30, cab31, cab32, cab33;
    // float caa00, caa01, caa02, caa03;
    // float caa10, caa11, caa12, caa13;
    // float caa20, caa21, caa22, caa23;
    // float caa30, caa31, caa32, caa33;

    float j0Ab00, j0Ab01, j0Ab02, j0Ab03;
    float j0Bb00, j0Bb01, j0Bb02, j0Bb03;
    float j0Aa00, j0Aa01, j0Aa02, j0Aa03;
    float j0Ba00, j0Ba01, j0Ba02, j0Ba03;
    // float j1Ab00, j1Ab01, j1Ab02, j1Ab03;
    // float j1Bb00, j1Bb01, j1Bb02, j1Bb03;
    // float j1Aa00, j1Aa01, j1Aa02, j1Aa03;
    // float j1Ba00, j1Ba01, j1Ba02, j1Ba03;

    // float j0Ab00=1, j0Ab01=1, j0Ab02=1, j0Ab03=1;
    // float j0Bb00=1, j0Bb01=1, j0Bb02=1, j0Bb03=1;
    // float j0Aa00=1, j0Aa01=1, j0Aa02=1, j0Aa03=1;
    // float j0Ba00=1, j0Ba01=1, j0Ba02=1, j0Ba03=1;
    // float j1Ab00=1, j1Ab01=1, j1Ab02=1, j1Ab03=1;
    // float j1Bb00=1, j1Bb01=1, j1Bb02=1, j1Bb03=1;
    // float j1Aa00=1, j1Aa01=1, j1Aa02=1, j1Aa03=1;
    // float j1Ba00=1, j1Ba01=1, j1Ba02=1, j1Ba03=1;

    while (track0 <= end) {
#if USE_TEXTURE
        a_b_shm[write_offset + 0 * 64 + 0] = tex1Dfetch(tex, track0 + 0);
        a_b_shm[write_offset + 0 * 64 + 1] = tex1Dfetch(tex, track0 + 1);
        a_b_shm[write_offset + 0 * 64 + 2] = tex1Dfetch(tex, track0 + 2);
        a_b_shm[write_offset + 0 * 64 + 3] = tex1Dfetch(tex, track0 + 3);
        a_b_shm[write_offset + 2 * 64 + 0] = tex1Dfetch(tex, track2 + 0);
        a_b_shm[write_offset + 2 * 64 + 1] = tex1Dfetch(tex, track2 + 1);
        a_b_shm[write_offset + 2 * 64 + 2] = tex1Dfetch(tex, track2 + 2);
        a_b_shm[write_offset + 2 * 64 + 3] = tex1Dfetch(tex, track2 + 3);
        a_b_shm[write_offset + 4 * 64 + 0] = tex1Dfetch(tex, track4 + 0);
        a_b_shm[write_offset + 4 * 64 + 1] = tex1Dfetch(tex, track4 + 1);
        a_b_shm[write_offset + 4 * 64 + 2] = tex1Dfetch(tex, track4 + 2);
        a_b_shm[write_offset + 4 * 64 + 3] = tex1Dfetch(tex, track4 + 3);
        a_b_shm[write_offset + 6 * 64 + 0] = tex1Dfetch(tex, track6 + 0);
        a_b_shm[write_offset + 6 * 64 + 1] = tex1Dfetch(tex, track6 + 1);
        a_b_shm[write_offset + 6 * 64 + 2] = tex1Dfetch(tex, track6 + 2);
        a_b_shm[write_offset + 6 * 64 + 3] = tex1Dfetch(tex, track6 + 3);
#else
        a_b_shm[write_offset + 0 * 64 + 0] = read_addr[track0 + 0];
        a_b_shm[write_offset + 0 * 64 + 1] = read_addr[track0 + 1];
        a_b_shm[write_offset + 0 * 64 + 2] = read_addr[track0 + 2];
        a_b_shm[write_offset + 0 * 64 + 3] = read_addr[track0 + 3];
        a_b_shm[write_offset + 2 * 64 + 0] = read_addr[track2 + 0];
        a_b_shm[write_offset + 2 * 64 + 1] = read_addr[track2 + 1];
        a_b_shm[write_offset + 2 * 64 + 2] = read_addr[track2 + 2];
        a_b_shm[write_offset + 2 * 64 + 3] = read_addr[track2 + 3];
        a_b_shm[write_offset + 4 * 64 + 0] = read_addr[track4 + 0];
        a_b_shm[write_offset + 4 * 64 + 1] = read_addr[track4 + 1];
        a_b_shm[write_offset + 4 * 64 + 2] = read_addr[track4 + 2];
        a_b_shm[write_offset + 4 * 64 + 3] = read_addr[track4 + 3];
        a_b_shm[write_offset + 6 * 64 + 0] = read_addr[track6 + 0];
        a_b_shm[write_offset + 6 * 64 + 1] = read_addr[track6 + 1];
        a_b_shm[write_offset + 6 * 64 + 2] = read_addr[track6 + 2];
        a_b_shm[write_offset + 6 * 64 + 3] = read_addr[track6 + 3];
#endif
        __syncthreads();
        // __syncwarp(0xFFFFFFFF);

        write_offset ^= 16 * 64;
        track0 += 8 * ldx;
        track2 += 8 * ldx;
        track4 += 8 * ldx;
        track6 += 8 * ldx;

        for (int j = 0; j < 8; ++j) {
            // int prefetch = (j + 1) % 8;
            int prefetch = j;

            j0Ab00 = a_b_shm[readAs + prefetch * 64 + 0];
            j0Ab01 = a_b_shm[readAs + prefetch * 64 + 1];
            j0Ab02 = a_b_shm[readAs + prefetch * 64 + 2];
            j0Ab03 = a_b_shm[readAs + prefetch * 64 + 3];

            j0Bb00 = a_b_shm[readBs + prefetch * 64 + 0];
            j0Bb01 = a_b_shm[readBs + prefetch * 64 + 1];
            j0Bb02 = a_b_shm[readBs + prefetch * 64 + 2];
            j0Bb03 = a_b_shm[readBs + prefetch * 64 + 3];

            j0Aa00 = a_b_shm[readAs + prefetch * 64 + 32 + 0];
            j0Aa01 = a_b_shm[readAs + prefetch * 64 + 32 + 1];
            j0Aa02 = a_b_shm[readAs + prefetch * 64 + 32 + 2];
            j0Aa03 = a_b_shm[readAs + prefetch * 64 + 32 + 3];

            j0Ba00 = a_b_shm[readBs + prefetch * 64 + 32 + 0];
            j0Ba01 = a_b_shm[readBs + prefetch * 64 + 32 + 1];
            j0Ba02 = a_b_shm[readBs + prefetch * 64 + 32 + 2];
            j0Ba03 = a_b_shm[readBs + prefetch * 64 + 32 + 3];

            cbb00 += j0Bb00 * j0Ab00;
            cbb01 += j0Bb00 * j0Ab01;
            // j1Ab00 = a_b_shm[readAs + prefetch * 64 + 0];
            // j1Ab01 = a_b_shm[readAs + prefetch * 64 + 1];
            // j1Ab02 = a_b_shm[readAs + prefetch * 64 + 2];
            // j1Ab03 = a_b_shm[readAs + prefetch * 64 + 3];
            cbb02 += j0Bb00 * j0Ab02;
            cbb03 += j0Bb00 * j0Ab03;
            // j1Bb00 = a_b_shm[readBs + prefetch * 64 + 0];
            // j1Bb01 = a_b_shm[readBs + prefetch * 64 + 1];
            // j1Bb02 = a_b_shm[readBs + prefetch * 64 + 2];
            // j1Bb03 = a_b_shm[readBs + prefetch * 64 + 3];

            cbb10 += j0Bb01 * j0Ab00;
            cbb11 += j0Bb01 * j0Ab01;
            // j1Aa00 = a_b_shm[readAs + prefetch * 64 + 32 + 0];
            // j1Aa01 = a_b_shm[readAs + prefetch * 64 + 32 + 1];
            // j1Aa02 = a_b_shm[readAs + prefetch * 64 + 32 + 2];
            // j1Aa03 = a_b_shm[readAs + prefetch * 64 + 32 + 3];
            cbb12 += j0Bb01 * j0Ab02;
            cbb13 += j0Bb01 * j0Ab03;
            // j1Ba00 = a_b_shm[readBs + prefetch * 64 + 32 + 0];
            // j1Ba01 = a_b_shm[readBs + prefetch * 64 + 32 + 1];
            // j1Ba02 = a_b_shm[readBs + prefetch * 64 + 32 + 2];
            // j1Ba03 = a_b_shm[readBs + prefetch * 64 + 32 + 3];

            cbb20 += j0Bb02 * j0Ab00;
            cbb21 += j0Bb02 * j0Ab01;
            cbb22 += j0Bb02 * j0Ab02;
            cbb23 += j0Bb02 * j0Ab03;

            cbb30 += j0Bb03 * j0Ab00;
            cbb31 += j0Bb03 * j0Ab01;
            cbb32 += j0Bb03 * j0Ab02;
            cbb33 += j0Bb03 * j0Ab03;

            cba00 += j0Ba00 * j0Ab00;
            cba01 += j0Ba00 * j0Ab01;
            cba02 += j0Ba00 * j0Ab02;
            cba03 += j0Ba00 * j0Ab03;

            cba10 += j0Ba01 * j0Ab00;
            cba11 += j0Ba01 * j0Ab01;
            cba12 += j0Ba01 * j0Ab02;
            cba13 += j0Ba01 * j0Ab03;

            cba20 += j0Ba02 * j0Ab00;
            cba21 += j0Ba02 * j0Ab01;
            cba22 += j0Ba02 * j0Ab02;
            cba23 += j0Ba02 * j0Ab03;

            cba30 += j0Ba03 * j0Ab00;
            cba31 += j0Ba03 * j0Ab01;
            cba32 += j0Ba03 * j0Ab02;
            cba33 += j0Ba03 * j0Ab03;

            cab00 += j0Bb00 * j0Aa00;
            cab01 += j0Bb00 * j0Aa01;
            cab02 += j0Bb00 * j0Aa02;
            cab03 += j0Bb00 * j0Aa03;

            cab10 += j0Bb01 * j0Aa00;
            cab11 += j0Bb01 * j0Aa01;
            cab12 += j0Bb01 * j0Aa02;
            cab13 += j0Bb01 * j0Aa03;

            cab20 += j0Bb02 * j0Aa00;
            cab21 += j0Bb02 * j0Aa01;
            cab22 += j0Bb02 * j0Aa02;
            cab23 += j0Bb02 * j0Aa03;

            cab30 += j0Bb03 * j0Aa00;
            cab31 += j0Bb03 * j0Aa01;
            cab32 += j0Bb03 * j0Aa02;
            cab33 += j0Bb03 * j0Aa03;

            caa00 += j0Ba00 * j0Aa00;
            caa01 += j0Ba00 * j0Aa01;
            caa02 += j0Ba00 * j0Aa02;
            caa03 += j0Ba00 * j0Aa03;

            caa10 += j0Ba01 * j0Aa00;
            caa11 += j0Ba01 * j0Aa01;
            caa12 += j0Ba01 * j0Aa02;
            caa13 += j0Ba01 * j0Aa03;

            caa20 += j0Ba02 * j0Aa00;
            caa21 += j0Ba02 * j0Aa01;
            caa22 += j0Ba02 * j0Aa02;
            caa23 += j0Ba02 * j0Aa03;

            caa30 += j0Ba03 * j0Aa00;
            caa31 += j0Ba03 * j0Aa01;
            caa32 += j0Ba03 * j0Aa02;
            caa33 += j0Ba03 * j0Aa03;
        }

        readAs ^= 16 * 64;
        readBs ^= 16 * 64;
    }
    __syncthreads();


    int tid31 = tid & 31;
    int tid32 = tid & 32;
    int coord_x = readBs & 0x7f;
    int coord_y = readAs & 0x7f;
    int writeCs = coord_x / 4 * 64 + coord_y;
    int readCs = (tid32 << 3) + tid31;
    int ldc4 = M * 4;
    int Cy00 = block_x * 64 * M + block_y * 64 + (tid32 >> 1) * M + tid31;
    int Cy04 = Cy00 + ldc4;
    int Cy08 = Cy00 + 2 * ldc4;
    int Cy12 = Cy00 + 3 * ldc4;

    a_b_shm[writeCs + 0] = cbb00;
    a_b_shm[writeCs + 1] = cbb01;
    a_b_shm[writeCs + 2] = cbb02;
    a_b_shm[writeCs + 3] = cbb03;
    a_b_shm[writeCs + 32 + 0] = cab00;
    a_b_shm[writeCs + 32 + 1] = cab01;
    a_b_shm[writeCs + 32 + 2] = cab02;
    a_b_shm[writeCs + 32 + 3] = cab03;
    // if (threadIdx.x == 1) {
    //     printf("reg r0,  c4: %f\n", cbb00);
    // }
    // if (threadIdx.x == 18) {
    //     printf("reg r7,  c8: %f\n", cbb03);
    //     printf("reg r39, c8: %f\n", cab03);
    // }

    cbb00 = a_b_shm[readCs + 0 * 64 + 0 ];
    cbb01 = a_b_shm[readCs + 0 * 64 + 32];
    cbb02 = a_b_shm[readCs + 1 * 64 + 0 ];
    cbb03 = a_b_shm[readCs + 1 * 64 + 32];
    cab00 = a_b_shm[readCs + 2 * 64 + 0 ];
    cab01 = a_b_shm[readCs + 2 * 64 + 32];
    cab02 = a_b_shm[readCs + 3 * 64 + 0 ];
    cab03 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cbb00;
    c[Cy00 + 32] = cbb01;
    c[Cy04 + 0 ] = cbb02;
    c[Cy04 + 32] = cbb03;
    c[Cy08 + 0 ] = cab00;
    c[Cy08 + 32] = cab01;
    c[Cy12 + 0 ] = cab02;
    c[Cy12 + 32] = cab03;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cbb10;
    a_b_shm[writeCs + 1] = cbb11;
    a_b_shm[writeCs + 2] = cbb12;
    a_b_shm[writeCs + 3] = cbb13;
    a_b_shm[writeCs + 32 + 0] = cab10;
    a_b_shm[writeCs + 32 + 1] = cab11;
    a_b_shm[writeCs + 32 + 2] = cab12;
    a_b_shm[writeCs + 32 + 3] = cab13;

    cbb10 = a_b_shm[readCs + 0 * 64 + 0 ];
    cbb11 = a_b_shm[readCs + 0 * 64 + 32];
    cbb12 = a_b_shm[readCs + 1 * 64 + 0 ];
    cbb13 = a_b_shm[readCs + 1 * 64 + 32];
    cab10 = a_b_shm[readCs + 2 * 64 + 0 ];
    cab11 = a_b_shm[readCs + 2 * 64 + 32];
    cab12 = a_b_shm[readCs + 3 * 64 + 0 ];
    cab13 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cbb10;
    c[Cy00 + 32] = cbb11;
    c[Cy04 + 0 ] = cbb12;
    c[Cy04 + 32] = cbb13;
    c[Cy08 + 0 ] = cab10;
    c[Cy08 + 32] = cab11;
    c[Cy12 + 0 ] = cab12;
    c[Cy12 + 32] = cab13;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cbb20;
    a_b_shm[writeCs + 1] = cbb21;
    a_b_shm[writeCs + 2] = cbb22;
    a_b_shm[writeCs + 3] = cbb23;
    a_b_shm[writeCs + 32 + 0] = cab20;
    a_b_shm[writeCs + 32 + 1] = cab21;
    a_b_shm[writeCs + 32 + 2] = cab22;
    a_b_shm[writeCs + 32 + 3] = cab23;

    cbb20 = a_b_shm[readCs + 0 * 64 + 0 ];
    cbb21 = a_b_shm[readCs + 0 * 64 + 32];
    cbb22 = a_b_shm[readCs + 1 * 64 + 0 ];
    cbb23 = a_b_shm[readCs + 1 * 64 + 32];
    cab20 = a_b_shm[readCs + 2 * 64 + 0 ];
    cab21 = a_b_shm[readCs + 2 * 64 + 32];
    cab22 = a_b_shm[readCs + 3 * 64 + 0 ];
    cab23 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cbb20;
    c[Cy00 + 32] = cbb21;
    c[Cy04 + 0 ] = cbb22;
    c[Cy04 + 32] = cbb23;
    c[Cy08 + 0 ] = cab20;
    c[Cy08 + 32] = cab21;
    c[Cy12 + 0 ] = cab22;
    c[Cy12 + 32] = cab23;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cbb30;
    a_b_shm[writeCs + 1] = cbb31;
    a_b_shm[writeCs + 2] = cbb32;
    a_b_shm[writeCs + 3] = cbb33;
    a_b_shm[writeCs + 32 + 0] = cab30;
    a_b_shm[writeCs + 32 + 1] = cab31;
    a_b_shm[writeCs + 32 + 2] = cab32;
    a_b_shm[writeCs + 32 + 3] = cab33;

    cbb30 = a_b_shm[readCs + 0 * 64 + 0 ];
    cbb31 = a_b_shm[readCs + 0 * 64 + 32];
    cbb32 = a_b_shm[readCs + 1 * 64 + 0 ];
    cbb33 = a_b_shm[readCs + 1 * 64 + 32];
    cab30 = a_b_shm[readCs + 2 * 64 + 0 ];
    cab31 = a_b_shm[readCs + 2 * 64 + 32];
    cab32 = a_b_shm[readCs + 3 * 64 + 0 ];
    cab33 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cbb30;
    c[Cy00 + 32] = cbb31;
    c[Cy04 + 0 ] = cbb32;
    c[Cy04 + 32] = cbb33;
    c[Cy08 + 0 ] = cab30;
    c[Cy08 + 32] = cab31;
    c[Cy12 + 0 ] = cab32;
    c[Cy12 + 32] = cab33;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;

    Cy00 += 28 * M;
    Cy04 += 28 * M;
    Cy08 += 28 * M;
    Cy12 += 28 * M;

    a_b_shm[writeCs + 0] = cba00;
    a_b_shm[writeCs + 1] = cba01;
    a_b_shm[writeCs + 2] = cba02;
    a_b_shm[writeCs + 3] = cba03;
    a_b_shm[writeCs + 32 + 0] = caa00;
    a_b_shm[writeCs + 32 + 1] = caa01;
    a_b_shm[writeCs + 32 + 2] = caa02;
    a_b_shm[writeCs + 32 + 3] = caa03;

    cba00 = a_b_shm[readCs + 0 * 64 + 0 ];
    cba01 = a_b_shm[readCs + 0 * 64 + 32];
    cba02 = a_b_shm[readCs + 1 * 64 + 0 ];
    cba03 = a_b_shm[readCs + 1 * 64 + 32];
    caa00 = a_b_shm[readCs + 2 * 64 + 0 ];
    caa01 = a_b_shm[readCs + 2 * 64 + 32];
    caa02 = a_b_shm[readCs + 3 * 64 + 0 ];
    caa03 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cba00;
    c[Cy00 + 32] = cba01;
    c[Cy04 + 0 ] = cba02;
    c[Cy04 + 32] = cba03;
    c[Cy08 + 0 ] = caa00;
    c[Cy08 + 32] = caa01;
    c[Cy12 + 0 ] = caa02;
    c[Cy12 + 32] = caa03;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cba10;
    a_b_shm[writeCs + 1] = cba11;
    a_b_shm[writeCs + 2] = cba12;
    a_b_shm[writeCs + 3] = cba13;
    a_b_shm[writeCs + 32 + 0] = caa10;
    a_b_shm[writeCs + 32 + 1] = caa11;
    a_b_shm[writeCs + 32 + 2] = caa12;
    a_b_shm[writeCs + 32 + 3] = caa13;

    cba10 = a_b_shm[readCs + 0 * 64 + 0 ];
    cba11 = a_b_shm[readCs + 0 * 64 + 32];
    cba12 = a_b_shm[readCs + 1 * 64 + 0 ];
    cba13 = a_b_shm[readCs + 1 * 64 + 32];
    caa10 = a_b_shm[readCs + 2 * 64 + 0 ];
    caa11 = a_b_shm[readCs + 2 * 64 + 32];
    caa12 = a_b_shm[readCs + 3 * 64 + 0 ];
    caa13 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cba10;
    c[Cy00 + 32] = cba11;
    c[Cy04 + 0 ] = cba12;
    c[Cy04 + 32] = cba13;
    c[Cy08 + 0 ] = caa10;
    c[Cy08 + 32] = caa11;
    c[Cy12 + 0 ] = caa12;
    c[Cy12 + 32] = caa13;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cba20;
    a_b_shm[writeCs + 1] = cba21;
    a_b_shm[writeCs + 2] = cba22;
    a_b_shm[writeCs + 3] = cba23;
    a_b_shm[writeCs + 32 + 0] = caa20;
    a_b_shm[writeCs + 32 + 1] = caa21;
    a_b_shm[writeCs + 32 + 2] = caa22;
    a_b_shm[writeCs + 32 + 3] = caa23;

    cba20 = a_b_shm[readCs + 0 * 64 + 0 ];
    cba21 = a_b_shm[readCs + 0 * 64 + 32];
    cba22 = a_b_shm[readCs + 1 * 64 + 0 ];
    cba23 = a_b_shm[readCs + 1 * 64 + 32];
    caa20 = a_b_shm[readCs + 2 * 64 + 0 ];
    caa21 = a_b_shm[readCs + 2 * 64 + 32];
    caa22 = a_b_shm[readCs + 3 * 64 + 0 ];
    caa23 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cba20;
    c[Cy00 + 32] = cba21;
    c[Cy04 + 0 ] = cba22;
    c[Cy04 + 32] = cba23;
    c[Cy08 + 0 ] = caa20;
    c[Cy08 + 32] = caa21;
    c[Cy12 + 0 ] = caa22;
    c[Cy12 + 32] = caa23;

    Cy00 += M;
    Cy04 += M;
    Cy08 += M;
    Cy12 += M;
    a_b_shm[writeCs + 0] = cba30;
    a_b_shm[writeCs + 1] = cba31;
    a_b_shm[writeCs + 2] = cba32;
    a_b_shm[writeCs + 3] = cba33;
    a_b_shm[writeCs + 32 + 0] = caa30;
    a_b_shm[writeCs + 32 + 1] = caa31;
    a_b_shm[writeCs + 32 + 2] = caa32;
    a_b_shm[writeCs + 32 + 3] = caa33;

    cba30 = a_b_shm[readCs + 0 * 64 + 0 ];
    cba31 = a_b_shm[readCs + 0 * 64 + 32];
    cba32 = a_b_shm[readCs + 1 * 64 + 0 ];
    cba33 = a_b_shm[readCs + 1 * 64 + 32];
    caa30 = a_b_shm[readCs + 2 * 64 + 0 ];
    caa31 = a_b_shm[readCs + 2 * 64 + 32];
    caa32 = a_b_shm[readCs + 3 * 64 + 0 ];
    caa33 = a_b_shm[readCs + 3 * 64 + 32];
    c[Cy00 + 0 ] = cba30;
    c[Cy00 + 32] = cba31;
    c[Cy04 + 0 ] = cba32;
    c[Cy04 + 32] = cba33;
    c[Cy08 + 0 ] = caa30;
    c[Cy08 + 32] = caa31;
    c[Cy12 + 0 ] = caa32;
    c[Cy12 + 32] = caa33;
}

__global__ void cuda_kernel_sgemm_100_tex(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta) {
    sgemm_block_64x64_tex(a, b, c, M, N, K, alpha, beta);
}
