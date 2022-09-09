/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// includes, project
//#include <cutil_inline.h>
//#include <helper_functions.h>

// includes, kernels
#include "matrixMul_kernel.cuh"
#include "matrixMul_naive.cuh"
#include "matrixMul_tiling.cuh"
#include "matrixMul_coalescing.cuh"
#include "matrixMul_noBankConflict.cuh"
#include "matrixMul_compOpt.cuh"
#include "matrixMul_unroll.cuh"
#include "matrixMul_prefetch.cuh"
#include "cutlass/gemm/device/gemm.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {
        return 0;
    }

    return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = (int)strlen(string_ref);

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
            {
                bFound = true;
                continue;
            }
        }
    }

    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

cudaError_t CutlassSgemmNN(
        int M,
        int N,
        int K,
        float alpha,
        float const *A,
        int lda,
        float const *B,
        int ldb,
        float beta,
        float *C,
        int ldc) {

    // Define type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
    //
    // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

    using ColumnMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
            ColumnMajor,  // Layout of A matrix
            float,        // Data-type of B matrix
            ColumnMajor,  // Layout of B matrix
            float,        // Data-type of C matrix
            ColumnMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{

    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/

    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // utilities
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    float flop = 2 * (float)WC * (float)HC * (float)WA;

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

//#if CHECK_RESULT == 1
//    // create and start timer
//    cudaEventCreate(&start);
//    cudaEventRecord(start, NULL);
//    // compute reference solution
//    float* reference = (float*) malloc(mem_size_C);
//    computeGold(reference, h_A, h_B, HA, WA, WB);
//    // stop and destroy timer
//    cudaEventCreate(&stop);
//    cudaEventRecord(stop, NULL);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&msecTotal, start, stop);
//    printf("Naive CPU (Golden Reference)\n");
//    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
//#endif

    dim3 threads,grid;

    /****************************************************/
    /*  CUDA SDK example                                */
    /****************************************************/

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
   // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // execute the kernel
    matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    float *reference = (float*) malloc(mem_size_C);
    cudaMemcpy(reference, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    printf("GPU SDK Sample\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  naive implementation on GPU                     */
    /****************************************************/

#if ENABLE_NAIVE == 1
    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_naive<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    printf("Naive GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

#endif

    /****************************************************/
    /*  Tiling without global mem coalescing            */
    /****************************************************/

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_tiling<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    printf("Tiling GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Global mem coalescing with smem bank conflict   */
    /****************************************************/

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_coalescing<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    printf("Global mem coalescing GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Global mem coalescing w/o smem bank conflict    */
    /****************************************************/

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_noBankConflict<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    printf("Remove shared mem bank conflict GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Threads perform computation optimizatin         */
    /****************************************************/

    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // setup execution parameters
    threads = dim3(BLOCK_SIZE, 4);
    grid = dim3(WC / (BLOCK_SIZE*4), HC / BLOCK_SIZE);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // naive implementation
    matrixMul_compOpt<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Threads perform computation optimization GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif


    /****************************************************/
    /*  Loop Unrolling                                  */
    /****************************************************/

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, 4);
    grid = dim3(WC / (BLOCK_SIZE*4), HC / BLOCK_SIZE);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_unroll<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
    printf("Loop unrolling GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Prefetching                                     */
    /****************************************************/

    // setup execution parameters
    threads = dim3(BLOCK_SIZE, 4);
    grid = dim3(WC / (BLOCK_SIZE*4), HC / BLOCK_SIZE);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    // naive implementation
    matrixMul_prefetch<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    printf("Prefetching GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    for (size_t i = 0; i < HA * WB; ++i) {
        h_C[i] = 0.0;
    }
    printf("rewrite d_C before compute\n");

    float alpha = 1.0;
    float beta = 0.0;
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    cudaError_t result = CutlassSgemmNN(HA, WB, WA, 1.0, d_A, WA, d_B, WB, 0.0, d_C, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    if (result == cudaSuccess) {
        printf("CutlassSgemmNN success\n");
    }
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    printf("CutlassSgemmNN GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    for (size_t i = 0; i < HA * WB; ++i) {
        h_C[i] = 0.0;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WB, HA, WA, &alpha, d_B, WB, d_A, WA, &beta, d_C, WB);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    if (result == cudaSuccess) {
        printf("cublasSgemm success\n");
    }
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    // stop and destroy timer
    printf("cublasSgemm GPU\n");
    printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
    cublasDestroy(handle);
#if CHECK_RESULT == 1
    // check result
    printDiff(reference, h_C, WC, HC);
#endif

    /****************************************************/
    /*  Cleaning                                        */
    /****************************************************/
    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
#if CHECK_RESULT == 1
    free(reference);
#endif
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  int count = 0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (fabs(data1[k] - data2[k]) > 0.1 ) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i,j, data1[k], data2[k]);
         error_count++;
      }
      if (count < 10) {
          printf("CPU=%4.4f, GPU=%4.4f \n", data1[k], data2[k]);
      }
      ++count;
    }
  }
  printf("Total Errors = %d \n", error_count);
}
