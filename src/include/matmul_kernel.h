#ifndef __MATMUL_KERNEL_H__
#define __MATMUL_KERNEL_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

#define STRIDE_X 16
#define STRIDE_Y 16

void cpu_kernel_sgemm_0(float* a, float* b, float* c, size_t N, size_t M, size_t K, float alpha, float beta);

__global__ void cuda_kernel_warmup(float* p);
__global__ void cuda_kernel_sgemm_0(
        float* a, float* b, float* c, size_t N, size_t M, size_t K, float alpha, float beta);
__global__ void cuda_kernel_sgemm_1(
        float* a, float* b, float* c, size_t N, size_t M, size_t K, float alpha, float beta);
__global__ void cuda_kernel_sgemm_2(
        float* a, float* b, float* c, size_t N, size_t M, size_t K, float alpha, float beta);
__global__ void cuda_kernel_sgemm_2_64x64(
        float* a, float* b, float* c, size_t N, size_t M, size_t K, float alpha, float beta);
__global__ void wmma_sgemm_kernel(half* a, half* b, float* c, float* d, int M, int N, int K, float alpha, float beta);
__global__ void ReferenceGemm_kernel(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb,
        float beta, float* C, int ldc);
cudaError_t ReferenceGemm(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb,
        float beta, float* C, int ldc);
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb,
        float beta, float* C, int ldc, int cycle_count);

__global__ void cuda_kernel_sgemm_100(
        float* a, float* b, float* c, size_t M, size_t N, size_t K, float alpha, float beta);
__global__ void cuda_kernel_sgemm_100_v2(
        float* a, float* b, float* c, size_t M, size_t N, size_t K, float alpha, float beta);
__global__ __launch_bounds__(256) void ampere_sgemm_128x256x8_kernel(
        const float* A, const float* B, float* C, size_t m, size_t n, size_t k, size_t B_ldg_step, int cycle_count);
__global__ __launch_bounds__(256) void ampere_sgemm_my_opt_128x256x8_kernel_no_pingpong(
        const float* A, const float* B, float* C, size_t m, size_t n, size_t k, size_t B_ldg_step, int cycle_count);
__global__ __launch_bounds__(256) void ampere_sgemm_my_opt_128x256x8_kernel_sm_pingpong(
        const float* A, const float* B, float* C, size_t m, size_t n, size_t k, size_t B_ldg_step, int cycle_count);
__global__ __launch_bounds__(256) void ampere_sgemm_my_opt_128x256x8_kernel_sm_reg_pingpong(
        const float* A, const float* B, float* C, size_t m, size_t n, size_t k, size_t B_ldg_step, int cycle_count);
#endif // __MATMUL_KERNEL_H__
