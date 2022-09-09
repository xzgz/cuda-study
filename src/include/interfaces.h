#ifndef __INTERFACES_H__
#define __INTERFACES_H__

#include <stdlib.h>

void gpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta, int kernel_type);
void gpu_warmup();
void cpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta, int kernel_type);
void cpu_warmup();

#endif  // __INTERFACES_H__
