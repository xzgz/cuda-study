#include "interfaces.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>

#include <cuda_fp16.h>

using namespace std;

void verify(float* a, float* b, size_t arr_size, float eps) {
    float abs_err_ave = 0;
    for (size_t i = 0; i < arr_size; ++i) {
        abs_err_ave += abs(a[i] - b[i]);

        //        if (i < 32) {
        //            cout << a[i] << '\t' << b[i] << '\t' << abs(a[i] - b[i]) << endl;
        //        }
        if (fabs((a[i] - b[i]) / a[i]) > eps) {
            printf("[wrong answer]: at=%llu, a=%f, b=%f\n", i, a[i], b[i]);
            break;
        }
    }
    cout << "abs_err_ave=" << abs_err_ave / float(arr_size) << endl;
}

class Obj {
    // string s;
    float* ptr;
    int a[10];
};

class MyObj {
    int a;
    float b;
    int k;
    string ss;
    Obj obj;
    float ff[100];
};

int main() {
    MyObj* pi = new MyObj[30];
    // int* pi = new int[30];
    size_t piv = *((size_t*)&pi);
    std::cout << "piv=" << piv << std::endl;
    piv -= 4;
    int b4piv = *(*((int**)&piv));
    std::cout << "b4piv=" << b4piv << std::endl;
    // delete pi;
    delete[] pi;

    // dont use 2^n, it will cause cache crash on CPU
    size_t N, M, K;
    size_t NN = 55;
    // M = 1 * 64 * NN;
    // N = 1 * 64 * NN;
    // K = 1 * 64 * NN;
    // M = 191;
    // N = 256;
    // K = 3408;
    // M = 300;
    // N = 48;
    // K = 48;
    // M = 5120;
    M = 4096;
    N = 4096;
    K = 4096;
    // M = 1536;
    // N = 3072;
    // K = 1024;

    float alpha = 1.0f;
    float beta = 0.0f;
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* cc = new float[M * N];
    float* cb = new float[M * N];
    float* co = new float[M * N];
    half* A_h = NULL;
    half* B_h = NULL;
    A_h = (half*)malloc(sizeof(half) * M * K);
    B_h = (half*)malloc(sizeof(half) * K * N);
    //    printf("[data size]: A(%llux%llu), B(%llux%llu)\n", M, K, K, N);
    //    printf("[data size]: A(%lldx%ld), B(%dx%lu)\n", M, K, K, N);
    printf("[data size]: A(%dx%u), B(%lux%llu)\n", M, K, K, N);

    //    unsigned int seed = time(nullptr);
    //    srand(seed);
    for (int i = 0; i < M * K; i++) {
        // a[i] = rand() / float(RAND_MAX);
        a[i] = rand() % 3;
        A_h[i] = __float2half(a[i]);
        // A_h[i] = __float2half(__half2float(__float2half(a[i])));
    }
    for (int i = 0; i < K * N; i++) {
        // b[i] = rand() / float(RAND_MAX);
        b[i] = rand() % 3;
        B_h[i] = __float2half(b[i]);
    }

    memset(cc, 0, M * N * sizeof(float));
    memset(cb, 0, M * N * sizeof(float));
    memset(co, 0, M * N * sizeof(float));
    gpu_warmup();
    cpu_warmup();
    printf("[cpu sgemm kernel]\n");
    cpu_sgemm(a, b, cc, N, M, K, alpha, beta, 'm');
    float eps = 1e-6;

    memset(co, 0, M * N * sizeof(float));
    printf("[CutlassSgemmNN]\n");
    gpu_sgemm(a, b, co, N, M, K, alpha, beta, 'c', false, false);
    // gpu_sgemm((float*)A_h, (float*)B_h, co, N, M, K, alpha, beta, 'c', true, false);
    verify(cc, co, M * N, eps);

    // memset(co, 0, M * N * sizeof(float));
    // printf("[cuda_kernel_sgemm_100]\n");
    // gpu_sgemm(a, b, co, N, M, K, alpha, beta, 100, false, true);
    // verify(cc, co, M * N, eps);

    memset(co, 0, M * N * sizeof(float));
    printf("[ampere_sgemm_my_opt_128x256x8_kernel_no_pingpong]\n");
    gpu_sgemm(a, b, co, N, M, K, alpha, beta, 102, false, false);
    verify(cc, co, M * N, eps);

    memset(co, 0, M * N * sizeof(float));
    printf("[ampere_sgemm_my_opt_128x256x8_kernel_sm_pingpong]\n");
    gpu_sgemm(a, b, co, N, M, K, alpha, beta, 103, false, false);
    verify(cc, co, M * N, eps);

    memset(co, 0, M * N * sizeof(float));
    printf("[ampere_sgemm_my_opt_128x256x8_kernel_sm_reg_pingpong]\n");
    gpu_sgemm(a, b, co, N, M, K, alpha, beta, 104, false, false);
    verify(cc, co, M * N, eps);

    memset(co, 0, M * N * sizeof(float));
    printf("[ampere_sgemm_128x256x8_kernel]\n");
    gpu_sgemm(a, b, co, N, M, K, alpha, beta, 101, false, false);
    verify(cc, co, M * N, eps);

    memset(cb, 0, M * N * sizeof(float));
    printf("[cublasGemmEx]\n");
    gpu_sgemm(a, b, cb, N, M, K, alpha, beta, 21, false, false);
    // gpu_sgemm((float*)A_h, (float*)B_h, cb, N, M, K, alpha, beta, 21, true, false);
    verify(cc, cb, M * N, eps);

    memset(cb, 0, M * N * sizeof(float));
    printf("[cublasSgemm]\n");
    gpu_sgemm(a, b, cb, N, M, K, alpha, beta, 'b', false, false);
    verify(cc, cb, M * N, eps);

    free(A_h);
    free(B_h);
    delete[] a;
    delete[] b;
    delete[] cc;
    delete[] cb;
    delete[] co;

    return 0;
}
