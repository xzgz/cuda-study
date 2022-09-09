#include "interfaces.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#include <cuda_fp16.h>

using namespace std;

void verify(float *a, float *b, size_t arr_size, float eps)
{
    float abs_err_ave = 0;
    // printf("out  r0, c4: %f\n", b[4]);
    // printf("out  r7, c8: %f\n", b[7  * 64 + 8]);
    // printf("out r39, c8: %f\n", b[39 * 64 + 8]);
    // printf("out r16, c5: %f\n", b[16 * 64 + 5]);
    // printf("out b r0, c31: %f\n", b[0 * 64 + 31]);

    // printf("out  r0, c4: %f\n", a[4]);
    // printf("out  r7, c8: %f\n", a[7  * 64 + 8]);
    // printf("out r39, c8: %f\n", a[39 * 64 + 8]);
    // printf("out r16, c5: %f\n", a[16 * 64 + 5]);
    // printf("out a r0, c31: %f\n", a[0 * 64 + 31]);

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
    delete [] pi;

    // dont use 2^n, it will cause cache crash on CPU
    size_t N, M, K;
    size_t NN = 55;
    // M = 1 * 64 * NN;
    // N = 1 * 64 * NN;
    // K = 1 * 64 * NN;
    // M = 191;
    // N = 256;
    // K = 3408;
    M = 300;
    N = 48;
    K = 48;

    float alpha = 1.0f;
    float beta = 0.0f;
    float *a = new float[M * K];
    float *b = new float[K * N];
    float *c1 = new float[M * N];
    float *c2 = new float[M * N];
    float *c3 = new float[M * N];
    float *c4 = new float[M * N];
    float *cb = new float[M * N];
    float *cm = new float[M * N];
    float *cc = new float[M * N];
    float *cr = new float[M * N];
    float *ct = new float[M * N];
    half *A_h = NULL;
    half *B_h = NULL;
    A_h = (half*) malloc(sizeof(half) * M * K);
    B_h = (half*) malloc(sizeof(half) * K * N);
//    printf("[data size]: A(%llux%llu), B(%llux%llu)\n", M, K, K, N);
//    printf("[data size]: A(%lldx%ld), B(%dx%lu)\n", M, K, K, N);
    printf("[data size]: A(%dx%u), B(%lux%llu)\n", M, K, K, N);

//    unsigned int seed = time(nullptr);
//    srand(seed);
    for (int i = 0; i < M * K; i++) {
//        a[i] = rand()/float(RAND_MAX);
        a[i] = rand() % 3;
        // A_h[i] = __float2half(a[i]);
        A_h[i] = __float2half(__half2float(__float2half(a[i])));
    }
    for (int i = 0; i < K * N; i++) {
//        b[i] = rand()/float(RAND_MAX);
        b[i] = rand() % 3;
        // B_h[i] = __float2half(b[i]);
        B_h[i] = __float2half(__half2float(__float2half(b[i])));
    }
    for (size_t i = 0; i < M * N; ++i) {
        c1[i] = 0.0;
        c2[i] = 0.0;
        c3[i] = 0.0;
        c4[i] = 0.0;
        cb[i] = 0.0;
        cm[i] = 0.0;
        cc[i] = 0.0;
        cr[i] = 0.0;
        ct[i] = 0.0;
    }

    gpu_warmup();
    cpu_warmup();
    printf("[cpu sgemm kernel 0]\n");
    cpu_sgemm(a, b, c1, N, M, K, alpha, beta, 0);

    float eps = 1e-5; // mkl's error is larger, why?
//    printf("[cpu sgemm kernel openblas]\n");
//    cpu_sgemm(a, b, cm, N, M, K, alpha, beta, 'm');
//    verify(c1, cm, M * N, eps);

    printf("[gpu sgemm kernel cublas]\n");
    gpu_sgemm(a, b, cb, N, M, K, alpha, beta, 'b');
    verify(c1, cb, M * N, eps);

    eps = 1e-6;
//    printf("[gpu sgemm kernel 0]\n");
//    gpu_sgemm(a, b, c2, N, M, K, alpha, beta, 0);
//    verify(c1, c2, M * N, eps);
    // printf("[gpu sgemm kernel 1]\n");
    // gpu_sgemm(a, b, c3, N, M, K, alpha, beta, 1);
    // verify(c1, c3, M * N, eps);
    // printf("[gpu sgemm kernel 2]\n");
    // gpu_sgemm(a, b, c4, N, M, K, alpha, beta, 2);
    // verify(c1, c4, M * N, eps);
    // gpu_sgemm(a, b, c3, N, M, K, alpha, beta, 20);
    // verify(c1, c3, M * N, eps);
    // gpu_sgemm(a, b, c3, N, M, K, alpha, beta, 100);
    // verify(c1, c3, M * N, eps);

    printf("[gpu sgemm kernel cutlass]\n");
    gpu_sgemm(a, b, cc, N, M, K, alpha, beta, 'c');
    verify(c1, cc, M * N, eps);
//    printf("[gpu sgemm kernel reference]\n");
//    gpu_sgemm(a, b, cr, N, M, K, alpha, beta, 'r');
//    verify(c1, cr, M * N, eps);

//    printf("[gpu sgemm kernel use tensor core]\n");
//    gpu_sgemm((float *)A_h, (float *)B_h, ct, N, M, K, alpha, beta, 't');
//    verify(c1, ct, M * N, eps);

    free(A_h);
    free(B_h);
    delete [] a;
    delete [] b;
    delete [] c1;
    delete [] c2;
    delete [] c3;
    delete [] c4;
    delete [] cb;
    delete [] cm;
    delete [] cc;
    delete [] cr;
    delete [] ct;
    return 0;
}
