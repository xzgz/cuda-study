#include <iostream>

#include "matmul_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float aa[10];

__device__ void sgemm_block_64x64_v2(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta) {

    // float* dev_p = nullptr;
    // cudaMalloc((void **)&dev_p, 16 * 32 * sizeof(float));
    // float aa[100000];
    // for (int i = threadIdx.x; i < 10000; i += blockDim.x) {
    //     aa[i] = i;
    // }

    __shared__ float a_b_shm[2 * 16 * 64];

    // int tid = threadIdx.x & 0x3f;
    int tid = threadIdx.x;
    int ldx = tid >= 32 ? N : M;
    int tid2 = (tid >> 4) & 1;
    // int tid15 = tid & 0xf;
    int tid15 = tid & 15;
    int track0 = tid2 * ldx + tid15 * 4;
    int track2 = track0 + 2 * ldx;
    int track4 = track0 + 4 * ldx;
    int track6 = track0 + 6 * ldx;
    // int end = track0 + (K - 8) * ldx;
    int end = track0 + K * ldx;
    int write_offset = tid2 * 64 + tid15 * 4;
    write_offset += tid >= 32 ? 512 : 0;

    int readAs = ((tid >> 1) & 7) << 2;
    int readBs = ((((tid & 0x30) >> 3) | (tid & 1)) << 2) + 512;

    float* read_addr = tid >= 32 ? b : a;

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

    float j0Ab00, j0Ab01, j0Ab02, j0Ab03;
    float j0Bb00, j0Bb01, j0Bb02, j0Bb03;
    float j0Aa00, j0Aa01, j0Aa02, j0Aa03;
    float j0Ba00, j0Ba01, j0Ba02, j0Ba03;

    float j1Ab00, j1Ab01, j1Ab02, j1Ab03;
    float j1Bb00, j1Bb01, j1Bb02, j1Bb03;
    float j1Aa00, j1Aa01, j1Aa02, j1Aa03;
    float j1Ba00, j1Ba01, j1Ba02, j1Ba03;

    float loadX00, loadX01, loadX02, loadX03;
    float loadX20, loadX21, loadX22, loadX23;
    float loadX40, loadX41, loadX42, loadX43;
    float loadX60, loadX61, loadX62, loadX63;

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

    track0 += 8 * ldx;
    track2 += 8 * ldx;
    track4 += 8 * ldx;
    track6 += 8 * ldx;
    __syncthreads();
    write_offset ^= 16 * 64;

    j0Ab00 = a_b_shm[readAs + 0];
    j0Ab01 = a_b_shm[readAs + 1];
    j0Ab02 = a_b_shm[readAs + 2];
    j0Ab03 = a_b_shm[readAs + 3];

    j0Bb00 = a_b_shm[readBs + 0];
    j0Bb01 = a_b_shm[readBs + 1];
    j0Bb02 = a_b_shm[readBs + 2];
    j0Bb03 = a_b_shm[readBs + 3];

    j0Aa00 = a_b_shm[readAs + 32 + 0];
    j0Aa01 = a_b_shm[readAs + 32 + 1];
    j0Aa02 = a_b_shm[readAs + 32 + 2];
    j0Aa03 = a_b_shm[readAs + 32 + 3];

    j0Ba00 = a_b_shm[readBs + 32 + 0];
    j0Ba01 = a_b_shm[readBs + 32 + 1];
    j0Ba02 = a_b_shm[readBs + 32 + 2];
    j0Ba03 = a_b_shm[readBs + 32 + 3];

    while (track0 <= end) {
        // a_b_shm[write_offset + 0 * 64 + 0] = read_addr[track0 + 0];
        // a_b_shm[write_offset + 0 * 64 + 1] = read_addr[track0 + 1];
        // a_b_shm[write_offset + 0 * 64 + 2] = read_addr[track0 + 2];
        // a_b_shm[write_offset + 0 * 64 + 3] = read_addr[track0 + 3];

        // a_b_shm[write_offset + 2 * 64 + 0] = read_addr[track2 + 0];
        // a_b_shm[write_offset + 2 * 64 + 1] = read_addr[track2 + 1];
        // a_b_shm[write_offset + 2 * 64 + 2] = read_addr[track2 + 2];
        // a_b_shm[write_offset + 2 * 64 + 3] = read_addr[track2 + 3];

        // a_b_shm[write_offset + 4 * 64 + 0] = read_addr[track4 + 0];
        // a_b_shm[write_offset + 4 * 64 + 1] = read_addr[track4 + 1];
        // a_b_shm[write_offset + 4 * 64 + 2] = read_addr[track4 + 2];
        // a_b_shm[write_offset + 4 * 64 + 3] = read_addr[track4 + 3];

        // a_b_shm[write_offset + 6 * 64 + 0] = read_addr[track6 + 0];
        // a_b_shm[write_offset + 6 * 64 + 1] = read_addr[track6 + 1];
        // a_b_shm[write_offset + 6 * 64 + 2] = read_addr[track6 + 2];
        // a_b_shm[write_offset + 6 * 64 + 3] = read_addr[track6 + 3];

        // loadX00 = read_addr[track0 + 0];
        // loadX01 = read_addr[track0 + 1];
        // loadX02 = read_addr[track0 + 2];
        // loadX03 = read_addr[track0 + 3];
        // loadX20 = read_addr[track2 + 0];
        // loadX21 = read_addr[track2 + 1];
        // loadX22 = read_addr[track2 + 2];
        // loadX23 = read_addr[track2 + 3];
        // loadX40 = read_addr[track4 + 0];
        // loadX41 = read_addr[track4 + 1];
        // loadX42 = read_addr[track4 + 2];
        // loadX43 = read_addr[track4 + 3];
        // loadX60 = read_addr[track6 + 0];
        // loadX61 = read_addr[track6 + 1];
        // loadX62 = read_addr[track6 + 2];
        // loadX63 = read_addr[track6 + 3];

        // a_b_shm[write_offset + 0 * 64 + 0] = loadX00;
        // a_b_shm[write_offset + 0 * 64 + 1] = loadX01;
        // a_b_shm[write_offset + 0 * 64 + 2] = loadX02;
        // a_b_shm[write_offset + 0 * 64 + 3] = loadX03;
        // a_b_shm[write_offset + 2 * 64 + 0] = loadX20;
        // a_b_shm[write_offset + 2 * 64 + 1] = loadX21;
        // a_b_shm[write_offset + 2 * 64 + 2] = loadX22;
        // a_b_shm[write_offset + 2 * 64 + 3] = loadX23;
        // a_b_shm[write_offset + 4 * 64 + 0] = loadX40;
        // a_b_shm[write_offset + 4 * 64 + 1] = loadX41;
        // a_b_shm[write_offset + 4 * 64 + 2] = loadX42;
        // a_b_shm[write_offset + 4 * 64 + 3] = loadX43;
        // a_b_shm[write_offset + 6 * 64 + 0] = loadX60;
        // a_b_shm[write_offset + 6 * 64 + 1] = loadX61;
        // a_b_shm[write_offset + 6 * 64 + 2] = loadX62;
        // a_b_shm[write_offset + 6 * 64 + 3] = loadX63;

        cbb00 += j0Bb00 * j0Ab00;
        cbb01 += j0Bb00 * j0Ab01;
        j1Ab00 = a_b_shm[readAs + 1 * 64 + 0];
        j1Ab01 = a_b_shm[readAs + 1 * 64 + 1];
        j1Ab02 = a_b_shm[readAs + 1 * 64 + 2];
        j1Ab03 = a_b_shm[readAs + 1 * 64 + 3];
        cbb02 += j0Bb00 * j0Ab02;
        cbb03 += j0Bb00 * j0Ab03;
        j1Bb00 = a_b_shm[readBs + 1 * 64 + 0];
        j1Bb01 = a_b_shm[readBs + 1 * 64 + 1];
        j1Bb02 = a_b_shm[readBs + 1 * 64 + 2];
        j1Bb03 = a_b_shm[readBs + 1 * 64 + 3];
        cbb10 += j0Bb01 * j0Ab00;
        cbb11 += j0Bb01 * j0Ab01;
        j1Aa00 = a_b_shm[readAs + 1 * 64 + 32 + 0];
        j1Aa01 = a_b_shm[readAs + 1 * 64 + 32 + 1];
        j1Aa02 = a_b_shm[readAs + 1 * 64 + 32 + 2];
        j1Aa03 = a_b_shm[readAs + 1 * 64 + 32 + 3];
        cbb12 += j0Bb01 * j0Ab02;
        cbb13 += j0Bb01 * j0Ab03;
        j1Ba00 = a_b_shm[readBs + 1 * 64 + 32 + 0];
        j1Ba01 = a_b_shm[readBs + 1 * 64 + 32 + 1];
        j1Ba02 = a_b_shm[readBs + 1 * 64 + 32 + 2];
        j1Ba03 = a_b_shm[readBs + 1 * 64 + 32 + 3];

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

        loadX00 = read_addr[track0 + 0];
        loadX01 = read_addr[track0 + 1];
        loadX02 = read_addr[track0 + 2];
        loadX03 = read_addr[track0 + 3];
        cab00 += j0Bb00 * j0Aa00;
        cab01 += j0Bb00 * j0Aa01;
        loadX20 = read_addr[track2 + 0];
        loadX21 = read_addr[track2 + 1];
        loadX22 = read_addr[track2 + 2];
        loadX23 = read_addr[track2 + 3];
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


        cbb00 += j1Bb00 * j1Ab00;
        cbb01 += j1Bb00 * j1Ab01;
        j0Ab00 = a_b_shm[readAs + 2 * 64 + 0];
        j0Ab01 = a_b_shm[readAs + 2 * 64 + 1];
        j0Ab02 = a_b_shm[readAs + 2 * 64 + 2];
        j0Ab03 = a_b_shm[readAs + 2 * 64 + 3];
        cbb02 += j1Bb00 * j1Ab02;
        cbb03 += j1Bb00 * j1Ab03;
        j0Bb00 = a_b_shm[readBs + 2 * 64 + 0];
        j0Bb01 = a_b_shm[readBs + 2 * 64 + 1];
        j0Bb02 = a_b_shm[readBs + 2 * 64 + 2];
        j0Bb03 = a_b_shm[readBs + 2 * 64 + 3];
        cbb10 += j1Bb01 * j1Ab00;
        cbb11 += j1Bb01 * j1Ab01;
        j0Aa00 = a_b_shm[readAs + 2 * 64 + 32 + 0];
        j0Aa01 = a_b_shm[readAs + 2 * 64 + 32 + 1];
        j0Aa02 = a_b_shm[readAs + 2 * 64 + 32 + 2];
        j0Aa03 = a_b_shm[readAs + 2 * 64 + 32 + 3];
        cbb12 += j1Bb01 * j1Ab02;
        cbb13 += j1Bb01 * j1Ab03;
        j0Ba00 = a_b_shm[readBs + 2 * 64 + 32 + 0];
        j0Ba01 = a_b_shm[readBs + 2 * 64 + 32 + 1];
        j0Ba02 = a_b_shm[readBs + 2 * 64 + 32 + 2];
        j0Ba03 = a_b_shm[readBs + 2 * 64 + 32 + 3];

        cbb20 += j1Bb02 * j1Ab00;
        cbb21 += j1Bb02 * j1Ab01;
        cbb22 += j1Bb02 * j1Ab02;
        cbb23 += j1Bb02 * j1Ab03;
        cbb30 += j1Bb03 * j1Ab00;
        cbb31 += j1Bb03 * j1Ab01;
        cbb32 += j1Bb03 * j1Ab02;
        cbb33 += j1Bb03 * j1Ab03;
        cba00 += j1Ba00 * j1Ab00;
        cba01 += j1Ba00 * j1Ab01;
        cba02 += j1Ba00 * j1Ab02;
        cba03 += j1Ba00 * j1Ab03;
        cba10 += j1Ba01 * j1Ab00;
        cba11 += j1Ba01 * j1Ab01;
        cba12 += j1Ba01 * j1Ab02;
        cba13 += j1Ba01 * j1Ab03;
        cba20 += j1Ba02 * j1Ab00;
        cba21 += j1Ba02 * j1Ab01;
        cba22 += j1Ba02 * j1Ab02;
        cba23 += j1Ba02 * j1Ab03;
        cba30 += j1Ba03 * j1Ab00;
        cba31 += j1Ba03 * j1Ab01;
        cba32 += j1Ba03 * j1Ab02;
        cba33 += j1Ba03 * j1Ab03;

        loadX40 = read_addr[track4 + 0];
        loadX41 = read_addr[track4 + 1];
        loadX42 = read_addr[track4 + 2];
        loadX43 = read_addr[track4 + 3];
        cab00 += j1Bb00 * j1Aa00;
        cab01 += j1Bb00 * j1Aa01;
        loadX60 = read_addr[track6 + 0];
        loadX61 = read_addr[track6 + 1];
        loadX62 = read_addr[track6 + 2];
        loadX63 = read_addr[track6 + 3];
        cab02 += j1Bb00 * j1Aa02;
        cab03 += j1Bb00 * j1Aa03;
        cab10 += j1Bb01 * j1Aa00;
        cab11 += j1Bb01 * j1Aa01;
        cab12 += j1Bb01 * j1Aa02;
        cab13 += j1Bb01 * j1Aa03;
        cab20 += j1Bb02 * j1Aa00;
        cab21 += j1Bb02 * j1Aa01;
        cab22 += j1Bb02 * j1Aa02;
        cab23 += j1Bb02 * j1Aa03;
        cab30 += j1Bb03 * j1Aa00;
        cab31 += j1Bb03 * j1Aa01;
        cab32 += j1Bb03 * j1Aa02;
        cab33 += j1Bb03 * j1Aa03;
        caa00 += j1Ba00 * j1Aa00;
        caa01 += j1Ba00 * j1Aa01;
        caa02 += j1Ba00 * j1Aa02;
        caa03 += j1Ba00 * j1Aa03;
        caa10 += j1Ba01 * j1Aa00;
        caa11 += j1Ba01 * j1Aa01;
        caa12 += j1Ba01 * j1Aa02;
        caa13 += j1Ba01 * j1Aa03;
        caa20 += j1Ba02 * j1Aa00;
        caa21 += j1Ba02 * j1Aa01;
        caa22 += j1Ba02 * j1Aa02;
        caa23 += j1Ba02 * j1Aa03;
        caa30 += j1Ba03 * j1Aa00;
        caa31 += j1Ba03 * j1Aa01;
        caa32 += j1Ba03 * j1Aa02;
        caa33 += j1Ba03 * j1Aa03;


        cbb00 += j0Bb00 * j0Ab00;
        cbb01 += j0Bb00 * j0Ab01;
        j1Ab00 = a_b_shm[readAs + 3 * 64 + 0];
        j1Ab01 = a_b_shm[readAs + 3 * 64 + 1];
        j1Ab02 = a_b_shm[readAs + 3 * 64 + 2];
        j1Ab03 = a_b_shm[readAs + 3 * 64 + 3];
        cbb02 += j0Bb00 * j0Ab02;
        cbb03 += j0Bb00 * j0Ab03;
        j1Bb00 = a_b_shm[readBs + 3 * 64 + 0];
        j1Bb01 = a_b_shm[readBs + 3 * 64 + 1];
        j1Bb02 = a_b_shm[readBs + 3 * 64 + 2];
        j1Bb03 = a_b_shm[readBs + 3 * 64 + 3];
        cbb10 += j0Bb01 * j0Ab00;
        cbb11 += j0Bb01 * j0Ab01;
        j1Aa00 = a_b_shm[readAs + 3 * 64 + 32 + 0];
        j1Aa01 = a_b_shm[readAs + 3 * 64 + 32 + 1];
        j1Aa02 = a_b_shm[readAs + 3 * 64 + 32 + 2];
        j1Aa03 = a_b_shm[readAs + 3 * 64 + 32 + 3];
        cbb12 += j0Bb01 * j0Ab02;
        cbb13 += j0Bb01 * j0Ab03;
        j1Ba00 = a_b_shm[readBs + 3 * 64 + 32 + 0];
        j1Ba01 = a_b_shm[readBs + 3 * 64 + 32 + 1];
        j1Ba02 = a_b_shm[readBs + 3 * 64 + 32 + 2];
        j1Ba03 = a_b_shm[readBs + 3 * 64 + 32 + 3];

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


        cbb00 += j1Bb00 * j1Ab00;
        cbb01 += j1Bb00 * j1Ab01;
        j0Ab00 = a_b_shm[readAs + 4 * 64 + 0];
        j0Ab01 = a_b_shm[readAs + 4 * 64 + 1];
        j0Ab02 = a_b_shm[readAs + 4 * 64 + 2];
        j0Ab03 = a_b_shm[readAs + 4 * 64 + 3];
        cbb02 += j1Bb00 * j1Ab02;
        cbb03 += j1Bb00 * j1Ab03;
        j0Bb00 = a_b_shm[readBs + 4 * 64 + 0];
        j0Bb01 = a_b_shm[readBs + 4 * 64 + 1];
        j0Bb02 = a_b_shm[readBs + 4 * 64 + 2];
        j0Bb03 = a_b_shm[readBs + 4 * 64 + 3];
        cbb10 += j1Bb01 * j1Ab00;
        cbb11 += j1Bb01 * j1Ab01;
        j0Aa00 = a_b_shm[readAs + 4 * 64 + 32 + 0];
        j0Aa01 = a_b_shm[readAs + 4 * 64 + 32 + 1];
        j0Aa02 = a_b_shm[readAs + 4 * 64 + 32 + 2];
        j0Aa03 = a_b_shm[readAs + 4 * 64 + 32 + 3];
        cbb12 += j1Bb01 * j1Ab02;
        cbb13 += j1Bb01 * j1Ab03;
        j0Ba00 = a_b_shm[readBs + 4 * 64 + 32 + 0];
        j0Ba01 = a_b_shm[readBs + 4 * 64 + 32 + 1];
        j0Ba02 = a_b_shm[readBs + 4 * 64 + 32 + 2];
        j0Ba03 = a_b_shm[readBs + 4 * 64 + 32 + 3];

        cbb20 += j1Bb02 * j1Ab00;
        cbb21 += j1Bb02 * j1Ab01;
        cbb22 += j1Bb02 * j1Ab02;
        cbb23 += j1Bb02 * j1Ab03;
        cbb30 += j1Bb03 * j1Ab00;
        cbb31 += j1Bb03 * j1Ab01;
        cbb32 += j1Bb03 * j1Ab02;
        cbb33 += j1Bb03 * j1Ab03;
        cba00 += j1Ba00 * j1Ab00;
        cba01 += j1Ba00 * j1Ab01;
        cba02 += j1Ba00 * j1Ab02;
        cba03 += j1Ba00 * j1Ab03;
        cba10 += j1Ba01 * j1Ab00;
        cba11 += j1Ba01 * j1Ab01;
        cba12 += j1Ba01 * j1Ab02;
        cba13 += j1Ba01 * j1Ab03;
        cba20 += j1Ba02 * j1Ab00;
        cba21 += j1Ba02 * j1Ab01;
        cba22 += j1Ba02 * j1Ab02;
        cba23 += j1Ba02 * j1Ab03;
        cba30 += j1Ba03 * j1Ab00;
        cba31 += j1Ba03 * j1Ab01;
        cba32 += j1Ba03 * j1Ab02;
        cba33 += j1Ba03 * j1Ab03;
        cab00 += j1Bb00 * j1Aa00;
        cab01 += j1Bb00 * j1Aa01;
        cab02 += j1Bb00 * j1Aa02;
        cab03 += j1Bb00 * j1Aa03;
        cab10 += j1Bb01 * j1Aa00;
        cab11 += j1Bb01 * j1Aa01;
        cab12 += j1Bb01 * j1Aa02;
        cab13 += j1Bb01 * j1Aa03;
        cab20 += j1Bb02 * j1Aa00;
        cab21 += j1Bb02 * j1Aa01;
        cab22 += j1Bb02 * j1Aa02;
        cab23 += j1Bb02 * j1Aa03;
        cab30 += j1Bb03 * j1Aa00;
        cab31 += j1Bb03 * j1Aa01;
        cab32 += j1Bb03 * j1Aa02;
        cab33 += j1Bb03 * j1Aa03;
        caa00 += j1Ba00 * j1Aa00;
        caa01 += j1Ba00 * j1Aa01;
        caa02 += j1Ba00 * j1Aa02;
        caa03 += j1Ba00 * j1Aa03;
        caa10 += j1Ba01 * j1Aa00;
        caa11 += j1Ba01 * j1Aa01;
        caa12 += j1Ba01 * j1Aa02;
        caa13 += j1Ba01 * j1Aa03;
        caa20 += j1Ba02 * j1Aa00;
        caa21 += j1Ba02 * j1Aa01;
        caa22 += j1Ba02 * j1Aa02;
        caa23 += j1Ba02 * j1Aa03;
        caa30 += j1Ba03 * j1Aa00;
        caa31 += j1Ba03 * j1Aa01;
        caa32 += j1Ba03 * j1Aa02;
        caa33 += j1Ba03 * j1Aa03;


        cbb00 += j0Bb00 * j0Ab00;
        cbb01 += j0Bb00 * j0Ab01;
        j1Ab00 = a_b_shm[readAs + 5 * 64 + 0];
        j1Ab01 = a_b_shm[readAs + 5 * 64 + 1];
        j1Ab02 = a_b_shm[readAs + 5 * 64 + 2];
        j1Ab03 = a_b_shm[readAs + 5 * 64 + 3];
        cbb02 += j0Bb00 * j0Ab02;
        cbb03 += j0Bb00 * j0Ab03;
        j1Bb00 = a_b_shm[readBs + 5 * 64 + 0];
        j1Bb01 = a_b_shm[readBs + 5 * 64 + 1];
        j1Bb02 = a_b_shm[readBs + 5 * 64 + 2];
        j1Bb03 = a_b_shm[readBs + 5 * 64 + 3];
        cbb10 += j0Bb01 * j0Ab00;
        cbb11 += j0Bb01 * j0Ab01;
        j1Aa00 = a_b_shm[readAs + 5 * 64 + 32 + 0];
        j1Aa01 = a_b_shm[readAs + 5 * 64 + 32 + 1];
        j1Aa02 = a_b_shm[readAs + 5 * 64 + 32 + 2];
        j1Aa03 = a_b_shm[readAs + 5 * 64 + 32 + 3];
        cbb12 += j0Bb01 * j0Ab02;
        cbb13 += j0Bb01 * j0Ab03;
        j1Ba00 = a_b_shm[readBs + 5 * 64 + 32 + 0];
        j1Ba01 = a_b_shm[readBs + 5 * 64 + 32 + 1];
        j1Ba02 = a_b_shm[readBs + 5 * 64 + 32 + 2];
        j1Ba03 = a_b_shm[readBs + 5 * 64 + 32 + 3];

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


        cbb00 += j1Bb00 * j1Ab00;
        cbb01 += j1Bb00 * j1Ab01;
        j0Ab00 = a_b_shm[readAs + 6 * 64 + 0];
        j0Ab01 = a_b_shm[readAs + 6 * 64 + 1];
        j0Ab02 = a_b_shm[readAs + 6 * 64 + 2];
        j0Ab03 = a_b_shm[readAs + 6 * 64 + 3];
        cbb02 += j1Bb00 * j1Ab02;
        cbb03 += j1Bb00 * j1Ab03;
        j0Bb00 = a_b_shm[readBs + 6 * 64 + 0];
        j0Bb01 = a_b_shm[readBs + 6 * 64 + 1];
        j0Bb02 = a_b_shm[readBs + 6 * 64 + 2];
        j0Bb03 = a_b_shm[readBs + 6 * 64 + 3];
        cbb10 += j1Bb01 * j1Ab00;
        cbb11 += j1Bb01 * j1Ab01;
        j0Aa00 = a_b_shm[readAs + 6 * 64 + 32 + 0];
        j0Aa01 = a_b_shm[readAs + 6 * 64 + 32 + 1];
        j0Aa02 = a_b_shm[readAs + 6 * 64 + 32 + 2];
        j0Aa03 = a_b_shm[readAs + 6 * 64 + 32 + 3];
        cbb12 += j1Bb01 * j1Ab02;
        cbb13 += j1Bb01 * j1Ab03;
        j0Ba00 = a_b_shm[readBs + 6 * 64 + 32 + 0];
        j0Ba01 = a_b_shm[readBs + 6 * 64 + 32 + 1];
        j0Ba02 = a_b_shm[readBs + 6 * 64 + 32 + 2];
        j0Ba03 = a_b_shm[readBs + 6 * 64 + 32 + 3];

        cbb20 += j1Bb02 * j1Ab00;
        cbb21 += j1Bb02 * j1Ab01;
        cbb22 += j1Bb02 * j1Ab02;
        cbb23 += j1Bb02 * j1Ab03;
        cbb30 += j1Bb03 * j1Ab00;
        cbb31 += j1Bb03 * j1Ab01;
        cbb32 += j1Bb03 * j1Ab02;
        cbb33 += j1Bb03 * j1Ab03;
        cba00 += j1Ba00 * j1Ab00;
        cba01 += j1Ba00 * j1Ab01;
        cba02 += j1Ba00 * j1Ab02;
        cba03 += j1Ba00 * j1Ab03;
        cba10 += j1Ba01 * j1Ab00;
        cba11 += j1Ba01 * j1Ab01;
        cba12 += j1Ba01 * j1Ab02;
        cba13 += j1Ba01 * j1Ab03;
        cba20 += j1Ba02 * j1Ab00;
        cba21 += j1Ba02 * j1Ab01;
        cba22 += j1Ba02 * j1Ab02;
        cba23 += j1Ba02 * j1Ab03;
        cba30 += j1Ba03 * j1Ab00;
        cba31 += j1Ba03 * j1Ab01;
        cba32 += j1Ba03 * j1Ab02;
        cba33 += j1Ba03 * j1Ab03;

        a_b_shm[write_offset + 0 * 64 + 0] = loadX00;
        a_b_shm[write_offset + 0 * 64 + 1] = loadX01;
        a_b_shm[write_offset + 0 * 64 + 2] = loadX02;
        a_b_shm[write_offset + 0 * 64 + 3] = loadX03;
        cab00 += j1Bb00 * j1Aa00;
        cab01 += j1Bb00 * j1Aa01;
        a_b_shm[write_offset + 2 * 64 + 0] = loadX20;
        a_b_shm[write_offset + 2 * 64 + 1] = loadX21;
        a_b_shm[write_offset + 2 * 64 + 2] = loadX22;
        a_b_shm[write_offset + 2 * 64 + 3] = loadX23;
        cab02 += j1Bb00 * j1Aa02;
        cab03 += j1Bb00 * j1Aa03;
        cab10 += j1Bb01 * j1Aa00;
        cab11 += j1Bb01 * j1Aa01;
        cab12 += j1Bb01 * j1Aa02;
        cab13 += j1Bb01 * j1Aa03;
        cab20 += j1Bb02 * j1Aa00;
        cab21 += j1Bb02 * j1Aa01;
        cab22 += j1Bb02 * j1Aa02;
        cab23 += j1Bb02 * j1Aa03;
        cab30 += j1Bb03 * j1Aa00;
        cab31 += j1Bb03 * j1Aa01;
        cab32 += j1Bb03 * j1Aa02;
        cab33 += j1Bb03 * j1Aa03;
        caa00 += j1Ba00 * j1Aa00;
        caa01 += j1Ba00 * j1Aa01;
        caa02 += j1Ba00 * j1Aa02;
        caa03 += j1Ba00 * j1Aa03;
        caa10 += j1Ba01 * j1Aa00;
        caa11 += j1Ba01 * j1Aa01;
        caa12 += j1Ba01 * j1Aa02;
        caa13 += j1Ba01 * j1Aa03;
        caa20 += j1Ba02 * j1Aa00;
        caa21 += j1Ba02 * j1Aa01;
        caa22 += j1Ba02 * j1Aa02;
        caa23 += j1Ba02 * j1Aa03;
        caa30 += j1Ba03 * j1Aa00;
        caa31 += j1Ba03 * j1Aa01;
        caa32 += j1Ba03 * j1Aa02;
        caa33 += j1Ba03 * j1Aa03;


        cbb00 += j0Bb00 * j0Ab00;
        cbb01 += j0Bb00 * j0Ab01;
        j1Ab00 = a_b_shm[readAs + 7 * 64 + 0];
        j1Ab01 = a_b_shm[readAs + 7 * 64 + 1];
        j1Ab02 = a_b_shm[readAs + 7 * 64 + 2];
        j1Ab03 = a_b_shm[readAs + 7 * 64 + 3];
        cbb02 += j0Bb00 * j0Ab02;
        cbb03 += j0Bb00 * j0Ab03;
        j1Bb00 = a_b_shm[readBs + 7 * 64 + 0];
        j1Bb01 = a_b_shm[readBs + 7 * 64 + 1];
        j1Bb02 = a_b_shm[readBs + 7 * 64 + 2];
        j1Bb03 = a_b_shm[readBs + 7 * 64 + 3];
        cbb10 += j0Bb01 * j0Ab00;
        cbb11 += j0Bb01 * j0Ab01;
        j1Aa00 = a_b_shm[readAs + 7 * 64 + 32 + 0];
        j1Aa01 = a_b_shm[readAs + 7 * 64 + 32 + 1];
        j1Aa02 = a_b_shm[readAs + 7 * 64 + 32 + 2];
        j1Aa03 = a_b_shm[readAs + 7 * 64 + 32 + 3];
        cbb12 += j0Bb01 * j0Ab02;
        cbb13 += j0Bb01 * j0Ab03;
        j1Ba00 = a_b_shm[readBs + 7 * 64 + 32 + 0];
        j1Ba01 = a_b_shm[readBs + 7 * 64 + 32 + 1];
        j1Ba02 = a_b_shm[readBs + 7 * 64 + 32 + 2];
        j1Ba03 = a_b_shm[readBs + 7 * 64 + 32 + 3];

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

        a_b_shm[write_offset + 4 * 64 + 0] = loadX40;
        a_b_shm[write_offset + 4 * 64 + 1] = loadX41;
        a_b_shm[write_offset + 4 * 64 + 2] = loadX42;
        a_b_shm[write_offset + 4 * 64 + 3] = loadX43;
        cab00 += j0Bb00 * j0Aa00;
        cab01 += j0Bb00 * j0Aa01;
        a_b_shm[write_offset + 6 * 64 + 0] = loadX60;
        a_b_shm[write_offset + 6 * 64 + 1] = loadX61;
        a_b_shm[write_offset + 6 * 64 + 2] = loadX62;
        a_b_shm[write_offset + 6 * 64 + 3] = loadX63;
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

        __syncthreads();
        readAs ^= 16 * 64;
        readBs ^= 16 * 64;
        write_offset ^= 16 * 64;

        cbb00 += j1Bb00 * j1Ab00;
        cbb01 += j1Bb00 * j1Ab01;
        j0Ab00 = a_b_shm[readAs + 0 * 64 + 0];
        j0Ab01 = a_b_shm[readAs + 0 * 64 + 1];
        j0Ab02 = a_b_shm[readAs + 0 * 64 + 2];
        j0Ab03 = a_b_shm[readAs + 0 * 64 + 3];
        cbb02 += j1Bb00 * j1Ab02;
        cbb03 += j1Bb00 * j1Ab03;
        j0Bb00 = a_b_shm[readBs + 0 * 64 + 0];
        j0Bb01 = a_b_shm[readBs + 0 * 64 + 1];
        j0Bb02 = a_b_shm[readBs + 0 * 64 + 2];
        j0Bb03 = a_b_shm[readBs + 0 * 64 + 3];
        cbb10 += j1Bb01 * j1Ab00;
        cbb11 += j1Bb01 * j1Ab01;
        j0Aa00 = a_b_shm[readAs + 0 * 64 + 32 + 0];
        j0Aa01 = a_b_shm[readAs + 0 * 64 + 32 + 1];
        j0Aa02 = a_b_shm[readAs + 0 * 64 + 32 + 2];
        j0Aa03 = a_b_shm[readAs + 0 * 64 + 32 + 3];
        cbb12 += j1Bb01 * j1Ab02;
        cbb13 += j1Bb01 * j1Ab03;
        j0Ba00 = a_b_shm[readBs + 0 * 64 + 32 + 0];
        j0Ba01 = a_b_shm[readBs + 0 * 64 + 32 + 1];
        j0Ba02 = a_b_shm[readBs + 0 * 64 + 32 + 2];
        j0Ba03 = a_b_shm[readBs + 0 * 64 + 32 + 3];

        cbb20 += j1Bb02 * j1Ab00;
        cbb21 += j1Bb02 * j1Ab01;
        cbb22 += j1Bb02 * j1Ab02;
        cbb23 += j1Bb02 * j1Ab03;
        cbb30 += j1Bb03 * j1Ab00;
        cbb31 += j1Bb03 * j1Ab01;
        cbb32 += j1Bb03 * j1Ab02;
        cbb33 += j1Bb03 * j1Ab03;
        cba00 += j1Ba00 * j1Ab00;
        cba01 += j1Ba00 * j1Ab01;
        cba02 += j1Ba00 * j1Ab02;
        cba03 += j1Ba00 * j1Ab03;
        cba10 += j1Ba01 * j1Ab00;
        cba11 += j1Ba01 * j1Ab01;
        cba12 += j1Ba01 * j1Ab02;
        cba13 += j1Ba01 * j1Ab03;
        cba20 += j1Ba02 * j1Ab00;
        cba21 += j1Ba02 * j1Ab01;
        cba22 += j1Ba02 * j1Ab02;
        cba23 += j1Ba02 * j1Ab03;
        cba30 += j1Ba03 * j1Ab00;
        cba31 += j1Ba03 * j1Ab01;
        cba32 += j1Ba03 * j1Ab02;
        cba33 += j1Ba03 * j1Ab03;
        cab00 += j1Bb00 * j1Aa00;
        cab01 += j1Bb00 * j1Aa01;
        cab02 += j1Bb00 * j1Aa02;
        cab03 += j1Bb00 * j1Aa03;
        cab10 += j1Bb01 * j1Aa00;
        cab11 += j1Bb01 * j1Aa01;
        cab12 += j1Bb01 * j1Aa02;
        cab13 += j1Bb01 * j1Aa03;
        cab20 += j1Bb02 * j1Aa00;
        cab21 += j1Bb02 * j1Aa01;
        cab22 += j1Bb02 * j1Aa02;
        cab23 += j1Bb02 * j1Aa03;
        cab30 += j1Bb03 * j1Aa00;
        cab31 += j1Bb03 * j1Aa01;
        cab32 += j1Bb03 * j1Aa02;
        cab33 += j1Bb03 * j1Aa03;
        caa00 += j1Ba00 * j1Aa00;
        caa01 += j1Ba00 * j1Aa01;
        caa02 += j1Ba00 * j1Aa02;
        caa03 += j1Ba00 * j1Aa03;
        caa10 += j1Ba01 * j1Aa00;
        caa11 += j1Ba01 * j1Aa01;
        caa12 += j1Ba01 * j1Aa02;
        caa13 += j1Ba01 * j1Aa03;
        caa20 += j1Ba02 * j1Aa00;
        caa21 += j1Ba02 * j1Aa01;
        caa22 += j1Ba02 * j1Aa02;
        caa23 += j1Ba02 * j1Aa03;
        caa30 += j1Ba03 * j1Aa00;
        caa31 += j1Ba03 * j1Aa01;
        caa32 += j1Ba03 * j1Aa02;
        caa33 += j1Ba03 * j1Aa03;

        track0 += 8 * ldx;
        track2 += 8 * ldx;
        track4 += 8 * ldx;
        track6 += 8 * ldx;
    }
    __syncthreads();

    // j0Ab00 = a_b_shm[readAs + 0];
    // j0Ab01 = a_b_shm[readAs + 1];
    // j0Ab02 = a_b_shm[readAs + 2];
    // j0Ab03 = a_b_shm[readAs + 3];

    // j0Bb00 = a_b_shm[readBs + 0];
    // j0Bb01 = a_b_shm[readBs + 1];
    // j0Bb02 = a_b_shm[readBs + 2];
    // j0Bb03 = a_b_shm[readBs + 3];

    // j0Aa00 = a_b_shm[readAs + 32 + 0];
    // j0Aa01 = a_b_shm[readAs + 32 + 1];
    // j0Aa02 = a_b_shm[readAs + 32 + 2];
    // j0Aa03 = a_b_shm[readAs + 32 + 3];

    // j0Ba00 = a_b_shm[readBs + 32 + 0];
    // j0Ba01 = a_b_shm[readBs + 32 + 1];
    // j0Ba02 = a_b_shm[readBs + 32 + 2];
    // j0Ba03 = a_b_shm[readBs + 32 + 3];

    // for (int j = 0; j < 7; ++j) {
    //     int prefetch = (j + 1) % 8;

    //     if (!(j & 1)) {

    //         cbb00 += j0Bb00 * j0Ab00;
    //         cbb01 += j0Bb00 * j0Ab01;
    //         j1Ab00 = a_b_shm[readAs + prefetch * 64 + 0];
    //         j1Ab01 = a_b_shm[readAs + prefetch * 64 + 1];
    //         j1Ab02 = a_b_shm[readAs + prefetch * 64 + 2];
    //         j1Ab03 = a_b_shm[readAs + prefetch * 64 + 3];
    //         cbb02 += j0Bb00 * j0Ab02;
    //         cbb03 += j0Bb00 * j0Ab03;
    //         j1Bb00 = a_b_shm[readBs + prefetch * 64 + 0];
    //         j1Bb01 = a_b_shm[readBs + prefetch * 64 + 1];
    //         j1Bb02 = a_b_shm[readBs + prefetch * 64 + 2];
    //         j1Bb03 = a_b_shm[readBs + prefetch * 64 + 3];

    //         cbb10 += j0Bb01 * j0Ab00;
    //         cbb11 += j0Bb01 * j0Ab01;
    //         j1Aa00 = a_b_shm[readAs + prefetch * 64 + 32 + 0];
    //         j1Aa01 = a_b_shm[readAs + prefetch * 64 + 32 + 1];
    //         j1Aa02 = a_b_shm[readAs + prefetch * 64 + 32 + 2];
    //         j1Aa03 = a_b_shm[readAs + prefetch * 64 + 32 + 3];
    //         cbb12 += j0Bb01 * j0Ab02;
    //         cbb13 += j0Bb01 * j0Ab03;
    //         j1Ba00 = a_b_shm[readBs + prefetch * 64 + 32 + 0];
    //         j1Ba01 = a_b_shm[readBs + prefetch * 64 + 32 + 1];
    //         j1Ba02 = a_b_shm[readBs + prefetch * 64 + 32 + 2];
    //         j1Ba03 = a_b_shm[readBs + prefetch * 64 + 32 + 3];

    //         cbb20 += j0Bb02 * j0Ab00;
    //         cbb21 += j0Bb02 * j0Ab01;
    //         cbb22 += j0Bb02 * j0Ab02;
    //         cbb23 += j0Bb02 * j0Ab03;

    //         cbb30 += j0Bb03 * j0Ab00;
    //         cbb31 += j0Bb03 * j0Ab01;
    //         cbb32 += j0Bb03 * j0Ab02;
    //         cbb33 += j0Bb03 * j0Ab03;

    //         cba00 += j0Ba00 * j0Ab00;
    //         cba01 += j0Ba00 * j0Ab01;
    //         cba02 += j0Ba00 * j0Ab02;
    //         cba03 += j0Ba00 * j0Ab03;

    //         cba10 += j0Ba01 * j0Ab00;
    //         cba11 += j0Ba01 * j0Ab01;
    //         cba12 += j0Ba01 * j0Ab02;
    //         cba13 += j0Ba01 * j0Ab03;

    //         cba20 += j0Ba02 * j0Ab00;
    //         cba21 += j0Ba02 * j0Ab01;
    //         cba22 += j0Ba02 * j0Ab02;
    //         cba23 += j0Ba02 * j0Ab03;

    //         cba30 += j0Ba03 * j0Ab00;
    //         cba31 += j0Ba03 * j0Ab01;
    //         cba32 += j0Ba03 * j0Ab02;
    //         cba33 += j0Ba03 * j0Ab03;

    //         cab00 += j0Bb00 * j0Aa00;
    //         cab01 += j0Bb00 * j0Aa01;
    //         cab02 += j0Bb00 * j0Aa02;
    //         cab03 += j0Bb00 * j0Aa03;

    //         cab10 += j0Bb01 * j0Aa00;
    //         cab11 += j0Bb01 * j0Aa01;
    //         cab12 += j0Bb01 * j0Aa02;
    //         cab13 += j0Bb01 * j0Aa03;

    //         cab20 += j0Bb02 * j0Aa00;
    //         cab21 += j0Bb02 * j0Aa01;
    //         cab22 += j0Bb02 * j0Aa02;
    //         cab23 += j0Bb02 * j0Aa03;

    //         cab30 += j0Bb03 * j0Aa00;
    //         cab31 += j0Bb03 * j0Aa01;
    //         cab32 += j0Bb03 * j0Aa02;
    //         cab33 += j0Bb03 * j0Aa03;

    //         caa00 += j0Ba00 * j0Aa00;
    //         caa01 += j0Ba00 * j0Aa01;
    //         caa02 += j0Ba00 * j0Aa02;
    //         caa03 += j0Ba00 * j0Aa03;

    //         caa10 += j0Ba01 * j0Aa00;
    //         caa11 += j0Ba01 * j0Aa01;
    //         caa12 += j0Ba01 * j0Aa02;
    //         caa13 += j0Ba01 * j0Aa03;

    //         caa20 += j0Ba02 * j0Aa00;
    //         caa21 += j0Ba02 * j0Aa01;
    //         caa22 += j0Ba02 * j0Aa02;
    //         caa23 += j0Ba02 * j0Aa03;

    //         caa30 += j0Ba03 * j0Aa00;
    //         caa31 += j0Ba03 * j0Aa01;
    //         caa32 += j0Ba03 * j0Aa02;
    //         caa33 += j0Ba03 * j0Aa03;
    //     } else {
    //         // j1Ab00 = a_b_shm[readAs + prefetch * 64 + 0];
    //         // j1Ab01 = a_b_shm[readAs + prefetch * 64 + 1];
    //         // j1Ab02 = a_b_shm[readAs + prefetch * 64 + 2];
    //         // j1Ab03 = a_b_shm[readAs + prefetch * 64 + 3];

    //         // j1Bb00 = a_b_shm[readBs + prefetch * 64 + 0];
    //         // j1Bb01 = a_b_shm[readBs + prefetch * 64 + 1];
    //         // j1Bb02 = a_b_shm[readBs + prefetch * 64 + 2];
    //         // j1Bb03 = a_b_shm[readBs + prefetch * 64 + 3];

    //         // j1Aa00 = a_b_shm[readAs + prefetch * 64 + 32 + 0];
    //         // j1Aa01 = a_b_shm[readAs + prefetch * 64 + 32 + 1];
    //         // j1Aa02 = a_b_shm[readAs + prefetch * 64 + 32 + 2];
    //         // j1Aa03 = a_b_shm[readAs + prefetch * 64 + 32 + 3];

    //         // j1Ba00 = a_b_shm[readBs + prefetch * 64 + 32 + 0];
    //         // j1Ba01 = a_b_shm[readBs + prefetch * 64 + 32 + 1];
    //         // j1Ba02 = a_b_shm[readBs + prefetch * 64 + 32 + 2];
    //         // j1Ba03 = a_b_shm[readBs + prefetch * 64 + 32 + 3];

    //         cbb00 += j1Bb00 * j1Ab00;
    //         cbb01 += j1Bb00 * j1Ab01;
    //         j0Ab00 = a_b_shm[readAs + prefetch * 64 + 0];
    //         j0Ab01 = a_b_shm[readAs + prefetch * 64 + 1];
    //         j0Ab02 = a_b_shm[readAs + prefetch * 64 + 2];
    //         j0Ab03 = a_b_shm[readAs + prefetch * 64 + 3];
    //         cbb02 += j1Bb00 * j1Ab02;
    //         cbb03 += j1Bb00 * j1Ab03;
    //         j0Bb00 = a_b_shm[readBs + prefetch * 64 + 0];
    //         j0Bb01 = a_b_shm[readBs + prefetch * 64 + 1];
    //         j0Bb02 = a_b_shm[readBs + prefetch * 64 + 2];
    //         j0Bb03 = a_b_shm[readBs + prefetch * 64 + 3];

    //         cbb10 += j1Bb01 * j1Ab00;
    //         cbb11 += j1Bb01 * j1Ab01;
    //         j0Aa00 = a_b_shm[readAs + prefetch * 64 + 32 + 0];
    //         j0Aa01 = a_b_shm[readAs + prefetch * 64 + 32 + 1];
    //         j0Aa02 = a_b_shm[readAs + prefetch * 64 + 32 + 2];
    //         j0Aa03 = a_b_shm[readAs + prefetch * 64 + 32 + 3];
    //         cbb12 += j1Bb01 * j1Ab02;
    //         cbb13 += j1Bb01 * j1Ab03;
    //         j0Ba00 = a_b_shm[readBs + prefetch * 64 + 32 + 0];
    //         j0Ba01 = a_b_shm[readBs + prefetch * 64 + 32 + 1];
    //         j0Ba02 = a_b_shm[readBs + prefetch * 64 + 32 + 2];
    //         j0Ba03 = a_b_shm[readBs + prefetch * 64 + 32 + 3];

    //         cbb20 += j1Bb02 * j1Ab00;
    //         cbb21 += j1Bb02 * j1Ab01;
    //         cbb22 += j1Bb02 * j1Ab02;
    //         cbb23 += j1Bb02 * j1Ab03;

    //         cbb30 += j1Bb03 * j1Ab00;
    //         cbb31 += j1Bb03 * j1Ab01;
    //         cbb32 += j1Bb03 * j1Ab02;
    //         cbb33 += j1Bb03 * j1Ab03;

    //         cba00 += j1Ba00 * j1Ab00;
    //         cba01 += j1Ba00 * j1Ab01;
    //         cba02 += j1Ba00 * j1Ab02;
    //         cba03 += j1Ba00 * j1Ab03;

    //         cba10 += j1Ba01 * j1Ab00;
    //         cba11 += j1Ba01 * j1Ab01;
    //         cba12 += j1Ba01 * j1Ab02;
    //         cba13 += j1Ba01 * j1Ab03;

    //         cba20 += j1Ba02 * j1Ab00;
    //         cba21 += j1Ba02 * j1Ab01;
    //         cba22 += j1Ba02 * j1Ab02;
    //         cba23 += j1Ba02 * j1Ab03;

    //         cba30 += j1Ba03 * j1Ab00;
    //         cba31 += j1Ba03 * j1Ab01;
    //         cba32 += j1Ba03 * j1Ab02;
    //         cba33 += j1Ba03 * j1Ab03;

    //         cab00 += j1Bb00 * j1Aa00;
    //         cab01 += j1Bb00 * j1Aa01;
    //         cab02 += j1Bb00 * j1Aa02;
    //         cab03 += j1Bb00 * j1Aa03;

    //         cab10 += j1Bb01 * j1Aa00;
    //         cab11 += j1Bb01 * j1Aa01;
    //         cab12 += j1Bb01 * j1Aa02;
    //         cab13 += j1Bb01 * j1Aa03;

    //         cab20 += j1Bb02 * j1Aa00;
    //         cab21 += j1Bb02 * j1Aa01;
    //         cab22 += j1Bb02 * j1Aa02;
    //         cab23 += j1Bb02 * j1Aa03;

    //         cab30 += j1Bb03 * j1Aa00;
    //         cab31 += j1Bb03 * j1Aa01;
    //         cab32 += j1Bb03 * j1Aa02;
    //         cab33 += j1Bb03 * j1Aa03;

    //         caa00 += j1Ba00 * j1Aa00;
    //         caa01 += j1Ba00 * j1Aa01;
    //         caa02 += j1Ba00 * j1Aa02;
    //         caa03 += j1Ba00 * j1Aa03;

    //         caa10 += j1Ba01 * j1Aa00;
    //         caa11 += j1Ba01 * j1Aa01;
    //         caa12 += j1Ba01 * j1Aa02;
    //         caa13 += j1Ba01 * j1Aa03;

    //         caa20 += j1Ba02 * j1Aa00;
    //         caa21 += j1Ba02 * j1Aa01;
    //         caa22 += j1Ba02 * j1Aa02;
    //         caa23 += j1Ba02 * j1Aa03;

    //         caa30 += j1Ba03 * j1Aa00;
    //         caa31 += j1Ba03 * j1Aa01;
    //         caa32 += j1Ba03 * j1Aa02;
    //         caa33 += j1Ba03 * j1Aa03;
    //     }

    // }

    // cbb00 += j1Bb00 * j1Ab00;
    // cbb01 += j1Bb00 * j1Ab01;
    // cbb02 += j1Bb00 * j1Ab02;
    // cbb03 += j1Bb00 * j1Ab03;
    // cbb10 += j1Bb01 * j1Ab00;
    // cbb11 += j1Bb01 * j1Ab01;
    // cbb12 += j1Bb01 * j1Ab02;
    // cbb13 += j1Bb01 * j1Ab03;
    // cbb20 += j1Bb02 * j1Ab00;
    // cbb21 += j1Bb02 * j1Ab01;
    // cbb22 += j1Bb02 * j1Ab02;
    // cbb23 += j1Bb02 * j1Ab03;
    // cbb30 += j1Bb03 * j1Ab00;
    // cbb31 += j1Bb03 * j1Ab01;
    // cbb32 += j1Bb03 * j1Ab02;
    // cbb33 += j1Bb03 * j1Ab03;
    // cba00 += j1Ba00 * j1Ab00;
    // cba01 += j1Ba00 * j1Ab01;
    // cba02 += j1Ba00 * j1Ab02;
    // cba03 += j1Ba00 * j1Ab03;
    // cba10 += j1Ba01 * j1Ab00;
    // cba11 += j1Ba01 * j1Ab01;
    // cba12 += j1Ba01 * j1Ab02;
    // cba13 += j1Ba01 * j1Ab03;
    // cba20 += j1Ba02 * j1Ab00;
    // cba21 += j1Ba02 * j1Ab01;
    // cba22 += j1Ba02 * j1Ab02;
    // cba23 += j1Ba02 * j1Ab03;
    // cba30 += j1Ba03 * j1Ab00;
    // cba31 += j1Ba03 * j1Ab01;
    // cba32 += j1Ba03 * j1Ab02;
    // cba33 += j1Ba03 * j1Ab03;
    // cab00 += j1Bb00 * j1Aa00;
    // cab01 += j1Bb00 * j1Aa01;
    // cab02 += j1Bb00 * j1Aa02;
    // cab03 += j1Bb00 * j1Aa03;
    // cab10 += j1Bb01 * j1Aa00;
    // cab11 += j1Bb01 * j1Aa01;
    // cab12 += j1Bb01 * j1Aa02;
    // cab13 += j1Bb01 * j1Aa03;
    // cab20 += j1Bb02 * j1Aa00;
    // cab21 += j1Bb02 * j1Aa01;
    // cab22 += j1Bb02 * j1Aa02;
    // cab23 += j1Bb02 * j1Aa03;
    // cab30 += j1Bb03 * j1Aa00;
    // cab31 += j1Bb03 * j1Aa01;
    // cab32 += j1Bb03 * j1Aa02;
    // cab33 += j1Bb03 * j1Aa03;
    // caa00 += j1Ba00 * j1Aa00;
    // caa01 += j1Ba00 * j1Aa01;
    // caa02 += j1Ba00 * j1Aa02;
    // caa03 += j1Ba00 * j1Aa03;
    // caa10 += j1Ba01 * j1Aa00;
    // caa11 += j1Ba01 * j1Aa01;
    // caa12 += j1Ba01 * j1Aa02;
    // caa13 += j1Ba01 * j1Aa03;
    // caa20 += j1Ba02 * j1Aa00;
    // caa21 += j1Ba02 * j1Aa01;
    // caa22 += j1Ba02 * j1Aa02;
    // caa23 += j1Ba02 * j1Aa03;
    // caa30 += j1Ba03 * j1Aa00;
    // caa31 += j1Ba03 * j1Aa01;
    // caa32 += j1Ba03 * j1Aa02;
    // caa33 += j1Ba03 * j1Aa03;

    int tid31 = tid & 31;
    int tid32 = tid & 32;
    int coord_x = readBs & 0x7f;
    int coord_y = readAs & 0x7f;
    int writeCs = coord_x / 4 * 64 + coord_y;
    int readCs = (tid32 << 3) + tid31;
    int ldc4 = M * 4;
    int Cy00 = (tid32 >> 1) * M + tid31;
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

__global__ void cuda_kernel_sgemm_100_v2(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta) {
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    float* block_a = a + block_y * 64;
    float* block_b = b + block_x * 64;
    float* block_c = c + block_x * 64 * M + block_y * 64;
    sgemm_block_64x64_v2(block_a, block_b, block_c, M, N, K, alpha, beta);
}

__global__ void sgemm_block_test(
        float *a, float *b, float *c,
        size_t M, size_t N, size_t K,
        float alpha, float beta) {

    __shared__ float a_b_shm[2 * 16 * 64];

    int tid = threadIdx.x & 0x3f;
    int ldx = N;
    int tid2 = (tid >> 4) & 1;
    // int tid15 = tid & 0xf;
    int tid15 = tid & 15;
    int track0 = 8;
    // int track2 = track0 + 2 * ldx;
    // int track4 = track0 + 4 * ldx;
    // int track6 = track0 + 6 * ldx;
    int end = track0 + (K - 8) * ldx;
    int write_offset = tid2 * 64 + tid15 * 4;

    // int readAs = ((tid >> 1) & 7) << 2;
    // int readBs = ((((tid & 0x30) >> 3) | (tid & 1)) << 2) + 512;
    int readAs = 2;
    int readBs = 3;

    // float* read_addr = tid >= 32 ? b : a;
    float* read_addr = b;

    while (track0 <= end) {
        a_b_shm[write_offset + 0 * 64 + 0] = read_addr[track0 + 0];
        __syncthreads();

        write_offset ^= 16 * 64;
        track0 += 8 * ldx;
        // track2 += 8 * ldx;
        // track4 += 8 * ldx;
        // track6 += 8 * ldx;

        readAs ^= 16 * 64;
        // readBs ^= 16 * 64;

        a[track0] = a_b_shm[readAs];
        // b[track2] = a_b_shm[readBs];
        // c[track6] = c[track4] + a[write_offset] + b[write_offset];
    }

    // while (track0 < end) {
    //     a_b_shm[write_offset + 2 * 64 + 0] = read_addr[track2 + 0];
    //     __syncwarp(0xFFFFFFFF);

    //     write_offset ^= 16 * 64;
    //     readAs ^= 16 * 64;
    //     readBs ^= 16 * 64;

    //     a[track0] = a_b_shm[readAs];
    //     b[track2] = a_b_shm[readBs];
    //     c[track6] = c[track4] + a[write_offset] + b[write_offset];
    // }

}
