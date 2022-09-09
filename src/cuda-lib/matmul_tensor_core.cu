#include "matmul_kernel.h"

using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_sgemm_kernel(half *a, half *b, float *c, float *d,
        int M, int N, int K, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = N;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;
//        int bRow = i;
//        int bCol = warpN * WMMA_N;
        int bCol = i;
        int bRow = warpN * WMMA_N;
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
//            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, lda);
//            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, ldb);
            wmma::load_matrix_sync(a_frag, a + aRow * M + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow * K + bCol, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow * N + cCol, ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cRow * N + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
