#include "cutlass/gemm/device/gemm.h"
#include "matmul_kernel.h"

// #define USE_TENSOR_CORE

cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb,
        float beta, float* C, int ldc, int cycle_count) {
    // for (int i = 0; i < cycle_count; ++i) {

#ifndef USE_TENSOR_CORE

    using CutlassGemm = cutlass::gemm::device::Gemm<float, // Data-type of A matrix
            cutlass::layout::RowMajor,                     // Layout of A matrix
            float,                                         // Data-type of B matrix
            cutlass::layout::RowMajor,                     // Layout of B matrix
            float,                                         // Data-type of C matrix
            cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 8>, cutlass::gemm::GemmShape<64, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>
            // cutlass::gemm::GemmShape<32, 32, 8>,
            // cutlass::gemm::GemmShape<16, 16, 8>,
            // cutlass::gemm::GemmShape<1, 1, 1>,
            >;

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, // Gemm Problem dimensions
            {A, lda},                      // Tensor-ref for source matrix A
            {B, ldb},                      // Tensor-ref for source matrix B
            {C, ldc},                      // Tensor-ref for source matrix C
            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
            {alpha, beta}); // Scalars used in the Epilogue

#else

    using ElementAccumulator = float;                  // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
    using ElementOutput = float;                       // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices. Column Major for
    // Matrix A, Row Major for Matrix B and Row Major for Matrix C
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>; // <- threadblock tile M = 128, N = 128,
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>; // <- warp tile M = 64, N = 64, K = 16
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, // <- data type of output matrix
            128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                              // memory access. For a byte, it's 16
                                                              // elements. This becomes the vector width of
                                                              // math instructions in the epilogue too
            ElementAccumulator,                               // <- data type of accumulator
            ElementComputeEpilogue>; // <- data type for alpha/beta in linear combination function

    // Number of pipelines you want to use
    constexpr int NumStages = 4;

    using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
            ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K}, // Gemm Problem dimensions
            {(ElementInputA*)A, lda},      // Tensor-ref for source matrix A
            {(ElementInputB*)B, ldb},      // Tensor-ref for source matrix B
            {C, ldc},                      // Tensor-ref for source matrix C
            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
            {alpha, beta}); // Scalars used in the Epilogue

#endif

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    // }

    return cudaSuccess;
}
