// Copyright 2020, Tencent Inc.
// All rights reserved.
//
// @author shaorunwang <shaorunwang@tencent.com>

#include <cuda.h>
#include <cuda_fp16.h>
#include <thrust/extrema.h>

#include <iostream>

//#include "NvInfer.h"
//#include "sparse_fipnn_plugin.h"

#define CUDA_CHECK(condition)                                    \
    /* Code block avoids redefinition of cudaError_t error */    \
    do {                                                         \
        cudaError_t error = condition;                           \
        if (error != cudaSuccess) {                              \
            std::cout << cudaGetErrorString(error) << std::endl; \
        }                                                        \
    } while (0)

#define DIVUP(m, n) (((m) / (n)) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

namespace nvinfer1 {

namespace sparse_fipnn {

template <typename T = float>
__device__ T __reduce_sum_across_warp(T val) {
    T rtn = val;
    __syncwarp(0xFFFFFFFF);
    for (int32_t i = 1; i < 32; i *= 2) {
        rtn += __shfl_xor_sync(0xFFFFFFFF, rtn, i);
    }
    return rtn;
}

template <typename T = float, int32_t warp_num = 32>
__global__ void ProcessCommonPart(int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
                                  int32_t* sample_feature_start_addr, const T* weight_tensor,
                                  const int32_t* field_tensor, T* gmem_fw_cross_mean_sum,
                                  T* gmem_fw_cross_square_sum, int32_t* gmem_fw_field_map) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t global_warp_id = blockIdx.x * warp_num + warp_id;
    int32_t total_global_warp_num = gridDim.x * warp_num;

    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = warp_num * 32;
    int32_t common_feature_num = sample_feature_start_addr[0];

    for (int32_t wid = global_warp_id; wid < common_feature_num; wid += total_global_warp_num) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;

        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;

        if (lane_id == 0) gmem_fw_field_map[fw_field_1] = field_1;

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_size;
#pragma unroll
            for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
                if (n + lane_id < embedding_size) {
                    T reg = weight_tensor[wid * embedding_size * field_num +
                                          field_2 * embedding_size + n + lane_id];
                    atomicAdd(gmem_fw_cross_mean_sum + mem_field_offset + n + lane_id, reg);
                }
            }
        }

        int32_t mem_field_offset = fw_field_1 * embedding_size;
#pragma unroll
        for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
            if (n + lane_id < embedding_size) {
                T reg = weight_tensor[wid * embedding_size * field_num + field_1 * embedding_size +
                                      n + lane_id];
                float square = reg * reg;
                atomicAdd(gmem_fw_cross_square_sum + mem_field_offset + n + lane_id, square);
            }
        }
    }
}

template <typename T = float>
__global__ void BroadcastCommonPart(int32_t batch, int32_t embedding_size, int32_t field_num,
                                    int32_t fw_field_num, T* gmem_fw_cross_mean_sum,
                                    T* gmem_fw_cross_square_sum, T* output) {

    int32_t lane_id = threadIdx.x;
    int32_t fw_field_id = blockIdx.x % fw_field_num;
    int32_t tid = lane_id + fw_field_id * embedding_size;
    int32_t bid = blockIdx.x / fw_field_num;
    T Reg_square = gmem_fw_cross_square_sum[tid];
    output[bid * (embedding_size * (field_num + 1) * fw_field_num) +
           embedding_size * field_num * fw_field_num + tid] = Reg_square;

    T Reg_mean_0 = gmem_fw_cross_mean_sum[tid];
    T Reg_mean_1 = gmem_fw_cross_mean_sum[embedding_size * fw_field_num + tid];
    output[bid * (embedding_size * (field_num + 1) * fw_field_num) + tid] = Reg_mean_0;
    output[bid * (embedding_size * (field_num + 1) * fw_field_num) + embedding_size * fw_field_num +
           tid] = Reg_mean_1;
}

template <typename T = float, int32_t warp_num = 32>
__device__ void ProcessSamplePart(
    int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* gmem_cross_mean_sum,
    T* gmem_cross_square_sum, T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
    int32_t* gmem_common_field_map, int32_t weight_size, int32_t field_num, int32_t fw_field_num,
    int32_t this_sample_feature_num, int32_t this_sample_feature_start_addr, const T* weight_tensor,
    const int32_t* field_tensor, T* smem_output, int32_t shared_mem_elements) {

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = warp_num * 32;

    for (int32_t i = 0; i < fw_field_num; i += warp_num * 32) {
        if (i + tid < fw_field_num) {
            smem_fw_field_map[i + tid] = gmem_common_field_map[i + tid];
        }
    }

    __syncthreads();

    // sample feature phase
    int32_t sample_start_row = warp_id + this_sample_feature_start_addr;
    int32_t sample_end_row = this_sample_feature_num + this_sample_feature_start_addr;
    for (int32_t wid = sample_start_row; wid < sample_end_row; wid += warp_num) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;

        if (lane_id == 0) smem_fw_field_map[fw_field_1] = field_1;

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * weight_size;
#pragma unroll
            for (int32_t n = 0; n < (weight_size + 31) / 32 * 32; n += 32) {
                if (n + lane_id < weight_size) {
                    T reg = weight_tensor[wid * weight_size * field_num + field_2 * weight_size +
                                          n + lane_id];
                    atomicAdd(gmem_cross_mean_sum + mem_field_offset + n + lane_id, reg);
                }
            }
        }
        int32_t mem_field_offset = fw_field_1 * weight_size;
#pragma unroll
        for (int32_t n = 0; n < (weight_size + 31) / 32 * 32; n += 32) {
            if (n + lane_id < weight_size) {
                T reg = weight_tensor[wid * weight_size * field_num + field_1 * weight_size + n +
                                      lane_id];
                float square = reg * reg;
                atomicAdd(gmem_cross_square_sum + mem_field_offset + n + lane_id, square);
            }
        }
    }

    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        int32_t cnt = 0;
        for (int32_t i = 0; i < fw_field_num; i++) {
            if (smem_fw_field_map[i] >= 0) {
                smem_fw_map_idx[cnt++] = i;
            }
        }
        smem_fw_map_idx[fw_field_num] = cnt;
    }
    __syncthreads();
}

template <typename T = float, int32_t warp_num = 32>
__device__ void ProcessOutput(int32_t weight_size, int32_t field_num, int32_t fw_field_num,
                              T* mem_cross_mean_sum, T* mem_cross_square_sum,
                              int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
                              T* output_smem) {

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    int32_t weight_size_pad = (weight_size + 31) / 32 * 32;

    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
         fw_field_1_idx += warp_num) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_fw_field_map[fw_field_1];

        int32_t fw_iter = (2 + fw_field_1) * (fw_field_1 + 1) / 2 - (fw_field_1 + 1);

        T reg_cross_mean_sum_tmp[2][6] = { 0 };  // weight_size <= 192

        for (int32_t n = 0; n < weight_size_pad; n += 32) {
            if (n + lane_id < weight_size) {
                reg_cross_mean_sum_tmp[0][n / 32] =
                    mem_cross_mean_sum[(0 * fw_field_num + fw_field_1) * weight_size + n + lane_id];
                reg_cross_mean_sum_tmp[1][n / 32] =
                    mem_cross_mean_sum[(1 * fw_field_num + fw_field_1) * weight_size + n + lane_id];
            }
        }

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];

            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * weight_size;
            int32_t index_2 = field_2 * weight_size;

            T output_value = T(0);
            for (int32_t n = 0; n < (weight_size + 31) / 32 * 32; n += 32) {
                T reg_index_1 = T(0);
                T reg_index_2 = T(0);

                if (n + lane_id < weight_size) {
                    reg_index_1 = mem_cross_mean_sum[index_1 + n + lane_id];
                    if (field_2 == 0)
                        reg_index_2 = reg_cross_mean_sum_tmp[0][n / 32];
                    else
                        reg_index_2 = reg_cross_mean_sum_tmp[1][n / 32];

                    // reg_index_2 = mem_cross_mean_sum[index_2 + n + lane_id];
                }
                output_value += reg_index_1 * reg_index_2;
            }
            output_value = __reduce_sum_across_warp(output_value);
            // store here
            if (lane_id == 0) {
                output_smem[fw_iter + fw_field_2] = output_value;
            }
            __syncwarp(0xFFFFFFFF);
        }
        T output_value = T(0);
        for (int32_t n = 0; n < (weight_size + 31) / 32 * 32; n += 32) {
            int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * weight_size;
            T reg_mean = T(0);
            T reg_square = T(0);

            if (n + lane_id < weight_size) {
                reg_mean = mem_cross_mean_sum[index_1 + n + lane_id];
                reg_square = mem_cross_square_sum[fw_field_1 * weight_size + n + lane_id];
            }
            output_value += T(0.5) * (reg_mean * reg_mean - reg_square);
        }
        output_value = __reduce_sum_across_warp(output_value);
        // store here
        if (lane_id == 0) output_smem[fw_iter + fw_field_1] = output_value;
        __syncwarp(0xFFFFFFFF);
    }
}

__global__ void ComputeBatchBoundary(const int32_t* index_tensor, int32_t total_feature_num,
                                     int32_t batch_size, int32_t* sample_feature_start_addr,
                                     int32_t* sample_feature_end_addr) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_feature_num) {
        int32_t idx = index_tensor[tid];
        // atomicMin(sample_feature_start_addr + idx, tid);
        // atomicMax(sample_feature_end_addr + idx, tid + 1);
        if (tid > 0) {
            int32_t pre_idx = index_tensor[tid - 1];
            for (int32_t i = idx; i > pre_idx; --i) {
                sample_feature_start_addr[i] = tid;
            }
        } else {
            int32_t first_idx = index_tensor[0];
            for (int32_t i = 0; i <= first_idx; ++i) {
                sample_feature_start_addr[i] = 0;
            }
            int32_t last_idx = index_tensor[total_feature_num - 1];
            for (int32_t i = batch_size - 1; i > last_idx; --i) {
                sample_feature_start_addr[i] = total_feature_num;
            }
            sample_feature_start_addr[batch_size] = total_feature_num;
        }
    }
}

template <typename T = float, int32_t warp_num = 32>
__global__ void SparseFIPNNGpu(int32_t weight_size, int32_t field_num, int32_t fw_field_num,
                               int32_t* sample_feature_start_addr, int32_t* sample_feature_end_addr,
                               T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
                               int32_t* gmem_common_field_map, const T* weight_tensor,
                               const int32_t* field_tensor, T* output_tensor,
                               T* workspace  // for mean_sum and square_sum
)

{
    int32_t batch_id = blockIdx.x;
    int32_t embedding_size = fw_field_num * (fw_field_num + 1) / 2;

    extern __shared__ float smem_pool[];

    int32_t* smem_fw_field_map = reinterpret_cast<int*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + fw_field_num;
    T* gmem_output = output_tensor + batch_id * embedding_size;

    // Use global memory in case of lacking atomicAdd float in shared mem
    T* mem_cross_mean_sum = workspace + batch_id * (weight_size * (field_num + 1) * fw_field_num);
    T* mem_cross_square_sum = mem_cross_mean_sum + weight_size * field_num * fw_field_num;

    int this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
    int this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
    int this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

    ProcessSamplePart<T, warp_num>(
        smem_fw_field_map, smem_fw_map_idx, mem_cross_mean_sum, mem_cross_square_sum,
        gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_common_field_map,
        weight_size, field_num, fw_field_num, this_sample_feature_num,
        this_sample_feature_start_addr, weight_tensor, field_tensor, gmem_output, embedding_size);

    ProcessOutput<T, warp_num>(weight_size, field_num, fw_field_num, mem_cross_mean_sum,
                               mem_cross_square_sum, smem_fw_field_map, smem_fw_map_idx,
                               gmem_output);
    // postprocess<T>(smem_output, output_tensor + batch_id * embedding_size, embedding_size);
}

namespace functor {

template <typename T>
int32_t ComputeSparseFipnn(cudaStream_t stream, const void* const* input, T* output,
                           void* workspace, const int32_t fw_field_num,
                           const int32_t sample_feature_size, const int32_t field_num,
                           const int32_t field_neuron_size, const int32_t batch_size) {
    const T* weight_tensor = static_cast<const T*>(input[0]);
    const int32_t* field_tensor = static_cast<const int32_t*>(input[1]);
    const int32_t* index_tensor = static_cast<const int32_t*>(input[2]);
    int32_t* sample_feature_start_addr = static_cast<int32_t*>(workspace);
    int32_t* sample_feature_end_addr = nullptr;
    int32_t* gmem_field_map = sample_feature_start_addr + batch_size + 1;
    T* gmem_cross_sum = reinterpret_cast<T*>(gmem_field_map + fw_field_num);
    T* gmem_common_cross_mean_sum = gmem_cross_sum + batch_size * (field_neuron_size * (field_num + 1) * fw_field_num);
    T* gmem_common_cross_square_sum = gmem_common_cross_mean_sum + field_neuron_size * field_num * fw_field_num;

    int32_t embedding_size = fw_field_num * (fw_field_num + 1) / 2;
    int32_t shared_mem_required_bytes = (fw_field_num * 2 + 1) * sizeof(int);
    dim3 block(32, 32);
    dim3 grid(batch_size);
    cudaMemsetAsync(gmem_common_cross_mean_sum, 0,
                    sizeof(float) * (field_neuron_size * field_num * fw_field_num +
                                     field_neuron_size * fw_field_num),
                    stream);
    cudaMemsetAsync(gmem_field_map, -1, sizeof(int) * (fw_field_num), stream);
    cudaMemsetAsync(sample_feature_start_addr, 0, sizeof(int) * (batch_size + 1), stream);
    cudaMemsetAsync(output, 0, sizeof(float) * (batch_size * embedding_size), stream);

    ComputeBatchBoundary<<<DIVUP(sample_feature_size, 1024), 1024, 0, stream>>>(
        index_tensor, sample_feature_size, batch_size, sample_feature_start_addr,
        sample_feature_end_addr);

    dim3 grid0(1);
    ProcessCommonPart<T, 32><<<grid0, block, 0, stream>>>(
        field_neuron_size, field_num, fw_field_num, sample_feature_start_addr, weight_tensor,
        field_tensor, gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_field_map);
    dim3 block_set(field_neuron_size);
    dim3 grid_set(batch_size * fw_field_num);
    BroadcastCommonPart<T>
        <<<grid_set, block_set, 0, stream>>>(
        batch_size, field_neuron_size, field_num, fw_field_num, gmem_common_cross_mean_sum,
        gmem_common_cross_square_sum, gmem_cross_sum);

    SparseFIPNNGpu<T, 32><<<grid, block, shared_mem_required_bytes, stream>>>(
        field_neuron_size, field_num, fw_field_num, sample_feature_start_addr,
        sample_feature_end_addr, gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_field_map,
        reinterpret_cast<const T*>(weight_tensor), field_tensor, output, gmem_cross_sum);

    return 1;
}

template int32_t ComputeSparseFipnn(cudaStream_t stream, const void* const* input, float* output,
                                    void* workspace, const int32_t fw_field_num,
                                    const int32_t sample_feature_size, const int32_t field_num,
                                    const int32_t field_neuron_size, const int32_t batch_size);

template int32_t ComputeSparseFipnn(cudaStream_t stream, const void* const* input, half* output,
                                    void* workspace, const int32_t fw_field_num,
                                    const int32_t sample_feature_size, const int32_t field_num,
                                    const int32_t field_neuron_size, const int32_t batch_size);

}  // namespace functor
}  // namespace sparse_fipnn
}  // namespace nvinfer1
