// Copyright 2020, Tencent Inc.
// All rights reserved.
//
// @author shaorunwang <shaorunwang@tencent.com>
#include <cuda.h>
#include <cuda_fp16.h>
#include <thrust/extrema.h>

#include <fstream>
#include <iostream>

//#include "NvInfer.h"
//#include "sparse_fwffm_plugin.h"

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

namespace sparse_fwffm {

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
    // printf("bid : %d, tid: %d\n",bid, tid);
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
T* gmem_cross_square_sum, T* gmem_fw_cross_mean_sum, T* gmem_fw_cross_square_sum,
int32_t* gmem_fw_field_map, int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
int32_t this_sample_feature_num, int32_t this_sample_feature_start_addr,
int32_t sample_0_feature_num, int32_t sample_0_feature_start_addr, const T* weight_tensor,
const int* field_tensor, int32_t shared_mem_elements) {

    constexpr int32_t total_thread = warp_num * 32;

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t tid = threadIdx.x + threadIdx.y * 32;

    for (int32_t i = 0; i < fw_field_num; i += warp_num * 32) {
        if (i + tid < fw_field_num) {
            smem_fw_field_map[i + tid] = gmem_fw_field_map[i + tid];
        }
    }
    for (int32_t i = fw_field_num; i < fw_field_num * 2; i += warp_num * 32) {
        if (i + tid < fw_field_num * 2) {
            smem_fw_field_map[i + tid] = -1;
        }
    }

    __syncthreads();

    // patch
    if (blockIdx.x != 0) {
        int32_t sample_0_start_row = warp_id + sample_0_feature_start_addr;
        int32_t sample_0_end_row = sample_0_feature_num + sample_0_feature_start_addr;
        for (int32_t wid = sample_0_start_row; wid < sample_0_end_row; wid += warp_num) {
            int32_t field_1 = field_tensor[wid * 2] - 1;
            int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
            if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
                continue;

            if (lane_id == 0) smem_fw_field_map[fw_field_1] = field_1;
        }
    }

    // sample feature phase

    int32_t common_fw_field_map_offset_for_ad = blockIdx.x > 0 ? fw_field_num : 0;

    int32_t sample_start_row = warp_id + this_sample_feature_start_addr;
    int32_t sample_end_row = this_sample_feature_num + this_sample_feature_start_addr;
    for (int32_t wid = sample_start_row; wid < sample_end_row; wid += warp_num) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;

        if (lane_id == 0) {
            smem_fw_field_map[fw_field_1 + common_fw_field_map_offset_for_ad] = field_1;
        }

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_size;
#pragma unroll
            for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
                T reg = T(0);
                int32_t rd_offset =
                wid * embedding_size * field_num + field_2 * embedding_size + n + lane_id;
                T* wr_ptr = gmem_cross_mean_sum + mem_field_offset + n + lane_id;

                if (n + lane_id < embedding_size) {
                    reg = weight_tensor[rd_offset];
                    atomicAdd(wr_ptr, reg);
                }
            }
        }
        int32_t mem_field_offset = fw_field_1 * embedding_size;
#pragma unroll
        for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
            T reg = T(0);
            T square = T(0);
            int32_t rd_offset =
            wid * embedding_size * field_num + field_1 * embedding_size + n + lane_id;
            T* wr_ptr = gmem_cross_square_sum + mem_field_offset + n + lane_id;

            if (n + lane_id < embedding_size) {
                reg = weight_tensor[rd_offset];
                square = reg * reg;
                atomicAdd(wr_ptr, square);
            }
        }
    }

    __syncthreads();
    int32_t i = 0;
    for (int32_t i = 0; i < fw_field_num; i += total_thread) {
        if (i + tid < fw_field_num) {
            int32_t field_1 = smem_fw_field_map[common_fw_field_map_offset_for_ad + i + tid];
            int32_t field_1_part1 = smem_fw_field_map[i + tid];
            if (field_1 < 0 && field_1_part1 != 1) {
                field_1 = field_1_part1;
            }
            smem_fw_field_map[i + tid] = field_1;
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
__device__ void ProcessSamplePart_share(
int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* smem_cross_mean_sum,
T* smem_cross_square_sum, T* gmem_fw_cross_mean_sum, T* gmem_fw_cross_square_sum,
int32_t* gmem_fw_field_map, int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
int32_t this_sample_feature_num, int32_t this_sample_feature_start_addr,
int32_t sample_0_feature_num, int32_t sample_0_feature_start_addr, const T* weight_tensor,
const int* field_tensor, int32_t shared_mem_elements) {

    constexpr int32_t total_thread = warp_num * 32;

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t tid = threadIdx.x + threadIdx.y * 32;

    for (int32_t i = 0; i < fw_field_num; i += warp_num * 32) {
        if (i + tid < fw_field_num) {
            smem_fw_field_map[i + tid] = gmem_fw_field_map[i + tid];
        }
    }
    for (int32_t i = fw_field_num; i < fw_field_num * 2; i += warp_num * 32) {
        if (i + tid < fw_field_num * 2) {
            smem_fw_field_map[i + tid] = -1;
        }
    }

    for (int32_t i = tid; i < embedding_size * field_num * fw_field_num; i += warp_num * 32) {
        smem_cross_mean_sum[i] = gmem_fw_cross_mean_sum[i];
        if (i < embedding_size * fw_field_num) {
            smem_cross_square_sum[i] = gmem_fw_cross_square_sum[i];
        }
    }

    __syncthreads();

    // patch
    if (blockIdx.x != 0) {
        int32_t sample_0_start_row = warp_id + sample_0_feature_start_addr;
        int32_t sample_0_end_row = sample_0_feature_num + sample_0_feature_start_addr;
        for (int32_t wid = sample_0_start_row; wid < sample_0_end_row; wid += warp_num) {
            int32_t field_1 = field_tensor[wid * 2] - 1;
            int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
            if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
                continue;

            if (lane_id == 0) smem_fw_field_map[fw_field_1] = field_1;
        }
    }

    // sample feature phase

    int32_t common_fw_field_map_offset_for_ad = blockIdx.x > 0 ? fw_field_num : 0;

    int32_t sample_start_row = warp_id + this_sample_feature_start_addr;
    int32_t sample_end_row = this_sample_feature_num + this_sample_feature_start_addr;
    for (int32_t wid = sample_start_row; wid < sample_end_row; wid += warp_num) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;

        if (lane_id == 0) {
            smem_fw_field_map[fw_field_1 + common_fw_field_map_offset_for_ad] = field_1;
        }

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_size;
#pragma unroll
            for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
                T reg = T(0);
                int32_t rd_offset =
                wid * embedding_size * field_num + field_2 * embedding_size + n + lane_id;
                T* wr_ptr = smem_cross_mean_sum + mem_field_offset + n + lane_id;

                if (n + lane_id < embedding_size) {
                    reg = weight_tensor[rd_offset];
                    atomicAdd(wr_ptr, reg);
                }
            }
        }
        int32_t mem_field_offset = fw_field_1 * embedding_size;
#pragma unroll
        for (int32_t n = 0; n < (embedding_size + 31) / 32 * 32; n += 32) {
            T reg = T(0);
            T square = T(0);
            int32_t rd_offset =
            wid * embedding_size * field_num + field_1 * embedding_size + n + lane_id;
            T* wr_ptr = smem_cross_square_sum + mem_field_offset + n + lane_id;

            if (n + lane_id < embedding_size) {
                reg = weight_tensor[rd_offset];
                square = reg * reg;
                atomicAdd(wr_ptr, square);
            }
        }
    }

    __syncthreads();
    int32_t i = 0;
    for (int32_t i = 0; i < fw_field_num; i += total_thread) {
        if (i + tid < fw_field_num) {
            int32_t field_1 = smem_fw_field_map[common_fw_field_map_offset_for_ad + i + tid];
            int32_t field_1_part1 = smem_fw_field_map[i + tid];
            if (field_1 < 0 && field_1_part1 != 1) {
                field_1 = field_1_part1;
            }
            smem_fw_field_map[i + tid] = field_1;
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
__device__ void ProcessOutput_share(T* smem_cross_mean_sum, T* smem_cross_square_sum,
                                    T* mem_fw_cross_mean_sum, T* mem_fw_cross_square_sum,
                                    int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
                                    const T* fw_weight_tensor, T* output_gmem, int32_t batch_id,
                                    int32_t embedding_size, int32_t field_num,
                                    int32_t fw_field_num) {

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t weight_size_pad = (embedding_size + 31) / 32 * 32;
    int32_t common_fw_field_map_offset_for_ad = blockIdx.x > 0 ? fw_field_num : 0;

    // T output_accu[(embedding_size + 31) / 32] = {0};
    T output_accu[4] = {0};
    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
    fw_field_1_idx += warp_num) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t fw_iter = (2 + fw_field_1) * (fw_field_1 + 1) / 2 - (fw_field_1 + 1);

        int32_t field_1 = smem_fw_field_map[fw_field_1];

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];

            T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_2] + T(1);
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size;

            for (int32_t n = 0; n < weight_size_pad; n += 32) {

                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);

                if (n + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n + lane_id];
                }
                output_accu[n / 32] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }
        T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size;
        for (int32_t n = 0; n < weight_size_pad; n += 32) {

            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (n + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_size + n + lane_id];
            }
            output_accu[n / 32] +=
            T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum) * fw_weight_reg;
        }
    }

    for (int32_t n = 0; n < weight_size_pad; n += 32) {
        if (n + lane_id < embedding_size) {
            T* Outptr = (output_gmem + batch_id * embedding_size + n + lane_id);
            atomicAdd(Outptr, output_accu[n / 32]);
        }
    }
}

template <typename T = float, int32_t warp_num = 32>
__device__ void ProcessOutput(T* mem_cross_mean_sum, T* mem_cross_square_sum,
                              T* mem_fw_cross_mean_sum, T* mem_fw_cross_square_sum,
                              int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
                              const T* fw_weight_tensor, T* output_gmem, int32_t batch_id,
                              int32_t embedding_size, int32_t field_num, int32_t fw_field_num) {

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t weight_size_pad = (embedding_size + 31) / 32 * 32;
    int32_t common_fw_field_map_offset_for_ad = blockIdx.x > 0 ? fw_field_num : 0;

    // T output_accu[(embedding_size + 31) / 32] = {0};
    T output_accu[6] = { 0 };
    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
    fw_field_1_idx += warp_num) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t fw_iter = (2 + fw_field_1) * (fw_field_1 + 1) / 2 - (fw_field_1 + 1);

        int32_t field_1 = smem_fw_field_map[fw_field_1];

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];

            T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_2] + T(1);
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size;

            for (int32_t n = 0; n < weight_size_pad; n += 32) {

                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);

                if (n + lane_id < embedding_size) {
                    mean_index_1_sum = mem_cross_mean_sum[index_1 + n + lane_id];
                    mean_index_2_sum = mem_cross_mean_sum[index_2 + n + lane_id];
                }
                output_accu[n / 32] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }
        T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size;
        for (int32_t n = 0; n < weight_size_pad; n += 32) {

            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (n + lane_id < embedding_size) {
                cross_mean_sum = mem_cross_mean_sum[index_1 + n + lane_id];
                cross_square_sum = mem_cross_square_sum[fw_field_1 * embedding_size + n + lane_id];
            }
            output_accu[n / 32] +=
            T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum) * fw_weight_reg;
        }
    }

    for (int32_t n = 0; n < weight_size_pad; n += 32) {
        if (n + lane_id < embedding_size) {
            T* Outptr = (output_gmem + batch_id * embedding_size + n + lane_id);
            atomicAdd(Outptr, output_accu[n / 32]);
        }
    }
}

template <typename T = float, int32_t warp_num = 32>
__global__ void ProcessFwffmOutput(int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
                                   bool fw_weight_multil_flag, int32_t* sample_feature_start_addr,
                                   int32_t* sample_feature_end_addr, const T* weight_tensor,
                                   const int32_t* field_tensor, const T* fw_weight_tensor,
                                   T* output_tensor, T* workspace)

                                   {
    int32_t batch_size = gridDim.x;
    int32_t warp_id = threadIdx.y;

    extern __shared__ float smem_pool[];
    int32_t batch_id = blockIdx.x;
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * fw_field_num;
    // Use global memory in case of lacking atomicAdd float in shared mem

    T* mem_cross_mean_sum =
    workspace + batch_id * (embedding_size * (field_num + 1) * fw_field_num);
    T* mem_cross_square_sum = mem_cross_mean_sum + embedding_size * field_num * fw_field_num;
    T* mem_fw_cross_mean_sum =
    workspace + batch_size * (embedding_size * (field_num + 1) * fw_field_num);
    T* mem_fw_cross_square_sum = mem_fw_cross_mean_sum + embedding_size * field_num * fw_field_num;

    int32_t* mem_fw_field_map =
    reinterpret_cast<int*>(mem_fw_cross_square_sum + fw_field_num * embedding_size);

    const T* local_fw_weight_data = fw_weight_tensor;
    if (fw_weight_multil_flag) {
        local_fw_weight_data = fw_weight_tensor + batch_id * fw_weight_size;
    }

    int32_t this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
    int32_t this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
    int32_t this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

    int32_t sample_0_feature_start_addr = sample_feature_start_addr[0];
    int32_t sample_0_feature_end_addr = sample_feature_start_addr[1];
    int32_t sample_0_feature_num = sample_0_feature_end_addr - sample_0_feature_start_addr;

    ProcessSamplePart<T, warp_num>(
    smem_fw_field_map, smem_fw_map_idx, mem_cross_mean_sum, mem_cross_square_sum,
    mem_fw_cross_mean_sum, mem_fw_cross_square_sum, mem_fw_field_map, embedding_size, field_num,
    fw_field_num, this_sample_feature_num, this_sample_feature_start_addr, sample_0_feature_num,
    sample_0_feature_start_addr, weight_tensor, field_tensor, warp_num * embedding_size);

    ProcessOutput<T, warp_num>(mem_cross_mean_sum, mem_cross_square_sum, mem_fw_cross_mean_sum,
                               mem_fw_cross_square_sum, smem_fw_field_map, smem_fw_map_idx,
                               local_fw_weight_data, output_tensor, batch_id, embedding_size, field_num,
                               fw_field_num);
                                   }

                                   template <typename T = float, int32_t warp_num = 32>
                                   __global__ void ProcessFwffmOutput_share(int32_t embedding_size, int32_t field_num,
                                                                            int32_t fw_field_num, bool fw_weight_multil_flag,
                                                                            int32_t* sample_feature_start_addr,
                                                                            int32_t* sample_feature_end_addr, const T* weight_tensor,
                                                                            const int32_t* field_tensor, const T* fw_weight_tensor,
                                                                            T* output_tensor, T* workspace) {
    int32_t batch_size = gridDim.x;
    int32_t warp_id = threadIdx.y;

    extern __shared__ float smem_pool[];
    int32_t batch_id = blockIdx.x;
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * fw_field_num;
    // Use global memory in case of lacking atomicAdd float in shared mem

    T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + fw_field_num + 1);
    T* smem_cross_square_sum = smem_cross_mean_sum + embedding_size * field_num * fw_field_num;
    T* mem_fw_cross_mean_sum =
    workspace + batch_size * (embedding_size * (field_num + 1) * fw_field_num);
    T* mem_fw_cross_square_sum = mem_fw_cross_mean_sum + embedding_size * field_num * fw_field_num;

    int32_t* mem_fw_field_map =
    reinterpret_cast<int*>(mem_fw_cross_square_sum + fw_field_num * embedding_size);

    const T* local_fw_weight_data = fw_weight_tensor;
    if (fw_weight_multil_flag) {
        local_fw_weight_data = fw_weight_tensor + batch_id * fw_weight_size;
    }

    int32_t this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
    int32_t this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
    int32_t this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

    int32_t sample_0_feature_start_addr = sample_feature_start_addr[0];
    int32_t sample_0_feature_end_addr = sample_feature_start_addr[1];
    int32_t sample_0_feature_num = sample_0_feature_end_addr - sample_0_feature_start_addr;

    ProcessSamplePart_share<T, warp_num>(
    smem_fw_field_map, smem_fw_map_idx, smem_cross_mean_sum, smem_cross_square_sum,
    mem_fw_cross_mean_sum, mem_fw_cross_square_sum, mem_fw_field_map, embedding_size, field_num,
    fw_field_num, this_sample_feature_num, this_sample_feature_start_addr, sample_0_feature_num,
    sample_0_feature_start_addr, weight_tensor, field_tensor, warp_num * embedding_size);

    ProcessOutput_share<T, warp_num>(
    smem_cross_mean_sum, smem_cross_square_sum, mem_fw_cross_mean_sum, mem_fw_cross_square_sum,
    smem_fw_field_map, smem_fw_map_idx, fw_weight_tensor, output_tensor, batch_id,
    embedding_size, field_num, fw_field_num);
}

namespace functor {
template <typename T>
int32_t ComputeSparseFwffm(cudaStream_t stream, const void* const* input, T* output,
                           void* worksapce, const int32_t fw_field_num,
                           const int32_t fw_weight_size, const bool fw_weight_multil_flag,
                           const int32_t sample_feature_size, const int32_t field_num,
                           const int32_t embedding_size, const int32_t batch_size) {
    const T* weight_data = static_cast<const T*>(input[0]);
    const T* fw_weight_data = static_cast<const T*>(input[1]);
    const int32_t* field_data = static_cast<const int32_t*>(input[2]);
    const int32_t* index_data = static_cast<const int32_t*>(input[3]);

    const int32_t kThreadsPerBlock = 1024;
    const size_t kBufferSize = field_num * fw_field_num * embedding_size;

    int32_t* sample_feature_start_addr = reinterpret_cast<int32_t*>(worksapce);
    int32_t* sample_feature_end_addr = sample_feature_start_addr + batch_size + 1;
    T* gmem_cross_sum = reinterpret_cast<T*>(sample_feature_end_addr + batch_size + 1);
    T* gmem_fw_cross_mean_sum = gmem_cross_sum + batch_size * (field_num + 1) * embedding_size * fw_field_num;
    T* gmem_fw_cross_square_sum = gmem_fw_cross_mean_sum + embedding_size * field_num * fw_field_num;

    int32_t* gmem_fw_field_map =
    reinterpret_cast<int*>(gmem_fw_cross_square_sum + embedding_size * fw_field_num);

    CUDA_CHECK(cudaMemsetAsync(gmem_fw_cross_mean_sum, 0,
                               sizeof(float) * (embedding_size * field_num * fw_field_num +
                               embedding_size * fw_field_num), stream));
    CUDA_CHECK(cudaMemsetAsync(gmem_fw_field_map, -1, sizeof(int) * (fw_field_num), stream));
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float) * (batch_size * embedding_size), stream));

    CUDA_CHECK(cudaMemsetAsync(sample_feature_start_addr, 0, sizeof(int32_t) * (batch_size + 1), stream));
    CUDA_CHECK(cudaMemsetAsync(sample_feature_end_addr, 0, sizeof(int32_t) * (batch_size + 1), stream));

    ComputeBatchBoundary<<<DIVUP(sample_feature_size, 1024), 1024, 0, stream>>>(
    index_data, sample_feature_size, batch_size, sample_feature_start_addr,
    sample_feature_end_addr);
    constexpr int32_t warp_num = 32;
    dim3 block(32, 32);
    dim3 grid0(1);

    ProcessCommonPart<T, 32><<<grid0, block, 0, stream>>>(
    embedding_size, field_num, fw_field_num, sample_feature_start_addr, weight_data, field_data,
    gmem_fw_cross_mean_sum, gmem_fw_cross_square_sum, gmem_fw_field_map);
    int32_t share_mem_size = (fw_field_num * 3 + 1) * sizeof(int32_t) +
    embedding_size * (field_num + 1) * fw_field_num * sizeof(T);

    if (share_mem_size < 65536) {
        cudaFuncSetAttribute(ProcessFwffmOutput_share<T, warp_num>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        dim3 grid(batch_size);

        ProcessFwffmOutput_share<T, warp_num><<<grid, block, share_mem_size, stream>>>(
        embedding_size, field_num, fw_field_num, fw_weight_multil_flag,
        sample_feature_start_addr, sample_feature_end_addr, weight_data, field_data,
        fw_weight_data, output, gmem_cross_sum);
    } else {
        //                    printf("Do not use shared memory.\n");
        dim3 block_set(embedding_size);
        dim3 grid_set(batch_size * fw_field_num);
        BroadcastCommonPart<T><<<grid_set, block_set, 0, stream>>>(
        batch_size, embedding_size, field_num, fw_field_num, gmem_fw_cross_mean_sum,
        gmem_fw_cross_square_sum, gmem_cross_sum);

        int32_t shared_mem_required_bytes = (fw_field_num * (field_num + 1) + 1)* sizeof(int32_t);
        dim3 grid(batch_size);
        cudaFuncSetAttribute(ProcessFwffmOutput<T, warp_num>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        ProcessFwffmOutput<T, warp_num><<<grid, block, shared_mem_required_bytes, stream>>>(
        embedding_size, field_num, fw_field_num, fw_weight_multil_flag,
        sample_feature_start_addr, sample_feature_end_addr, weight_data, field_data,
        fw_weight_data, output, gmem_cross_sum);
    }
    return 1;
}

template int32_t ComputeSparseFwffm(cudaStream_t stream, const void* const* input, float* output,
void* worksapce, const int32_t fw_field_num,
const int32_t fw_weight_size, const bool fw_weight_multil_flag,
const int32_t sample_feature_size, const int32_t field_num,
const int32_t embedding_size, const int32_t batch_size);

template int32_t ComputeSparseFwffm(cudaStream_t stream, const void* const* input, half* output,
void* worksapce, const int32_t fw_field_num,
const int32_t fw_weight_size, const bool fw_weight_multil_flag,
const int32_t sample_feature_size, const int32_t field_num,
const int32_t embedding_size, const int32_t batch_size);
}  // namespace functor
}  // namespace sparse_fwffm
}  // namespace nvinfer1