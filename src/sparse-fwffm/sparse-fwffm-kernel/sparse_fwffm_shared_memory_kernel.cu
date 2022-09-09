// Copyright 2020, Tencent Inc.
// All rights reserved.
//
// @author shaorunwang <shaorunwang@tencent.com>
// @author hungryhe<hungryhe@tencent.com>

#include <cuda.h>
#include <cuda_fp16.h>
#include <thrust/extrema.h>

#include <iostream>

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

namespace sparse_fwffm_shared {

template <typename T>
__global__ void ComputeBatchBoundary(
const int32_t* index_tensor, int32_t total_feature_num, int32_t batch_size,
int32_t* sample_feature_start_addr, int32_t clear_size, T* output) {
    int32_t total_thread = gridDim.x * blockDim.x;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size + 1) {
        sample_feature_start_addr[tid] = 0;
    }
    __syncthreads();

    for (int32_t i = tid; i < clear_size; i += total_thread) {
        output[i] = T(0);
    }

    if (tid < total_feature_num) {
        int32_t idx = index_tensor[tid];
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

template <typename T>
__global__ void ComputeCommonPartOutput(
    int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num,
    int32_t fw_field_num, int32_t segment_common_fw_cross_size, int32_t output_size,
    int32_t* sample_feature_start_addr, const T* weight_tensor, const T* fw_weight_tensor,
    const int32_t* field_tensor, T* gmem_fw_cross_mean_sum,
    int32_t* gmem_fw_field_map, T* output) {
    extern __shared__ float smem_pool[];
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + fw_field_num + 1;

    T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + fw_field_num + 1);
    T* smem_cross_square_sum = smem_cross_mean_sum + embedding_segment_size * field_num * fw_field_num;
    T* smem_output = smem_cross_square_sum + embedding_segment_size * fw_field_num;

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t total_global_warp_num = blockDim.y;

    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = blockDim.x * blockDim.y;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t common_feature_num = sample_feature_start_addr[0];

    for (int32_t i = tid; i < segment_common_fw_cross_size; i += total_thread) {
        smem_cross_mean_sum[i] = T(0);
        if (i < fw_field_num) {
            smem_fw_field_map[i] = -1;
        }
    }
    for (int32_t i = tid; i < embedding_segment_size; i += total_thread) {
        smem_output[i] = 0;
    }
    __syncthreads();

    for (int32_t wid = warp_id; wid < common_feature_num; wid += total_global_warp_num) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;

        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;

        if (lane_id == 0) smem_fw_field_map[fw_field_1] = field_1;

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T reg = T(0);
                int32_t rd_offset = wid * embedding_size * field_num + field_2 * embedding_size
                + embedding_segment_start + n * blockDim.x + lane_id;
                T *wr_ptr = smem_cross_mean_sum + mem_field_offset + n * blockDim.x + lane_id;

                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    reg = weight_tensor[rd_offset];
                    atomicAdd(wr_ptr, reg);
                }
            }
        }

        int32_t mem_field_offset = fw_field_1 * embedding_segment_size;
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T reg = T(0);
            T square = T(0);
            int32_t rd_offset = wid * embedding_size * field_num + field_1 * embedding_size + embedding_segment_start
            + n * blockDim.x + lane_id;
            T *wr_ptr = smem_cross_square_sum + mem_field_offset + n * blockDim.x + lane_id;

            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                reg = weight_tensor[rd_offset];
                square = reg * reg;
                atomicAdd(wr_ptr, square);
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

    T output_accu[6] = { 0 };
    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];
    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
    fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_fw_field_map[fw_field_1];
        int32_t fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;

        T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_accu[n] += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum) * fw_weight_reg;
        }

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];
            fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_2] + T(1);
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_accu[n] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }

    }

    #pragma unroll
    for (int n = 0; n < segment_size_div_blockdimx; ++n) {
        if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
            atomicAdd(smem_output + n * blockDim.x + lane_id, output_accu[n]);
        }
    }
    __syncthreads();

    for (int32_t i = tid; i < fw_field_num + 1; i += total_thread) {
        gmem_fw_field_map[i] = smem_fw_field_map[i];
        gmem_fw_field_map[fw_field_num + 1 + i] = smem_fw_field_map[fw_field_num + 1 + i];
    }

    for (int32_t i = warp_id; i < (field_num + 1) * fw_field_num; i += blockDim.y) {
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            int32_t embedding_offset = embedding_segment_start + n * blockDim.x + lane_id;
            if (embedding_offset < embedding_size) {
                gmem_fw_cross_mean_sum[i * embedding_size + embedding_offset]
                = smem_cross_mean_sum[i * embedding_segment_size + n * blockDim.x + lane_id];
            }
        }
    }

    for (int32_t i = tid; i < embedding_size; i += total_thread) {
        if (i >= embedding_segment_start && i < embedding_segment_start + embedding_segment_size) {
            atomicAdd(output + i, smem_output[i - embedding_segment_start]);
        }
    }
}

template <typename T>
__global__ void BroadcastCommonPartOutput(int32_t embedding_size, T* output) {
    int32_t total_thread = blockDim.x * blockDim.y;
    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    T* batch_start_addr = output + (blockIdx.x + 1) * embedding_size;
    for (int32_t i = tid; i < embedding_size; i += total_thread) {
        batch_start_addr[i] = output[i];
    }
}

template <typename T>
__device__ void AccumulateSamplePart(
int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* smem_cross_mean_sum,
T* smem_cross_square_sum, T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
int32_t* gmem_common_field_map, int32_t embedding_size, int32_t embedding_segment_size,
int32_t field_num, int32_t fw_field_num, int32_t this_sample_feature_num,
int32_t this_sample_feature_start_addr, const T* weight_tensor, const int32_t* field_tensor) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = blockDim.x * blockDim.y;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;

    for (int32_t i = tid; i < fw_field_num + 1; i += total_thread) {
        smem_fw_field_map[i] = -1;
        smem_fw_field_map[fw_field_num + 1 + i] = gmem_common_field_map[i];
        smem_fw_map_idx[fw_field_num + 1 + i] = gmem_common_field_map[fw_field_num + 1 + i];
    }

    for (int32_t i = warp_id; i < (field_num + 1) * fw_field_num; i += blockDim.y) {
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            int32_t embedding_offset = embedding_segment_start + n * blockDim.x + lane_id;
            if (embedding_offset < embedding_size) {
                smem_cross_mean_sum[i * embedding_segment_size + n * blockDim.x + lane_id] =
                gmem_common_cross_mean_sum[i * embedding_size + embedding_offset];
            }
        }
    }
    __syncthreads();

    // sample feature phase
    int32_t sample_start_row = warp_id + this_sample_feature_start_addr;
    int32_t sample_end_row = this_sample_feature_num + this_sample_feature_start_addr;
    for (int32_t wid = sample_start_row; wid < sample_end_row; wid += blockDim.y) {
        int32_t field_1 = field_tensor[wid * 2] - 1;
        int32_t fw_field_1 = field_tensor[wid * 2 + 1] - 1;
        if (fw_field_1 < 0 || fw_field_1 >= fw_field_num || field_1 < 0 || field_1 >= field_num)
            continue;
        if (lane_id == 0) smem_fw_field_map[fw_field_1] = field_1;

#pragma unroll
        for (int32_t field_2 = 0; field_2 < field_num; field_2++) {
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T reg = T(0);
                int32_t rd_offset = wid * embedding_size * field_num + field_2 * embedding_size
                + embedding_segment_start + n * blockDim.x + lane_id;
                T *wr_ptr = smem_cross_mean_sum + mem_field_offset + n * blockDim.x + lane_id;

                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    reg = weight_tensor[rd_offset];
                    atomicAdd(wr_ptr, reg);
                }
            }
        }

        int32_t mem_field_offset = fw_field_1 * embedding_segment_size;
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T reg = T(0);
            T square = T(0);
            int32_t rd_offset = wid * embedding_size * field_num + field_1 * embedding_size + embedding_segment_start
            + n * blockDim.x + lane_id;
            T *wr_ptr = smem_cross_square_sum + mem_field_offset + n * blockDim.x + lane_id;

            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                reg = weight_tensor[rd_offset];
                square = reg * reg;
                atomicAdd(wr_ptr, square);
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

template <typename T>
__device__ void ComputeCommonSamplePartOutput(
    int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num, int32_t fw_field_num,
    const T* fw_weight_tensor, T* smem_cross_mean_sum, T* smem_cross_square_sum, int32_t* smem_fw_field_map,
    int32_t* smem_fw_map_idx, T* output) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;

    int32_t* smem_sample_fw_field_map = smem_fw_field_map;
    int32_t* smem_common_fw_field_map = smem_fw_field_map + fw_field_num + 1;
    int32_t* smem_sample_fw_map_idx = smem_fw_map_idx;
    int32_t* smem_common_fw_map_idx = smem_fw_map_idx + fw_field_num + 1;
    int32_t common_vaild_fw_field = smem_common_fw_map_idx[fw_field_num];
    int32_t sample_vaild_fw_field = smem_sample_fw_map_idx[fw_field_num];
//    if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 0 && lane_id == 0) {
//        printf("[FWFFM] common_vaild_fw_field=%d\n", common_vaild_fw_field);
//        printf("[FWFFM] sample_vaild_fw_field=%d\n", sample_vaild_fw_field);
//    }

    T output_accu[6] = { 0 };
    T fw_weight_reg = T(0);
    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < sample_vaild_fw_field; fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_sample_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_sample_fw_field_map[fw_field_1];
        int32_t fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;

        fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_accu[n] += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum) * fw_weight_reg;
        }

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];
            fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_2] + T(1);
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_accu[n] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }

    }

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < common_vaild_fw_field; fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_common_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_common_fw_field_map[fw_field_1];

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < sample_vaild_fw_field; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_sample_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_sample_fw_field_map[fw_field_2];
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;
            int32_t fw_iter = fw_field_2 * (fw_field_2 + 1) / 2;
            if (fw_field_2 < fw_field_1) {
                fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;
                fw_field_1 = fw_field_2;
            }

            fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_accu[n] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }
    }

#pragma unroll
    for (int n = 0; n < segment_size_div_blockdimx; ++n) {
        if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
            atomicAdd(output + embedding_segment_start + n * blockDim.x + lane_id, output_accu[n]);
        }
    }
}

template <typename T = float>
__device__ void ProcessSamplePartShare(
    int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* smem_cross_mean_sum,
    T* smem_cross_square_sum, T* gmem_fw_cross_mean_sum, T* gmem_fw_cross_square_sum,
    int32_t* gmem_fw_field_map, int32_t embedding_size, int32_t embedding_segment_size,
    int32_t field_num, int32_t fw_field_num, int32_t this_sample_feature_num,
    int32_t this_sample_feature_start_addr, int32_t sample_0_feature_num,
    int32_t sample_0_feature_start_addr, const T* weight_tensor,
    const int* field_tensor, int32_t shared_mem_elements) {

    int32_t warp_num = blockDim.y;
    int32_t total_thread = blockDim.x * blockDim.y;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;

    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;

    for (int32_t i = 0; i < fw_field_num; i += total_thread) {
        if (i + tid < fw_field_num) {
             smem_fw_field_map[i + tid] = gmem_fw_field_map[i + tid];
        }
    }
    for (int32_t i = fw_field_num; i < fw_field_num * 2; i += total_thread) {
        if (i + tid < fw_field_num * 2) {
             smem_fw_field_map[i + tid] = -1;
        }
    }

    for (int32_t i = warp_id; i < field_num * fw_field_num; i += warp_num) {
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                smem_cross_mean_sum[i * embedding_segment_size + n * blockDim.x + lane_id] =
                gmem_fw_cross_mean_sum[i * embedding_size + embedding_segment_start + n * blockDim.x + lane_id];
                if (i < fw_field_num) {
                    smem_cross_square_sum[i * embedding_segment_size + n * blockDim.x + lane_id] =
                    gmem_fw_cross_square_sum[i * embedding_size + embedding_segment_start + n * blockDim.x + lane_id];
                }
            }
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

            if (lane_id == 0) {
                 smem_fw_field_map[fw_field_1] = field_1;
            }
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
            int32_t mem_field_offset = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T reg = T(0);
                int32_t rd_offset = wid * embedding_size * field_num + field_2 * embedding_size
                        + embedding_segment_start + n * blockDim.x + lane_id;
                T *wr_ptr = smem_cross_mean_sum + mem_field_offset + n * blockDim.x + lane_id;

                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    reg = weight_tensor[rd_offset];
                    atomicAdd(wr_ptr, reg);
                }
            }
        }

        int32_t mem_field_offset = fw_field_1 * embedding_segment_size;
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T reg = T(0);
            T square = T(0);
            int32_t rd_offset = wid * embedding_size * field_num + field_1 * embedding_size + embedding_segment_start
                    + n * blockDim.x + lane_id;
            T *wr_ptr = smem_cross_square_sum + mem_field_offset + n * blockDim.x + lane_id;

            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                reg = weight_tensor[rd_offset];
                square = reg * reg;
                atomicAdd(wr_ptr, square);
            }
        }
    }
    __syncthreads();

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

template <typename T = float>
__device__ void ProcessOutputShare(
    T* smem_cross_mean_sum, T* smem_cross_square_sum,
    T* mem_fw_cross_mean_sum, T* mem_fw_cross_square_sum,
    int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
    const T* fw_weight_tensor, T* output_gmem, int32_t batch_id,
    int32_t embedding_size, int32_t embedding_segment_size,
    int32_t field_num, int32_t fw_field_num) {

    int32_t warp_num = blockDim.y;
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t common_fw_field_map_offset_for_ad = blockIdx.x > 0 ? fw_field_num : 0;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;

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
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_accu[n] += mean_index_1_sum * mean_index_2_sum * fw_weight_reg;
            }
        }

        T fw_weight_reg = fw_weight_tensor[fw_iter + fw_field_1] + T(1);
        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;

#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_accu[n] += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum) * fw_weight_reg;
        }
    }

#pragma unroll
    for (int n = 0; n < segment_size_div_blockdimx; ++n) {
        if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
            T* Outptr = (output_gmem + batch_id * embedding_size + embedding_segment_start + n * blockDim.x + lane_id);
            atomicAdd(Outptr, output_accu[n]);
        }
    }
}

template <typename T = float>
__global__ void ProcessFwffmOutputShare(
    int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num,
    int32_t fw_field_num, const bool fw_weight_multil_flag, int32_t* sample_feature_start_addr,
    int32_t* mem_fw_field_map, T* mem_fw_cross_mean_sum, T* mem_fw_cross_square_sum,
    const T* weight_tensor, const int32_t* field_tensor, const T* fw_weight_tensor,
    T* output_tensor) {
    int32_t batch_size = gridDim.x;
    int32_t warp_num = blockDim.y;
    int32_t warp_id = threadIdx.y;
    int32_t batch_id = blockIdx.x;
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;

    extern __shared__ float smem_pool[];
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * fw_field_num;
    T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + fw_field_num + 1);
    T* smem_cross_square_sum = smem_cross_mean_sum + embedding_segment_size * field_num * fw_field_num;

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

    ProcessSamplePartShare<T>(
        smem_fw_field_map, smem_fw_map_idx, smem_cross_mean_sum, smem_cross_square_sum, mem_fw_cross_mean_sum,
        mem_fw_cross_square_sum, mem_fw_field_map, embedding_size, embedding_segment_size, field_num,
        fw_field_num, this_sample_feature_num, this_sample_feature_start_addr, sample_0_feature_num,
        sample_0_feature_start_addr, weight_tensor, field_tensor, warp_num * embedding_size);

     ProcessOutputShare<T>(
         smem_cross_mean_sum, smem_cross_square_sum, mem_fw_cross_mean_sum, mem_fw_cross_square_sum,
         smem_fw_field_map, smem_fw_map_idx, local_fw_weight_data, output_tensor, batch_id,
         embedding_size, embedding_segment_size, field_num, fw_field_num);
}

template <typename T = float>
__global__ void ProcessFwffmOutputShareV2(
    int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num,
    int32_t fw_field_num, int32_t* sample_feature_start_addr,
    T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
    int32_t* gmem_common_field_map, const T* weight_tensor, const T* fw_weight_tensor,
    const int32_t* field_tensor, T* output_tensor) {
    extern __shared__ float smem_pool[];
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * (fw_field_num + 1);

    T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + 2 * (fw_field_num + 1));
    T* smem_cross_square_sum = smem_cross_mean_sum + embedding_segment_size * field_num * fw_field_num;

    int32_t batch_id = blockIdx.x;
    T* gmem_output = output_tensor + batch_id * embedding_size;
    int32_t this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
    int32_t this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
    int32_t this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

    AccumulateSamplePart<T>(
        smem_fw_field_map, smem_fw_map_idx, smem_cross_mean_sum, smem_cross_square_sum,
        gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_common_field_map,
        embedding_size, embedding_segment_size, field_num, fw_field_num, this_sample_feature_num,
        this_sample_feature_start_addr, weight_tensor, field_tensor);
    ComputeCommonSamplePartOutput<T>(
        embedding_size, embedding_segment_size, field_num, fw_field_num, fw_weight_tensor, smem_cross_mean_sum,
        smem_cross_square_sum, smem_fw_field_map, smem_fw_map_idx, gmem_output);
}

template <typename T = float, int32_t warp_num = 32>
__global__ void ProcessCommonPart(
    int32_t embedding_size, int32_t field_num, int32_t fw_field_num, int32_t common_fw_cross_size,
    int32_t* sample_feature_start_addr, const T* weight_tensor, const int32_t* field_tensor,
    T* gmem_fw_cross_mean_sum, T* gmem_fw_cross_square_sum, int32_t* gmem_fw_field_map) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t global_warp_id = blockIdx.x * warp_num + warp_id;
    int32_t total_global_warp_num = gridDim.x * warp_num;

    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = blockDim.x * blockDim.y;
    int32_t common_feature_num = sample_feature_start_addr[0];

    for (int32_t i = tid; i < common_fw_cross_size; i += total_thread) {
        gmem_fw_cross_mean_sum[i] = 0;
        if (i < fw_field_num) {
            gmem_fw_field_map[i] = -1;
        }
    }
    //    __syncthreads();

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
    int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* gmem_cross_mean_sum, T* gmem_cross_square_sum,
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
__device__ void ProcessOutput(T* mem_cross_mean_sum, T* mem_cross_square_sum,
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
__global__ void ProcessFwffmOutput(
    int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
    bool fw_weight_multil_flag, int32_t* sample_feature_start_addr, int32_t* mem_fw_field_map,
    const T* weight_tensor, const int32_t* field_tensor, const T* fw_weight_tensor,
    T* output_tensor, T* gmem_cross_sum) {
    int32_t batch_size = gridDim.x;
    int32_t warp_id = threadIdx.y;

    extern __shared__ float smem_pool[];
    int32_t batch_id = blockIdx.x;
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * fw_field_num;

    T* mem_cross_mean_sum = gmem_cross_sum + batch_id * (embedding_size * (field_num + 1) * fw_field_num);
    T* mem_cross_square_sum = mem_cross_mean_sum + embedding_size * field_num * fw_field_num;
    T* mem_fw_cross_mean_sum = gmem_cross_sum + batch_size * (embedding_size * (field_num + 1) * fw_field_num);
    T* mem_fw_cross_square_sum = mem_fw_cross_mean_sum + embedding_size * field_num * fw_field_num;
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
        smem_fw_field_map, smem_fw_map_idx, mem_cross_mean_sum, mem_cross_square_sum, mem_fw_field_map,
        embedding_size, field_num, fw_field_num, this_sample_feature_num, this_sample_feature_start_addr,
        sample_0_feature_num, sample_0_feature_start_addr, weight_tensor, field_tensor, warp_num * embedding_size);

    ProcessOutput<T, warp_num>(
        mem_cross_mean_sum, mem_cross_square_sum, smem_fw_field_map, smem_fw_map_idx,
        local_fw_weight_data, output_tensor, batch_id, embedding_size, field_num,
        fw_field_num);
}

namespace functor {
    template <typename T>
    int32_t ComputeSparseFwffmSharedMemory(cudaStream_t stream, const void* const* input, T* output,
                                           void* workspace, const int32_t fw_field_num,
                                           const int32_t fw_weight_size, const bool fw_weight_multil_flag,
                                           const int32_t sample_feature_size, const int32_t field_num,
                                           const int32_t embedding_size, const int32_t batch_size) {
        const T* weight_tensor = static_cast<const T*>(input[0]);
        const T* fw_weight_tensor = static_cast<const T*>(input[1]);
        const int32_t* field_tensor = static_cast<const int32_t*>(input[2]);
        const int32_t* index_tensor = static_cast<const int32_t*>(input[3]);
//        printf("[FWFFM] batch_size=%d\n", batch_size);

        int32_t data_offset = 0;
        int32_t data_align_count = 0;
        data_align_count = 128 / sizeof(int32_t);
        int32_t* sample_feature_start_addr = static_cast<int32_t*>(workspace);
        data_offset = batch_size + 1;
        data_offset = (data_offset + data_align_count - 1) / data_align_count * data_align_count;
        int32_t* gmem_field_map = sample_feature_start_addr + data_offset;

        data_offset = 2 * (fw_field_num + 1);
        data_offset = (data_offset + data_align_count - 1) / data_align_count * data_align_count;
        T* gmem_cross_sum = reinterpret_cast<T*>(gmem_field_map + data_offset);
        T* gmem_common_cross_mean_sum = gmem_cross_sum + batch_size * (embedding_size * (field_num + 1) * fw_field_num);
        T* gmem_common_cross_square_sum = gmem_common_cross_mean_sum + embedding_size * field_num * fw_field_num;

        int32_t max_share_mem_size = 65536;
        int32_t max_embedding_segment_size =
        (max_share_mem_size - 4 * (fw_field_num + 1) * sizeof(int32_t) - embedding_size * sizeof(T))
        / ((field_num + 1) * fw_field_num * sizeof(T));
        max_embedding_segment_size = max_embedding_segment_size / 32 * 32;
        int32_t embedding_segment_count = (embedding_size + max_embedding_segment_size - 1) / max_embedding_segment_size;
        int32_t embedding_segment_size = embedding_size / embedding_segment_count;
        embedding_segment_size = (embedding_segment_size + 31) / 32 * 32;
        int32_t share_mem_size = 4 * (fw_field_num + 1) * sizeof(int32_t) + embedding_segment_size * sizeof(T)
        + embedding_segment_size * (field_num + 1) * fw_field_num * sizeof(T);
//        int32_t common_part_share_mem_size = (fw_field_num * 2 + 1) * sizeof(int32_t)
//                + embedding_segment_size * (field_num + 1) * fw_field_num * sizeof(T);
//        std::cout << "max_embedding_segment_size=" << max_embedding_segment_size << std::endl;
//        std::cout << "embedding_segment_size=" << embedding_segment_size << std::endl;
//        std::cout << "share_mem_size=" << share_mem_size << std::endl;

        int32_t output_size = batch_size * embedding_size;
        if ((max_embedding_segment_size > 0) && (embedding_size >= 32) && (embedding_size <= 192)) {
            if (fw_weight_multil_flag) {
                ComputeBatchBoundary<<<DIVUP(sample_feature_size, 1024), 1024, 0, stream>>>(
                    index_tensor, sample_feature_size, batch_size, sample_feature_start_addr, output_size, output);

                dim3 grid_common(1);
                dim3 block_common(32, 32);
                int32_t common_fw_cross_size = (field_num + 1) * fw_field_num * embedding_size;
                ProcessCommonPart<T, 32><<<grid_common, block_common, 0, stream>>>(
                    embedding_size, field_num, fw_field_num, common_fw_cross_size, sample_feature_start_addr, weight_tensor, field_tensor,
                    gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_field_map);

                cudaFuncSetAttribute(ProcessFwffmOutputShare<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);
                dim3 grid(batch_size, embedding_segment_count);
                dim3 block(32, 32);
                ProcessFwffmOutputShare<T><<<grid, block, share_mem_size, stream>>>(
                    embedding_size, embedding_segment_size, field_num, fw_field_num, fw_weight_multil_flag,
                    sample_feature_start_addr, gmem_field_map, gmem_common_cross_mean_sum, gmem_common_cross_square_sum,
                    weight_tensor, field_tensor, fw_weight_tensor, output);

            } else {
                ComputeBatchBoundary<<<DIVUP(sample_feature_size, 1024), 1024, 0, stream>>>(
                    index_tensor, sample_feature_size, batch_size, sample_feature_start_addr, embedding_size, output);

                cudaFuncSetAttribute(ComputeCommonPartOutput<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);
                dim3 grid_common(1, embedding_segment_count);
                dim3 block_common(32, 32);
                int32_t common_fw_cross_size = (field_num + 1) * fw_field_num * embedding_segment_size;
                ComputeCommonPartOutput<T><<<grid_common, block_common, share_mem_size, stream>>>(
                    embedding_size, embedding_segment_size, field_num, fw_field_num, common_fw_cross_size, output_size,
                    sample_feature_start_addr, weight_tensor, fw_weight_tensor, field_tensor, gmem_common_cross_mean_sum, gmem_field_map, output);
                if (batch_size > 1) {
                    dim3 grid_broadcast(batch_size - 1);
                    dim3 block_broadcast(32, 32);
                    BroadcastCommonPartOutput<T><<<grid_broadcast, block_broadcast>>>(embedding_size, output);
                }

                cudaFuncSetAttribute(ProcessFwffmOutputShareV2<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);
                dim3 grid(batch_size, embedding_segment_count);
                dim3 block(32, 32);
                ProcessFwffmOutputShareV2<T><<<grid, block, share_mem_size, stream>>>(
                    embedding_size, embedding_segment_size, field_num, fw_field_num, sample_feature_start_addr,
                    gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_field_map, weight_tensor, fw_weight_tensor,
                    field_tensor, output);
            }

        } else if (embedding_size <= 192) {
            ComputeBatchBoundary<<<DIVUP(sample_feature_size, 1024), 1024, 0, stream>>>(
                index_tensor, sample_feature_size, batch_size, sample_feature_start_addr, output_size, output);

            int32_t shared_mem_required_bytes = (fw_field_num * (field_num + 1) + 1)* sizeof(int32_t);
            if (shared_mem_required_bytes < max_share_mem_size) {
                dim3 grid_common(1);
                dim3 block_common(32, 32);
                int32_t common_fw_cross_size = (field_num + 1) * fw_field_num * embedding_size;
                ProcessCommonPart<T, 32><<<grid_common, block_common, 0, stream>>>(
                    embedding_size, field_num, fw_field_num, common_fw_cross_size, sample_feature_start_addr, weight_tensor, field_tensor,
                    gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_field_map);

                dim3 block_set(embedding_size);
                dim3 grid_set(batch_size * fw_field_num);
                BroadcastCommonPart<T><<<grid_set, block_set, 0, stream>>>(
                    batch_size, embedding_size, field_num, fw_field_num, gmem_common_cross_mean_sum, gmem_common_cross_square_sum,
                    gmem_cross_sum);

                dim3 grid(batch_size);
                dim3 block(32, 32);
                constexpr int32_t warp_num = 32;
                cudaFuncSetAttribute(ProcessFwffmOutput<T, warp_num>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                ProcessFwffmOutput<T, warp_num><<<grid, block, shared_mem_required_bytes, stream>>>(
                    embedding_size, field_num, fw_field_num, fw_weight_multil_flag,
                    sample_feature_start_addr, gmem_field_map, weight_tensor, field_tensor,
                    fw_weight_tensor, output, gmem_cross_sum);
            } else {
                printf("This Fwffm op is not support such input parameters!\n");
            }

        } else {
            printf("This Fwffm op is not support such input parameters!\n");
        }
        return 1;
    }

    template int32_t ComputeSparseFwffmSharedMemory(cudaStream_t stream, const void* const* input, float* output,
    void* worksapce, const int32_t fw_field_num,
    const int32_t fw_weight_size, const bool fw_weight_multil_flag,
    const int32_t sample_feature_size, const int32_t field_num,
    const int32_t embedding_size, const int32_t batch_size);

    template int32_t ComputeSparseFwffmSharedMemory(cudaStream_t stream, const void* const* input, half* output,
    void* worksapce, const int32_t fw_field_num,
    const int32_t fw_weight_size, const bool fw_weight_multil_flag,
    const int32_t sample_feature_size, const int32_t field_num,
    const int32_t embedding_size, const int32_t batch_size);

}  // namespace functor
}  // namespace sparse_fwffm
}  // namespace nvinfer1
