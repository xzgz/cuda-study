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

#define ALIGN_UP(x, align_count)    (((x) + ((align_count) - 1)) / (align_count) * (align_count))
#define DATA_ALIGN_BYTE_COUNT       128
#define DATA_ALIGN_INT32_COUNT      32

namespace nvinfer1 {

namespace sparse_fipnn_shared_multi {

template <typename T = float>
__device__ T __reduce_sum_across_warp(T val) {
    T rtn = val;
    //    __syncwarp(0xFFFFFFFF);
    for (int32_t i = 1; i < 32; i *= 2) {
        rtn += __shfl_xor_sync(0xFFFFFFFF, rtn, i);
    }
    return rtn;
}

__global__ void ComputeBatchBoundary(
    const int32_t* index_tensor, void* workspace,
    const int32_t* sample_count_prefix_sum_vec, const int32_t* sample_count_vec,
    const int32_t* batch_size_prefix_sum_vec, const int32_t* batch_size_vec) {
    int32_t total_thread = gridDim.x * blockDim.x * blockDim.y;
    int32_t tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int32_t batch_size = batch_size_vec[blockIdx.y];
    int32_t total_feature_num = sample_count_vec[blockIdx.y];

    uint32_t bound_data_offset = uint32_t(batch_size_prefix_sum_vec[blockIdx.y]) + blockIdx.y * DATA_ALIGN_INT32_COUNT;
    bound_data_offset = ALIGN_UP(bound_data_offset, DATA_ALIGN_INT32_COUNT);

    int32_t* sample_feature_start_addr = static_cast<int32_t*>(workspace) + bound_data_offset;
    const int32_t* index_ptr = index_tensor + sample_count_prefix_sum_vec[blockIdx.y];

    if (tid < batch_size + 1) {
        sample_feature_start_addr[tid] = 0;
    }
    __syncthreads();

    if (tid < total_feature_num) {
        int32_t idx = index_ptr[tid];
        if (tid > 0) {
            int32_t pre_idx = index_ptr[tid - 1];
            for (int32_t i = idx; i > pre_idx; --i) {
                sample_feature_start_addr[i] = tid;
            }
        } else {
            int32_t first_idx = index_ptr[0];
            for (int32_t i = 0; i <= first_idx; ++i) {
                sample_feature_start_addr[i] = 0;
            }
            int32_t last_idx = index_ptr[total_feature_num - 1];
            for (int32_t i = batch_size - 1; i > last_idx; --i) {
                sample_feature_start_addr[i] = total_feature_num;
            }
            sample_feature_start_addr[batch_size] = total_feature_num;
        }
    }
}

template <typename T>
__global__ void ComputeCommonPartOutput(
    const T* multi_weight_tensor, const int32_t* multi_field_tensor, void* workspace,
    const int32_t* sample_count_prefix_sum_vec, const int32_t* batch_size_prefix_sum_vec,
    const uint32_t common_cross_start_offset, const uint32_t fw_field_map_start_offset,
    const uint32_t common_output_start_offset, const uint32_t embedding_size,
    const uint32_t embedding_segment_size, const int32_t field_num, const uint32_t fw_field_num,
    const uint32_t segment_common_fw_cross_size, const uint32_t fw_weight_size) {

    const uint32_t data_align_count = DATA_ALIGN_BYTE_COUNT / sizeof(T);
    uint32_t common_cross_offset = common_cross_start_offset + blockIdx.y * ((field_num + 1) * fw_field_num * embedding_size + data_align_count);
    common_cross_offset = ALIGN_UP(common_cross_offset, data_align_count);
    uint32_t common_output_offset = common_output_start_offset + blockIdx.y * (fw_weight_size + data_align_count);
    common_output_offset = ALIGN_UP(common_output_offset, data_align_count);
    uint32_t bound_data_offset = uint32_t(batch_size_prefix_sum_vec[blockIdx.y]) + blockIdx.y * DATA_ALIGN_INT32_COUNT;
    bound_data_offset = ALIGN_UP(bound_data_offset, DATA_ALIGN_INT32_COUNT);
    uint32_t fw_field_map_offset = fw_field_map_start_offset + blockIdx.y * (2 * (fw_field_num + 1) + DATA_ALIGN_INT32_COUNT);
    fw_field_map_offset = ALIGN_UP(fw_field_map_offset, DATA_ALIGN_INT32_COUNT);

    const T* weight_tensor = multi_weight_tensor + sample_count_prefix_sum_vec[blockIdx.y] * field_num * embedding_size;
    const int32_t* field_tensor = multi_field_tensor + sample_count_prefix_sum_vec[blockIdx.y] * field_num;
    T* gmem_fw_cross_mean_sum = static_cast<T*>(workspace) + common_cross_offset;
    T* common_output = static_cast<T*>(workspace) + common_output_offset;
    int32_t* sample_feature_start_addr = static_cast<int32_t*>(workspace) + bound_data_offset;
    int32_t* gmem_fw_field_map = static_cast<int32_t*>(workspace) + fw_field_map_offset;

    extern __shared__ float smem_pool[];
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
    int32_t embedding_segment_start = blockIdx.x * embedding_segment_size;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t common_feature_num = sample_feature_start_addr[0];

    for (int32_t i = tid; i < segment_common_fw_cross_size; i += total_thread) {
        smem_cross_mean_sum[i] = T(0);
        if (i < fw_field_num) {
            smem_fw_field_map[i] = -1;
        }
    }
    for (int32_t i = tid; i < fw_weight_size; i += total_thread) {
        smem_output[i] = T(0);
    }
    if (blockIdx.x == 0) {
        for (int32_t i = tid; i < fw_weight_size; i += total_thread) {
            common_output[i] = T(0);
        }
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

    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];
    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
    fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_fw_field_map[fw_field_1];
        int32_t fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;

        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;
        T output_value = T(0);
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_value += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum);
        }
        output_value = __reduce_sum_across_warp(output_value);
        // store here
        if (lane_id == 0) {
            smem_output[fw_iter + fw_field_1] = output_value;
        }
        //        __syncwarp(0xFFFFFFFF);

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

            T output_value = T(0);
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_value += mean_index_1_sum * mean_index_2_sum;
            }
            output_value = __reduce_sum_across_warp(output_value);
            // store here
            if (lane_id == 0) {
                smem_output[fw_iter + fw_field_2] = output_value;
            }
            //            __syncwarp(0xFFFFFFFF);
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

    for (int32_t i = tid; i < fw_weight_size; i += total_thread) {
        atomicAdd(common_output + i, smem_output[i]);
//        common_output[i] += smem_output[i];
//        atomicAdd(common_output + i, smem_output[i] / T(gridDim.x));
    }
}

template <typename T>
__device__ void AccumulateSamplePart(
int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx, T* smem_cross_mean_sum,
T* smem_cross_square_sum, T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
int32_t* gmem_common_field_map, int32_t embedding_size, int32_t embedding_segment_size,
int32_t field_num, int32_t fw_field_num, int32_t this_sample_feature_num, int32_t this_sample_feature_start_addr,
const T* weight_tensor, const int32_t* field_tensor) {
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
T* smem_cross_mean_sum, T* smem_cross_square_sum, int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
T* output) {
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
    //        printf("[FIPNN] common_vaild_fw_field=%d\n", common_vaild_fw_field);
    //        printf("[FIPNN] sample_vaild_fw_field=%d\n", sample_vaild_fw_field);
    //    }

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < sample_vaild_fw_field; fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_sample_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_sample_fw_field_map[fw_field_1];
        int32_t fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;

        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;
        T output_value = T(0);
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_value += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum);
        }
        output_value = __reduce_sum_across_warp(output_value);
        // store here
        if (lane_id == 0) {
            atomicAdd(output + fw_iter + fw_field_1, output_value);
        }
        __syncwarp(0xFFFFFFFF);

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

            T output_value = T(0);
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_value += mean_index_1_sum * mean_index_2_sum;
            }
            output_value = __reduce_sum_across_warp(output_value);
            // store here
            if (lane_id == 0) {
                atomicAdd(output + fw_iter + fw_field_2, output_value);
            }
            __syncwarp(0xFFFFFFFF);
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

            T output_value = T(0);
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_value += mean_index_1_sum * mean_index_2_sum;
            }
            output_value = __reduce_sum_across_warp(output_value);
            // store here
            if (lane_id == 0) {
                atomicAdd(output + fw_iter + fw_field_1, output_value);
            }
            __syncwarp(0xFFFFFFFF);
        }

    }
}

template <typename T>
__device__ void ProcessSamplePartShare(
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

    for (int32_t i = tid; i < fw_field_num; i += total_thread) {
        smem_fw_field_map[i] = gmem_common_field_map[i];
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

template <typename T = float>
__device__ void ProcessOutputShare(
int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num, int32_t fw_field_num,
T* smem_cross_mean_sum, T* smem_cross_square_sum, int32_t* smem_fw_field_map, int32_t* smem_fw_map_idx,
T* output) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    int32_t segment_size_div_blockdimx = embedding_segment_size / blockDim.x;
    int32_t embedding_segment_start = blockIdx.y * embedding_segment_size;
    int32_t total_vaild_fw_field = smem_fw_map_idx[fw_field_num];

    for (int32_t fw_field_1_idx = warp_id; fw_field_1_idx < total_vaild_fw_field;
    fw_field_1_idx += blockDim.y) {
        int32_t fw_field_1 = smem_fw_map_idx[fw_field_1_idx];
        int32_t field_1 = smem_fw_field_map[fw_field_1];
        int32_t fw_iter = fw_field_1 * (fw_field_1 + 1) / 2;

        for (int32_t fw_field_2_idx = 0; fw_field_2_idx < fw_field_1_idx; fw_field_2_idx++) {
            int32_t fw_field_2 = smem_fw_map_idx[fw_field_2_idx];
            int32_t field_2 = smem_fw_field_map[fw_field_2];
            int32_t index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_segment_size;
            int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_segment_size;

            T output_value = T(0);
#pragma unroll
            for (int n = 0; n < segment_size_div_blockdimx; ++n) {
                T mean_index_1_sum = T(0);
                T mean_index_2_sum = T(0);
                if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                    mean_index_1_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                    mean_index_2_sum = smem_cross_mean_sum[index_2 + n * blockDim.x + lane_id];
                }
                output_value += mean_index_1_sum * mean_index_2_sum;
            }
            output_value = __reduce_sum_across_warp(output_value);
            // store here
            if (lane_id == 0) {
                atomicAdd(output + fw_iter + fw_field_2, output_value);
            }
            __syncwarp(0xFFFFFFFF);
        }

        int32_t index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_segment_size;
        T output_value = T(0);
#pragma unroll
        for (int n = 0; n < segment_size_div_blockdimx; ++n) {
            T cross_mean_sum = T(0);
            T cross_square_sum = T(0);
            if (embedding_segment_start + n * blockDim.x + lane_id < embedding_size) {
                cross_mean_sum = smem_cross_mean_sum[index_1 + n * blockDim.x + lane_id];
                cross_square_sum = smem_cross_square_sum[fw_field_1 * embedding_segment_size + n * blockDim.x + lane_id];
            }
            output_value += T(0.5) * (cross_mean_sum * cross_mean_sum - cross_square_sum);
        }
        output_value = __reduce_sum_across_warp(output_value);
        // store here
        if (lane_id == 0) {
            atomicAdd(output + fw_iter + fw_field_1, output_value);
        }
        __syncwarp(0xFFFFFFFF);
    }
}

template <typename T = float>
__global__ void SparseFIPNNGpuShare(
int32_t embedding_size, int32_t embedding_segment_size, int32_t field_num,
int32_t fw_field_num, int32_t* sample_feature_start_addr,
T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
int32_t* gmem_common_field_map, const T* weight_tensor,
const int32_t* field_tensor, T* output_tensor,
T* workspace) {
    extern __shared__ float smem_pool[];
    int32_t batch_id = blockIdx.x;
    int32_t fw_weight_size = (fw_field_num + 1) * fw_field_num / 2;
    int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
    int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * (fw_field_num + 1);

    T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + 2 * (fw_field_num + 1));
    T* smem_cross_square_sum = smem_cross_mean_sum + embedding_segment_size * field_num * fw_field_num;

    T* gmem_output = output_tensor + batch_id * fw_weight_size;
    int32_t this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
    int32_t this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
    int32_t this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

    ProcessSamplePartShare<T>(
    smem_fw_field_map, smem_fw_map_idx, smem_cross_mean_sum, smem_cross_square_sum,
    gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_common_field_map,
    embedding_size, embedding_segment_size, field_num, fw_field_num, this_sample_feature_num,
    this_sample_feature_start_addr, weight_tensor, field_tensor);
    ProcessOutputShare<T>(
    embedding_size, embedding_segment_size, field_num, fw_field_num, smem_cross_mean_sum,
    smem_cross_square_sum, smem_fw_field_map, smem_fw_map_idx, gmem_output);
}

template <typename T = float>
__global__ void SparseFIPNNGpuShareV2(
    const T* multi_weight_tensor, const int32_t* multi_field_tensor, void* workspace,
    const int32_t* sample_count_prefix_sum_vec, const int32_t* batch_size_prefix_sum_vec, const int32_t* batch_size_vec,
    const uint32_t common_cross_start_offset, const uint32_t fw_field_map_start_offset,
    const uint32_t common_output_start_offset, const int32_t embedding_size,
    const int32_t embedding_segment_size, const int32_t field_num,
    const int32_t fw_field_num, const int32_t fw_weight_size,
    T* multi_output_tensor) {

    const int32_t batch_size = batch_size_vec[blockIdx.z];
    if (blockIdx.x < batch_size) {
        const uint32_t data_align_count = DATA_ALIGN_BYTE_COUNT / sizeof(T);
        uint32_t common_cross_offset = common_cross_start_offset + blockIdx.z * ((field_num + 1) * fw_field_num * embedding_size + data_align_count);
        common_cross_offset = ALIGN_UP(common_cross_offset, data_align_count);
        uint32_t common_output_offset = common_output_start_offset + blockIdx.z * (fw_weight_size + data_align_count);
        common_output_offset = ALIGN_UP(common_output_offset, data_align_count);
        uint32_t bound_data_offset = uint32_t(batch_size_prefix_sum_vec[blockIdx.z]) + blockIdx.z * DATA_ALIGN_INT32_COUNT;
        bound_data_offset = ALIGN_UP(bound_data_offset, DATA_ALIGN_INT32_COUNT);
        uint32_t fw_field_map_offset = fw_field_map_start_offset + blockIdx.z * (2 * (fw_field_num + 1) + DATA_ALIGN_INT32_COUNT);
        fw_field_map_offset = ALIGN_UP(fw_field_map_offset, DATA_ALIGN_INT32_COUNT);

        const T* weight_tensor = multi_weight_tensor + sample_count_prefix_sum_vec[blockIdx.z] * field_num * embedding_size;
        const int32_t* field_tensor = multi_field_tensor + sample_count_prefix_sum_vec[blockIdx.z] * field_num;
        T* gmem_common_cross_mean_sum = static_cast<T*>(workspace) + common_cross_offset;
        T* gmem_common_cross_square_sum = gmem_common_cross_mean_sum + field_num * fw_field_num + embedding_size;
        T* common_output = static_cast<T*>(workspace) + common_output_offset;
        T* output_tensor = multi_output_tensor + batch_size_prefix_sum_vec[blockIdx.z] * fw_weight_size;
        int32_t* sample_feature_start_addr = static_cast<int32_t*>(workspace) + bound_data_offset;
        int32_t* gmem_common_field_map = static_cast<int32_t*>(workspace) + fw_field_map_offset;

        extern __shared__ float smem_pool[];
        int32_t* smem_fw_field_map = reinterpret_cast<int32_t*>(smem_pool);
        int32_t* smem_fw_map_idx = smem_fw_field_map + 2 * (fw_field_num + 1);
        T* smem_cross_mean_sum = reinterpret_cast<T*>(smem_fw_map_idx + 2 * (fw_field_num + 1));
        T* smem_cross_square_sum = smem_cross_mean_sum + embedding_segment_size * field_num * fw_field_num;

        int32_t batch_id = blockIdx.x;
        T* gmem_output = output_tensor + batch_id * fw_weight_size;
        int32_t this_sample_feature_start_addr = sample_feature_start_addr[batch_id];
        int32_t this_sample_feature_end_addr = sample_feature_start_addr[batch_id + 1];
        int32_t this_sample_feature_num = this_sample_feature_end_addr - this_sample_feature_start_addr;

        int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
        int32_t total_thread = blockDim.x * blockDim.y;
        if (blockIdx.y == 0) {
            for (int32_t i = tid; i < fw_weight_size; i += total_thread) {
//                gmem_output[i] += common_output[i];
                atomicAdd(gmem_output + i, common_output[i]);
            }
        }

        AccumulateSamplePart<T>(
            smem_fw_field_map, smem_fw_map_idx, smem_cross_mean_sum, smem_cross_square_sum,
            gmem_common_cross_mean_sum, gmem_common_cross_square_sum, gmem_common_field_map,
            embedding_size, embedding_segment_size, field_num, fw_field_num, this_sample_feature_num,
            this_sample_feature_start_addr, weight_tensor, field_tensor);
        ComputeCommonSamplePartOutput<T>(
            embedding_size, embedding_segment_size, field_num, fw_field_num,
            smem_cross_mean_sum, smem_cross_square_sum, smem_fw_field_map, smem_fw_map_idx, gmem_output);
    }
}

template <typename T = float, int32_t warp_num = 32>
__global__ void ProcessCommonPart(int32_t embedding_size, int32_t field_num, int32_t fw_field_num,
                                  int32_t common_fw_cross_size, int32_t output_size,
                                  int32_t* sample_feature_start_addr, const T* weight_tensor,
                                  const int32_t* field_tensor, T* gmem_fw_cross_mean_sum,
                                  T* gmem_fw_cross_square_sum, int32_t* gmem_fw_field_map,
                                  T* output) {
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    int32_t global_warp_id = blockIdx.x * warp_num + warp_id;
    int32_t total_global_warp_num = gridDim.x * warp_num;

    int32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t total_thread = warp_num * 32;
    int32_t common_feature_num = sample_feature_start_addr[0];

    for (int32_t i = tid; i < common_fw_cross_size; i += total_thread) {
        gmem_fw_cross_mean_sum[i] = 0;
        if (i < fw_field_num) {
            gmem_fw_field_map[i] = -1;
        }
    }
    for (int32_t i = tid; i < output_size; i += total_thread) {
        output[i] = 0;
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
                T square = reg * reg;
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
                T square = reg * reg;
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

template <typename T = float, int32_t warp_num = 32>
__global__ void SparseFIPNNGpu(int32_t weight_size, int32_t field_num, int32_t fw_field_num,
                               int32_t* sample_feature_start_addr,
                               T* gmem_common_cross_mean_sum, T* gmem_common_cross_square_sum,
                               int32_t* gmem_common_field_map, const T* weight_tensor,
                               const int32_t* field_tensor, T* output_tensor,
                               T* workspace  // for mean_sum and square_sum
                               ) {
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
}

namespace functor {
    template <typename T>
    int32_t ComputeSparseFipnnSharedMemoryMultiExample(
    cudaStream_t stream, const void* const* input, T* output, void* workspace,
    const int32_t total_batch_size, const int32_t example_count,
    const int32_t max_sample_count, const int32_t max_batch_size,
    const int32_t fw_field_num, const int32_t field_num, const int32_t embedding_size) {
        int32_t fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;
        int32_t output_size = total_batch_size * fw_weight_size;

        int32_t max_share_mem_size = 65536;
        int32_t max_embedding_segment_size =
        (max_share_mem_size - 4 * (fw_field_num + 1) * sizeof(int32_t) - fw_weight_size * sizeof(T))
        / ((field_num + 1) * fw_field_num * sizeof(T));
        max_embedding_segment_size = max_embedding_segment_size / 32 * 32;
        int32_t embedding_segment_count = (embedding_size + max_embedding_segment_size - 1) / max_embedding_segment_size;
        int32_t embedding_segment_size = embedding_size / embedding_segment_count;
        embedding_segment_size = (embedding_segment_size + 31) / 32 * 32;
        int32_t share_mem_size = 4 * (fw_field_num + 1) * sizeof(int32_t) + fw_weight_size * sizeof(T)
        + embedding_segment_size * (field_num + 1) * fw_field_num * sizeof(T);

        const T* weight_tensor = static_cast<const T*>(input[0]);
        const int32_t* field_tensor = static_cast<const int32_t*>(input[1]);
        const int32_t* index_tensor = static_cast<const int32_t*>(input[2]);
        const int32_t* sample_count_prefix_sum_vec = static_cast<const int32_t*>(input[3]);
        const int32_t* sample_count_vec = static_cast<const int32_t*>(input[4]);
        const int32_t* batch_size_prefix_sum_vec = static_cast<const int32_t*>(input[5]);
        const int32_t* batch_size_vec = static_cast<const int32_t*>(input[6]);

        uint32_t data_size = (total_batch_size + example_count) * sizeof(int32_t) + example_count * DATA_ALIGN_BYTE_COUNT;
        uint32_t fw_field_map_start_offset = ALIGN_UP(data_size, DATA_ALIGN_BYTE_COUNT) / sizeof(int32_t);

        data_size += example_count * (2 * (fw_field_num + 1) * sizeof(int32_t) + DATA_ALIGN_BYTE_COUNT);
        uint32_t common_output_start_offset = ALIGN_UP(data_size, DATA_ALIGN_BYTE_COUNT) / sizeof(T);

        data_size += example_count * (fw_weight_size * sizeof(T) + DATA_ALIGN_BYTE_COUNT);
        uint32_t sample_cross_start_offset = ALIGN_UP(data_size, DATA_ALIGN_BYTE_COUNT) / sizeof(T);

        data_size += (total_batch_size * (field_num + 1) * fw_field_num * embedding_size * sizeof(T)
                  + example_count * DATA_ALIGN_BYTE_COUNT);
        uint32_t common_cross_start_offset = ALIGN_UP(data_size, DATA_ALIGN_BYTE_COUNT) / sizeof(T);

        cudaMemsetAsync(output, 0, output_size * sizeof(T), stream);
        dim3 block_boundary(32, 32);
        dim3 grid_boundary(DIVUP(max_sample_count, 1024), example_count);
        if ((max_embedding_segment_size > 0) && (embedding_size >= 32)) {
            ComputeBatchBoundary<<<grid_boundary, block_boundary, 0, stream>>>(
                index_tensor, workspace, sample_count_prefix_sum_vec, sample_count_vec,
                batch_size_prefix_sum_vec, batch_size_vec);

            cudaFuncSetAttribute(ComputeCommonPartOutput<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);
            dim3 grid_common(embedding_segment_count, example_count);
            dim3 block_common(32, 32);
            int32_t common_fw_cross_size = (field_num + 1) * fw_field_num * embedding_segment_size;
            ComputeCommonPartOutput<T><<<grid_common, block_common, share_mem_size, stream>>>(
                weight_tensor, field_tensor, workspace, sample_count_prefix_sum_vec, batch_size_prefix_sum_vec,
                common_cross_start_offset, fw_field_map_start_offset, common_output_start_offset,
                embedding_size, embedding_segment_size, field_num, fw_field_num,
                common_fw_cross_size, fw_weight_size);

            cudaFuncSetAttribute(SparseFIPNNGpuShareV2<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);
            dim3 grid(max_batch_size, embedding_segment_count, example_count);
            dim3 block(32, 32);
            SparseFIPNNGpuShareV2<T><<<grid, block, share_mem_size, stream>>>(
                weight_tensor, field_tensor, workspace,
                sample_count_prefix_sum_vec, batch_size_prefix_sum_vec, batch_size_vec,
                common_cross_start_offset, fw_field_map_start_offset, common_output_start_offset,
                embedding_size, embedding_segment_size, field_num, fw_field_num, fw_weight_size,
                output);

        } else if (embedding_size <= 192) {
            printf("This Fipnn op will support such input parameters soon!\n");
        } else {
            printf("This Fipnn op is not support such input parameters!\n");
        }

        return 1;
    }

    template int32_t ComputeSparseFipnnSharedMemoryMultiExample(
    cudaStream_t stream, const void* const* input, float* output, void* workspace,
    const int32_t total_batch_size, const int32_t example_count,
    const int32_t max_sample_count, const int32_t max_batch_size,
    int32_t fw_field_num, const int32_t field_num, const int32_t embedding_size);
    template int32_t ComputeSparseFipnnSharedMemoryMultiExample(
    cudaStream_t stream, const void* const* input, half* output, void* workspace,
    const int32_t total_batch_size, const int32_t example_count,
    const int32_t max_sample_count, const int32_t max_batch_size,
    int32_t fw_field_num, const int32_t field_num, const int32_t embedding_size);

}  // namespace functor
}  // namespace sparse_fipnn_shared_multi
}  // namespace nvinfer1
