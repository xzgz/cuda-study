#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cfloat>
#include <cmath>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nvinfer1 {
namespace sparse_fwffm {
namespace functor {
template <typename T>
int ComputeSparseFwffm(cudaStream_t stream, const void* const* input, T* output,
                       void* worksapce,
                       const int32_t fw_field_num,
                       const int32_t fw_weight_size,
                       const bool fw_weight_multil_flag,
                       const int32_t sample_feature_size,
                       const int32_t field_num, const int32_t embedding_size,
                       const int32_t batch_size);
}
}
}

namespace nvinfer1 {
namespace sparse_fwffm_shared {
namespace functor {
template <typename T>
int ComputeSparseFwffmSharedMemory(cudaStream_t stream, const void* const* input, T* output,
                                   void* worksapce,
                                   const int32_t fw_field_num,
                                   const int32_t fw_weight_size,
                                   const bool fw_weight_multil_flag,
                                   const int32_t sample_feature_size,
                                   const int32_t field_num, const int32_t embedding_size,
                                   const int32_t batch_size);
}
}
}

using namespace std;

#define CUDA_CHECK(condition)                                    \
/* Code block avoids redefinition of cudaError_t error */    \
do                                                           \
{                                                            \
cudaError_t error = condition;                           \
if (error != cudaSuccess)                                \
{                                                        \
std::cout << cudaGetErrorString(error) << std::endl; \
}                                                        \
} while (0)

void PrepareInput(float *input_ptr, int size)
{
    std::random_device rd;
    default_random_engine e(rd());
    float min = -1.0;
    float max = 1.0;
    uniform_real_distribution<float> u(min, max);
    for (int i = 0; i < size; ++i)
    {
        input_ptr[i] = u(e);
    }
}

void PrepareInput(string &input_file_path, void **input_host, void **input_ptr_vec, void **workspace, void **output)
{
    const int32_t fw_field_num = 3;
    const int32_t fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;
    const int32_t field_num = 2;
    const int32_t real_embedding_size = 4;

    // int batch_size = 395;
    // const int32_t sample_feature_size = 1413;
    int batch_size = 2;
    const int32_t sample_feature_size = 10;

    int32_t feature_weight_size = sample_feature_size * field_num * real_embedding_size;
    int32_t fw_weight_add_size = batch_size * fw_weight_size;
    int32_t feature_field_size = sample_feature_size * field_num;
    int32_t feature_index_size = sample_feature_size;
    int32_t total_input_size = feature_weight_size + fw_weight_add_size + feature_field_size + feature_index_size;
    int32_t total_output_size = batch_size * real_embedding_size;

    size_t bound_data_size = sizeof(int32_t) * (batch_size + 1) * fw_field_num * 2;
    size_t fw_field_map_size = sizeof(int32_t) * batch_size * fw_field_num;
    size_t cross_mean_sum_size = (batch_size + 1) * field_num * fw_field_num * real_embedding_size * sizeof(float);
    size_t corss_cross_mean_square_sum_size = (batch_size + 1) * fw_field_num * real_embedding_size * sizeof(float);
    size_t fw_field_sparse_map_size = sizeof(int32_t) * batch_size * fw_field_num * field_num;
    size_t fw_field_sparse_num_size = sizeof(int32_t) * batch_size;
    size_t workspace_size = bound_data_size + fw_field_map_size + cross_mean_sum_size + corss_cross_mean_square_sum_size + fw_field_sparse_map_size + fw_field_sparse_num_size;

    float *input_buffer = new float[total_input_size];
    *input_host = input_buffer;
    void *input_dev_void = nullptr;
    void *output_dev_void = nullptr;
    void *workspace_dev_void = nullptr;
    CUDA_CHECK(cudaMalloc(&input_dev_void, sizeof(float) * (total_input_size)));
    CUDA_CHECK(cudaMalloc(&output_dev_void, sizeof(float) * (total_output_size)));
    CUDA_CHECK(cudaMalloc(&workspace_dev_void, workspace_size));
    *workspace = workspace_dev_void;
    *output = output_dev_void;

    // std::ifstream reader;
    // reader.open(input_file_path, std::ifstream::binary);
    // reader.read(reinterpret_cast<char *>(input_buffer), sizeof(float) * total_input_size);
    // reader.close();

    float feature_data[] = {
    1., 1., 1., 1.,
    2., 2., 2., 2.,

    3., 3., 3., 3.,
    4., 4., 4., 4.,

    5., 5., 5., 5.,
    6., 6., 6., 6.,

    7., 7., 7., 7.,
    8., 8., 8., 8.,

    9., 9., 9., 9.,
    10., 10., 10., 10.,

    1., 1., 1., 1.,
    2., 2., 2., 2.,

    3., 3., 3., 3.,
    4., 4., 4., 4.,

    5., 5., 5., 5.,
    6., 6., 6., 6.,

    7., 7., 7., 7.,
    8., 8., 8., 8.,

    9., 9., 9., 9.,
    10., 10., 10., 10.};

    float fw_weight[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

    int field_data[] = {1, 1,
                        1, 1,
                        1, 1,
                        2, 2,
                        2, 3,
                        2, 2,
                        2, 2,
                        2, 2,
                        2, 2,
                        2, 3};

    int index_data[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1};

    float *input_dev = (float *)input_dev_void;
    CUDA_CHECK(cudaMemcpy(input_dev, feature_data, sizeof(float) * feature_weight_size, cudaMemcpyHostToDevice));
    input_ptr_vec[0] = input_dev;
    input_dev += feature_weight_size;

    CUDA_CHECK(cudaMemcpy(input_dev, fw_weight, sizeof(float) * fw_weight_add_size, cudaMemcpyHostToDevice));
    input_ptr_vec[1] = input_dev;
    input_dev += fw_weight_add_size;

    CUDA_CHECK(cudaMemcpy(input_dev, field_data, sizeof(float) * feature_field_size, cudaMemcpyHostToDevice));
    input_ptr_vec[2] = input_dev;
    input_dev += feature_field_size;

    CUDA_CHECK(cudaMemcpy(input_dev, index_data, sizeof(float) * feature_index_size, cudaMemcpyHostToDevice));
    input_ptr_vec[3] = input_dev;
    input_dev += feature_index_size;
    // float *input_dev = (float *)input_dev_void;
    // CUDA_CHECK(cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_weight_size, cudaMemcpyHostToDevice));
    // input_ptr_vec[0] = input_dev;
    // input_buffer += feature_weight_size;
    // input_dev += feature_weight_size;

    // CUDA_CHECK(cudaMemcpy(input_dev, input_buffer, sizeof(float) * fw_weight_add_size, cudaMemcpyHostToDevice));
    // input_ptr_vec[1] = input_dev;
    // input_buffer += fw_weight_add_size;
    // input_dev += fw_weight_add_size;

    // cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_field_size, cudaMemcpyHostToDevice);
    // input_ptr_vec[2] = input_dev;
    // input_buffer += feature_field_size;
    // input_dev += feature_field_size;

    // cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_index_size, cudaMemcpyHostToDevice);
    // input_ptr_vec[3] = input_dev;
}

void DeleteInput(void **input_host, void **input_ptr_vec, void **workspace, void **output)
{
    cudaFree(input_ptr_vec[0]);
    cudaFree(*workspace);
    cudaFree(*output);
    float *input_ptr = (float *)(*input_host);
    delete[] input_ptr;
}

void CompareResult(const float *ptr1, const float *ptr2, int size)
{
    const float eps = 1e-6;
    float max_abs_err = -FLT_MAX;
    int max_abs_err_loc = -1;
    for (int i = 0; i < size; ++i)
    {
        float abs_err = abs(ptr1[i] - ptr2[i]);
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            max_abs_err_loc = i;
        }
    }
    if (max_abs_err_loc != -1)
    {
        std::cout << "max_abs_err: " << max_abs_err << ", loc: " << max_abs_err_loc
        << ", val1: " << ptr1[max_abs_err_loc] << ", val2: " << ptr2[max_abs_err_loc] << std::endl;
    }

    for (int i = 0; i < size; ++i)
    {
        if (abs(ptr1[i] - ptr2[i]) > eps)
        {
            std::cout << "loc: " << i << ", val1: " << ptr1[i] << ", val2: " << ptr2[i]
            << ", abs(err): " << abs(ptr1[i] - ptr2[i]) << endl;
            std::cout << "Compare not pass.\n";
            return;
        }
    }
    std::cout << "Compare pass.\n";

    return;
}

int main()
{
    void *input_host = nullptr;
    void *output_dev = nullptr;
    void *workspace_dev = nullptr;
    const int32_t input_num = 4;
    void **input_dev_ptr_vec = (void **)(new float *[input_num]);
    for (int i = 0; i < input_num; ++i)
        input_dev_ptr_vec[i] = nullptr;

    std::string input_file_path = "../../data/fwffm_data/sparse_fwffm_input_data2.bin";
    std::string result_file_path = "../../data/fwffm_data/sparse_fwffm_input_data_result2.bin";
    PrepareInput(input_file_path, &input_host, input_dev_ptr_vec, &workspace_dev, &output_dev);

    const int32_t fw_field_num = 3;
    const int32_t fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;
    const int32_t field_num = 2;
    const bool fw_weight_multil_flag = false;
    const int32_t real_embedding_size = 4;
    //const int32_t test_embedding_size = 96;
    const int32_t test_embedding_size = 4;

    // int batch_size = 395;
    // const int32_t sample_feature_size = 1413;
    int batch_size = 2;
    const int32_t sample_feature_size = 10;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t output_size = batch_size * test_embedding_size;
    float *output_host1 = new float[output_size];
    float *output_host2 = new float[output_size];

//    CUDA_CHECK(cudaMemset(output_dev, 0, sizeof(float) * batch_size * real_embedding_size));
    nvinfer1::sparse_fwffm::functor::ComputeSparseFwffm<float>(
            stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num,
            fw_weight_size, fw_weight_multil_flag, sample_feature_size, field_num, test_embedding_size,
            batch_size);
    cudaMemcpy((void *)output_host1, output_dev, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) {
        std::cout << output_host1[i] << " ";
    }
    std::cout << std::endl;

//    CUDA_CHECK(cudaMemset(output_dev, 0, sizeof(float) * batch_size * real_embedding_size));
    nvinfer1::sparse_fwffm_shared::functor::ComputeSparseFwffmSharedMemory<float>(
    stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num,
    fw_weight_size, fw_weight_multil_flag, sample_feature_size, field_num, test_embedding_size,
    batch_size);
    cudaMemcpy((void *)output_host2, output_dev, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) {
        std::cout << output_host2[i] << " ";
    }
    std::cout << std::endl;

    CompareResult(output_host1, output_host2, output_size);

    DeleteInput(&input_host, input_dev_ptr_vec, &workspace_dev, &output_dev);
    delete[] output_host1;
    delete[] output_host2;

    return 0;
}
