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
namespace sparse_fipnn {
namespace functor {
template <typename T>
int32_t ComputeSparseFipnn(cudaStream_t stream, const void* const* input, T* output,
                           void* workspace, const int32_t fw_field_num,
                           const int32_t sample_feature_size, const int32_t field_num,
                           const int32_t field_neuron_size, const int32_t batch_size);
}
}
}

namespace nvinfer1 {
namespace sparse_fipnn_shared {
namespace functor {
template <typename T>
int32_t ComputeSparseFipnnSharedMemory(cudaStream_t stream, const void* const* input, T* output,
                                       void* workspace, const int32_t fw_field_num,
                                       const int32_t sample_feature_size, const int32_t field_num,
                                       const int32_t field_neuron_size, const int32_t batch_size);
}
}
}

using namespace std;

#define CUDA_CHECK(condition)                                    \
/* Code block avoids redefinition of cudaError_t error */    \
do {                                                         \
cudaError_t error = condition;                           \
if (error != cudaSuccess) {                              \
std::cout << cudaGetErrorString(error) << std::endl; \
}                                                        \
} while (0)

void PrepareInput(float *input_ptr, int size) {
    std::random_device rd;
    default_random_engine e(rd());
    float min = -1.0;
    float max = 1.0;
    uniform_real_distribution<float> u(min, max);
    for (int i = 0; i < size; ++i) {
        input_ptr[i] = u(e);
    }
}

void PrepareInput(void **input_host, void **input_ptr_vec, void **workspace, void **output,
                  const int32_t fw_field_num, const int32_t sample_feature_size,
                  const int32_t batch_size, const int32_t field_num,
                  const int32_t real_embedding_size, const int32_t fw_weight_add_size,
                  std::string input_file_path) {
    const int32_t fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;
    int32_t feature_weight_size = sample_feature_size * field_num * real_embedding_size;
    int32_t feature_field_size = sample_feature_size * field_num;
    int32_t feature_index_size = sample_feature_size;
    int32_t total_input_size = feature_weight_size + fw_weight_add_size + feature_field_size + feature_index_size;
    uint64_t total_output_size = batch_size * fw_weight_size;

//    uint64_t bound_data_size = (batch_size + 1) * sizeof(int32_t);
//    uint64_t fw_field_map_size = 2 * (fw_field_num + 1) * sizeof(int32_t);
//    uint64_t cross_mean_sum_size = (batch_size + 1) * field_num * fw_field_num * real_embedding_size * sizeof(float);
//    uint64_t corss_cross_mean_square_sum_size = (batch_size + 1) * fw_field_num * real_embedding_size * sizeof(float);
//    uint64_t workspace_size = bound_data_size + fw_field_map_size + cross_mean_sum_size
//                            + corss_cross_mean_square_sum_size;

    uint64_t data_align_count = 128 / sizeof(int32_t);
    uint64_t data_offset = batch_size + 1;
    data_offset = (data_offset + data_align_count - 1) / data_align_count * data_align_count;
    uint64_t bound_data_size = data_offset * sizeof(int32_t);

    data_offset = 2 * (fw_field_num + 1);
    data_offset = (data_offset + data_align_count - 1) / data_align_count * data_align_count;
    uint64_t fw_field_map_size = data_offset * sizeof(int32_t);

    uint64_t common_output_size = (fw_weight_size + data_align_count - 1) / data_align_count * data_align_count * sizeof(float);
    uint64_t cross_mean_sum_size = (batch_size + 1) * field_num * fw_field_num * real_embedding_size * sizeof(float);
    uint64_t cross_mean_square_sum_size = (batch_size + 1) * fw_field_num * real_embedding_size * sizeof(float);
    uint64_t workspace_size = bound_data_size + fw_field_map_size + common_output_size + cross_mean_sum_size + cross_mean_square_sum_size;

//    uint64_t workspace_size = uint64_t(batch_size) * field_neuron_size * field_num * fw_field_num * 2 * sizeof(float)
//            + 2 * (batch_size + 1) * sizeof(int32_t);

    //    std::cout << "total_input_size=" << total_input_size << std::endl;
    //    std::cout << "total_output_size=" << total_output_size << std::endl;
    //    std::cout << "workspace_size=" << workspace_size << std::endl;

    float *input_buffer = new float[total_input_size];
    *input_host = input_buffer;
    void *input_dev_void = nullptr;
    void *output_dev_void = nullptr;
    void *workspace_dev_void = nullptr;
    CUDA_CHECK(cudaMalloc(&input_dev_void, sizeof(float) * (total_input_size - fw_weight_add_size)));
    CUDA_CHECK(cudaMalloc(&output_dev_void, sizeof(float) * (total_output_size)));
    CUDA_CHECK(cudaMalloc(&workspace_dev_void, workspace_size));
    *workspace = workspace_dev_void;
    *output = output_dev_void;

    std::ifstream reader;
    reader.open(input_file_path, std::ifstream::binary);
    reader.read(reinterpret_cast<char*>(input_buffer), sizeof(float) * total_input_size);
    reader.close();

    float *input_dev = (float *)input_dev_void;
    double sum = 0;
    for (int i = 0; i < feature_weight_size; ++i) {
        //    for (int i = 2893 * 2 * real_embedding_size; i < 2918 * 2 * real_embedding_size; ++i) {
        sum += input_buffer[i];
        if (isnan(input_buffer[i]) || isinf(input_buffer[i])) std::cout << "the " << i << "th number is nan or inf\n";

        //        if (i % (2 * real_embedding_size) == 0) std::cout << std::endl;
        //        std::cout << input_buffer[i] << " ";
    }
    //    std::cout << "\nsum = " << sum << std::endl;
    cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_weight_size, cudaMemcpyHostToDevice);
    //    for (int i = 0; i < 5; ++i) std::cout << input_buffer[i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << input_buffer[feature_weight_size / 2 - 5 + i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << input_buffer[feature_weight_size - 5 + i] << "\t";
    //    std::cout << std::endl;
    input_ptr_vec[0] = input_dev;
    input_buffer += feature_weight_size;
    input_dev += feature_weight_size;
    input_buffer += fw_weight_add_size;

    sum = 0;
    for (int i = 0; i < feature_field_size; ++i) {
        //    for (int i = 0 * field_num; i < 124 * field_num; ++i) {
        int32_t val = ((int32_t *)input_buffer)[i];
        sum += val;
        if (1 > val || fw_field_num < val)
            std::cout << "The " << i << "th number is invalid field.\n";
        if (i % 2 == 0 && !(val == 1 || val == 2))
            std::cout << "The " << i << "th number is invalid field.\n";

        //        if (i % field_num == 0) std::cout << std::endl;
        //        std::cout << val << " ";
    }
    //    std::cout << "\nsum = " << sum << std::endl;
    cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_field_size, cudaMemcpyHostToDevice);
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_field_size / 2 - 5 + i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_field_size - 5 + i] << "\t";
    //    std::cout << std::endl;
    input_ptr_vec[1] = input_dev;
    input_buffer += feature_field_size;
    input_dev += feature_field_size;

    //    std::cout << std::endl;
    sum = 0;
    for (int i = 0; i < feature_index_size; ++i) {
        //    for (int i = 2893; i < 2918; ++i) {
        int32_t val = ((int32_t *)input_buffer)[i];
        sum += val;
        if (-1 > val || batch_size <= val)
//            std::cout << "The " << i << "th number is invalid index.\n";
                if (val == -1) std::cout << i << " ";
        //        std::cout << val << " ";
    }
//        std::cout << "\nsum = " << sum << std::endl;
//        std::cout << std::endl;
    cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_index_size, cudaMemcpyHostToDevice);
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_index_size / 2 - 5 + i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_index_size - 5 + i] << "\t";
    //    std::cout << std::endl;
    input_ptr_vec[2] = input_dev;
}

void DeleteInput(void **input_host, void **input_ptr_vec, void **workspace, void **output) {
    cudaFree(input_ptr_vec[0]);
    cudaFree(*workspace);
    cudaFree(*output);
}

void CompareResult(const float *ptr1, const float *ptr2, int size) {
    const float eps = 1e-6;
    float max_abs_err = -FLT_MAX;
    int max_abs_err_loc = -1;
    for (int i = 0; i < size; ++i) {
        float abs_err = abs(ptr1[i] - ptr2[i]);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_abs_err_loc = i;
        }
    }
    if (max_abs_err_loc != -1) {
        std::cout << "max_abs_err: " << max_abs_err << ", loc: " << max_abs_err_loc
        <<", val1: " << ptr1[max_abs_err_loc] << ", val2: " << ptr2[max_abs_err_loc] << std::endl;
    }

    for (int i = 0; i < size; ++i) {
        //    for (int i = 0; i < 280; ++i) {
        if (abs(ptr1[i] - ptr2[i]) > eps) {
            std::cout << "loc: " << i << ", val1: " << ptr1[i] << ", val2: " << ptr2[i]
            << ", abs(err): " << abs(ptr1[i] - ptr2[i]) << endl;
            std::cout << "Compare not pass.\n";
            return;
        }
    }
    std::cout << "Compare pass.\n";

//    for (int i = 0; i < 10; ++i) {
//        std::cout << ptr1[i] << " " << ptr1[size / 2 + i] << " " << ptr1[size - 10 + i] << " ";
//    }
//    std::cout << std::endl;
//    for (int i = 0; i < 10; ++i) {
//        std::cout << ptr2[i] << " " << ptr2[size / 2 + i] << " " << ptr2[size - 10 + i] << " ";
//    }
//    std::cout << std::endl;

    return;
}

int main() {
    void *input_host = nullptr;
    void *output_dev = nullptr;
    void *workspace_dev = nullptr;
    const int32_t input_num = 3;
    void **input_dev_ptr_vec = new void*[input_num];
    for (int i = 0; i < input_num; ++i) input_dev_ptr_vec[i] = nullptr;

    struct input_param {
        int32_t fw_field_num;
        int32_t sample_feature_size;
        int32_t batch_size;
        int32_t embedding_size;
        int32_t field_num;
        bool fw_weight_multil_flag;
        std::string input_file_path;
        std::string result_file_path;
    };

    input_param param;

//    param = {
//    35,
//    2918,
//    254,
//    192,
//    2,
//    true,
//    "../data/sparse_fwffm_input_data2.bin",
//    "../data/sparse_fipnn_result2.bin",
//    };
    param = {
    67,
    2123,
    339,
    96,
    2,
    false,
    "../data/9003_fwffm_input_data_100.bin",
    "../data/9003_fwffm_input_data_100_res_fipnn.bin",
    };

//    const int32_t fw_field_num = 35;
//    const int32_t sample_feature_size = 1413;
//    const int32_t batch_size = 395;

//     const int32_t fw_field_num = 35;
//     const int32_t sample_feature_size = 2918;
//     const int32_t batch_size = 254;

//    const int32_t fw_field_num = 67;
//    const int32_t sample_feature_size = 2123;
//    const int32_t batch_size = 339;

//    const int32_t fw_field_num = 67;
//    const int32_t sample_feature_size = 1039;
//    const int32_t batch_size = 179;

    const int32_t fw_field_num = param.fw_field_num;
    const int32_t sample_feature_size = param.sample_feature_size;
    const int32_t batch_size = param.batch_size;

    const int32_t test_embedding_size = param.embedding_size;
    const int32_t field_num = param.field_num;

    const int32_t fw_weight_size = param.fw_field_num * (param.fw_field_num + 1) / 2;
    int32_t fw_weight_add_size;
    if (param.fw_weight_multil_flag) {
        fw_weight_add_size = batch_size * fw_weight_size;
    } else {
        fw_weight_add_size = fw_weight_size;
    }

    PrepareInput(&input_host, input_dev_ptr_vec, &workspace_dev, &output_dev,
                 param.fw_field_num, param.sample_feature_size, param.batch_size,
                 param.field_num, param.embedding_size, fw_weight_add_size, param.input_file_path);

    uint64_t output_size = param.batch_size * fw_weight_size;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *output_host1 = new float[output_size];
    float *output_host2 = new float[output_size];

//    CUDA_CHECK(cudaMemset(output_dev, 0, sizeof(float) * batch_size * real_embedding_size));
//    nvinfer1::sparse_fipnn::functor::ComputeSparseFipnn<float>(
//            stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num, sample_feature_size,
//            field_num, test_embedding_size, batch_size);
//    cudaMemcpy((void *)output_host1, output_dev, sizeof(float) * output_size, cudaMemcpyDeviceToHost);

//    CUDA_CHECK(cudaMemset(output_dev, 0, sizeof(float) * output_size));
    nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemory<float>(
            stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num, sample_feature_size,
            field_num, test_embedding_size, batch_size);
    cudaMemcpy((void *)output_host1, output_dev, sizeof(float) * output_size, cudaMemcpyDeviceToHost);

    std::ifstream reader;
    reader.open(param.result_file_path, std::ifstream::binary);
    reader.read(reinterpret_cast<char*>(output_host2), sizeof(float) * output_size);
    reader.close();
    CompareResult(output_host1, output_host2, output_size);

    int ite = 1000;
//    int ite = 0;

    {
        //        struct timeval start, end;
        //        double useus;
        //        gettimeofday(&start, NULL);

        cudaEvent_t estart;
        cudaEvent_t estop;
        float usems;
        cudaEventCreate(&estart);
        cudaEventRecord(estart, stream);
        for (int i = 0; i < ite; ++i) {
            nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemory<float>(
                    stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num, sample_feature_size,
                    field_num, test_embedding_size, batch_size);
        }
        cudaEventCreate(&estop);
        cudaEventRecord(estop, stream);
        cudaEventSynchronize(estop);
        cudaEventElapsedTime(&usems, estart, estop);

        //        cudaDeviceSynchronize();
        //        gettimeofday(&end, NULL);
        //        useus = (double(end.tv_sec - start.tv_sec) * 1000 * 1000 + double(end.tv_usec - start.tv_usec)) / ite;

        printf("[INFO] %.1f us ( %d iterations)\n", usems * 1000 / ite, ite);
        //        printf("[INFO] %.1f us ( %d iterations)\n", useus, ite);
    }

//    {
//        cudaEvent_t estart;
//        cudaEvent_t estop;
//        float usems;
//        cudaEventCreate(&estart);
//        cudaEventRecord(estart, stream);
//        for (int i = 0; i < ite; ++i) {
//            nvinfer1::sparse_fipnn::functor::ComputeSparseFipnn<float>(
//            stream, input_dev_ptr_vec, static_cast<float*>(output_dev), workspace_dev, fw_field_num, sample_feature_size,
//            field_num, test_embedding_size, batch_size);
//        }
//        cudaEventCreate(&estop);
//        cudaEventRecord(estop, stream);
//        cudaEventSynchronize(estop);
//        cudaEventElapsedTime(&usems, estart, estop);
//        printf("[INFO] %.1f us ( %d iterations)\n", usems * 1000 / ite, ite);
//    }

    DeleteInput(&input_host, input_dev_ptr_vec, &workspace_dev, &output_dev);
    delete [] output_host1;
    delete [] output_host2;

//    using floatptr = float *;
//    floatptr *input_ptr = (floatptr *)(input_dev_ptr_vec);
//    float **input_ptr = (float **)(input_dev_ptr_vec);
    delete [] input_dev_ptr_vec;

    return 0;
}
