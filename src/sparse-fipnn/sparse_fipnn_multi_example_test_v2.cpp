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

#define ALIGN_UP(x, align_count)    (((x) + ((align_count) - 1)) / (align_count) * (align_count))
#define DATA_ALIGN_BYTE_COUNT       128
#define DATA_ALIGN_INT32_COUNT      32

namespace nvinfer1 {
namespace sparse_fipnn {
namespace functor {
template <typename T>
int32_t ComputeSparseFipnn(
    cudaStream_t stream, const void* const* input, T* output,
    void* workspace, const int32_t fw_field_num,
    const int32_t sample_feature_size, const int32_t field_num,
    const int32_t field_neuron_size, const int32_t batch_size);
}
}

namespace sparse_fipnn_shared {
namespace functor {
template <typename T>
int32_t ComputeSparseFipnnSharedMemory(
    cudaStream_t stream, const void* const* input, T* output,
    void* workspace, const int32_t fw_field_num,
    const int32_t sample_feature_size, const int32_t field_num,
    const int32_t field_neuron_size, const int32_t batch_size);

template <typename T>
int32_t ComputeSparseFipnnSharedMemoryMultiExample(
    cudaStream_t stream, const void* const* input, T* output, void* workspace,
    const int32_t* sample_count_list, const int32_t* batch_size_list,
    const int32_t example_count, const int32_t fw_field_num, const int32_t field_num,
    const int32_t embedding_size);
}
}

namespace sparse_fipnn_shared_multi {
namespace functor {
template <typename T>
int32_t ComputeSparseFipnnSharedMemoryMultiExample(
cudaStream_t stream, const void* const* input, T* output, void* workspace,
const int32_t total_batch_size, const int32_t example_count,
const int32_t max_sample_count, const int32_t max_batch_size,
int32_t fw_field_num, const int32_t field_num, const int32_t embedding_size);
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

struct input_param {
    int fw_field_num;
    int sample_feature_size;
    int batch_size;
    int embedding_size;
    int field_num;
    int feature_weight_size;
    int feature_field_size;
    int feature_index_size;
    uint64_t input_size;
    uint64_t output_size;
    uint64_t workspace_size;
    float* feature_weight;
    int* feature_field;
    int* feature_index;
//    input_param(const input_param& param) {
    input_param() {
        feature_weight = nullptr;
        feature_field = nullptr;
        feature_index = nullptr;
    }
    ~input_param() {
        if (feature_weight) delete [] feature_weight;
        if (feature_field) delete [] feature_field;
        if (feature_index) delete [] feature_index;
    }
};

struct multi_input_param {
    int example_count;
    int fw_field_num;
    int sample_feature_size;
    int batch_size;
    int embedding_size;
    int field_num;
    int feature_weight_size;
    int feature_field_size;
    int feature_index_size;
    uint64_t input_size;
    uint64_t output_size;
    uint64_t workspace_size;
    float* feature_weight;
    int* feature_field;
    int* feature_index;
    int* sample_count_list;
    int* batch_size_list;

    multi_input_param() {
        feature_weight = nullptr;
        feature_field = nullptr;
        feature_index = nullptr;
        sample_count_list = nullptr;
        batch_size_list = nullptr;
    }
    ~multi_input_param() {
        if (feature_weight) delete [] feature_weight;
        if (feature_field) delete [] feature_field;
        if (feature_index) delete [] feature_index;
        if (sample_count_list) delete [] sample_count_list;
        if (batch_size_list) delete [] batch_size_list;
    }
};

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
            std::cout << "The " << i << "th number is invalid index.\n";
                if (val == -1) std::cout << i << " ";
        //        std::cout << val << " ";
    }
//        std::cout << "\nsum = " << sum << std::endl;
        std::cout << std::endl;
    cudaMemcpy(input_dev, input_buffer, sizeof(float) * feature_index_size, cudaMemcpyHostToDevice);
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_index_size / 2 - 5 + i] << "\t";
    //    for (int i = 0; i < 5; ++i) std::cout << ((int32_t *)input_buffer)[feature_index_size - 5 + i] << "\t";
    //    std::cout << std::endl;
    input_ptr_vec[2] = input_dev;
}

void PrepareInput2(vector<void**>& input_dev_list, vector<void*>& workspace_dev_list, vector<void*>& output_dev_list,
                   vector<input_param*> param_list) {
    const int example_count = input_dev_list.size();
    for (int i = 0; i < example_count; ++i) {
        input_param& param = *param_list[i];
        int fw_field_num = param.fw_field_num;
        int field_num = param.field_num;
        int real_embedding_size = param.embedding_size;
        int sample_feature_size = param.sample_feature_size;
        int batch_size = param.batch_size;

        int32_t fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;
        int32_t feature_weight_size = sample_feature_size * field_num * real_embedding_size * sizeof(float);
        int32_t feature_field_size = sample_feature_size * field_num * sizeof(int);
        int32_t feature_index_size = sample_feature_size * sizeof(int);
        int32_t total_input_size = (feature_weight_size + feature_field_size + feature_index_size) * sizeof(float);
        uint64_t total_output_size = batch_size * fw_weight_size * sizeof(float);

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

        void *input_dev_void = nullptr;
        void *output_dev_void = nullptr;
        void *workspace_dev_void = nullptr;
        CUDA_CHECK(cudaMalloc(&input_dev_void, total_input_size));
        CUDA_CHECK(cudaMalloc(&output_dev_void, total_output_size));
        CUDA_CHECK(cudaMalloc(&workspace_dev_void, workspace_size));
        param.feature_weight_size = feature_weight_size;
        param.feature_field_size = feature_field_size;
        param.feature_index_size = feature_index_size;
        param.input_size = total_input_size;
        param.output_size = total_output_size;
        param.workspace_size = workspace_size;
        workspace_dev_list[i] = workspace_dev_void;
        output_dev_list[i] = output_dev_void;

        float* input_dev = (float *)input_dev_void;
        float* feature_weight = param.feature_weight;
        double sum = 0;
        for (int i = 0; i < feature_weight_size / sizeof(float); ++i) {
            sum += feature_weight[i];
            if (isnan(feature_weight[i]) || isinf(feature_weight[i])) std::cout << "the " << i << "th number is nan or inf\n";
        }
        cudaMemcpy((void*)input_dev, (void*)feature_weight, feature_weight_size, cudaMemcpyHostToDevice);
        input_dev_list[i][0] = input_dev;
        input_dev += feature_weight_size / sizeof(float);

        sum = 0;
        int* feature_field = param.feature_field;
        for (int i = 0; i < feature_field_size / sizeof(int); ++i) {
            int32_t val = feature_field[i];
            sum += val;
            if (val < 1 || val > fw_field_num)
                std::cout << "The " << i << "th number is invalid field.\n";
            if (i % 2 == 0 && !(val == 1 || val == 2))
                std::cout << "The " << i << "th number is invalid field.\n";

//            std::cout << feature_field[i] << std::endl;
        }
        cudaMemcpy((void*)input_dev, (void*)feature_field, feature_field_size, cudaMemcpyHostToDevice);
        input_dev_list[i][1] = input_dev;
        input_dev += feature_field_size / sizeof(int);

        int* feature_index = param.feature_index;
        sum = 0;
        for (int i = 0; i < feature_index_size / sizeof(int); ++i) {
            //    for (int i = 2893; i < 2918; ++i) {
            int32_t val = feature_index[i];
            sum += val;
            if (-1 > val || batch_size <= val)
                std::cout << "The " << i << "th number is invalid index.\n";
            if (val == -1) std::cout << i << " ";
        }
        std::cout << std::endl;
        cudaMemcpy((void*)input_dev, (void*)feature_index, feature_index_size, cudaMemcpyHostToDevice);
        input_dev_list[i][2] = input_dev;
    }
}

void DeleteInput(void **input_host, void **input_ptr_vec, void **workspace, void **output) {
    cudaFree(input_ptr_vec[0]);
    cudaFree(*workspace);
    cudaFree(*output);
}

void DeleteInput2(vector<void**>& input_dev_list, vector<void*>& workspace_dev_list, vector<void*>& output_dev_list) {
    const int example_count = input_dev_list.size();
    for (int i = 0; i < example_count; ++i) {
        cudaFree(input_dev_list[i][0]);
        cudaFree(workspace_dev_list[i]);
        cudaFree(output_dev_list[i]);
    }
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
//        std::cout << ptr1[i] << " " << ptr2[i] << std::endl;
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

void InferenceMultiExample() {
    std::string lookup_name = "pair_sparse_embedding";
    std::string data_dir = "/data/plat/hungryhe/data/300_9003_input_dataset/";

    int input_num = 3;
    int field_num = 2;
    int fw_field_num = 67;
    int embedding_size = 96;
    int fw_weight_size = fw_field_num * (fw_field_num + 1) / 2;

    int* batch_sample_num = new int[2];
    int example_count = 4;
    int start_example_id = 10;
//    vector<input_param> param_list(example_count);
//    input_param* param_list = new input_param[example_count];
    vector<input_param*> param_list(example_count);
    for (int i = 0; i < example_count; ++i) {
        param_list[i] = new input_param();
    }

    for (int i = 0; i < example_count; ++i) {
        param_list[i]->field_num = field_num;
        param_list[i]->fw_field_num = fw_field_num;
        param_list[i]->embedding_size = embedding_size;
        std::string feature_weight_path = data_dir + lookup_name + "_feature_weight_" + std::to_string(start_example_id + i) + ".bin";
        std::string feature_field_path = data_dir + lookup_name + "_feature_field_" + std::to_string(start_example_id + i) + ".bin";
        std::string feature_index_path = data_dir + lookup_name + "_feature_index_" + std::to_string(start_example_id + i) + ".bin";
        std::string batch_sample_num_path = data_dir + lookup_name + "_batch_sample_num_" + std::to_string(start_example_id + i) + ".bin";

        std::ifstream rd1, rd2, rd3, rd4;
        rd1.open(batch_sample_num_path, std::ifstream::binary);
        rd1.read(reinterpret_cast<char*>(batch_sample_num), 2 * sizeof(int));
        rd1.close();

        int real_batch_size = batch_sample_num[0];
        int real_sample_num = batch_sample_num[1];
        float* feature_weight = new float[real_sample_num * param_list[i]->field_num * param_list[i]->embedding_size];
        int* feature_field = new int[real_sample_num * param_list[i]->field_num];
        int* feature_index = new int[real_sample_num];

        rd2.open(feature_weight_path, std::ifstream::binary);
        rd2.read(reinterpret_cast<char*>(feature_weight),
                 real_sample_num * param_list[i]->field_num * param_list[i]->embedding_size * sizeof(float));
        rd2.close();

        rd3.open(feature_field_path, std::ifstream::binary);
        rd3.read(reinterpret_cast<char*>(feature_field), real_sample_num * param_list[i]->field_num * sizeof(int));
        rd3.close();

        rd4.open(feature_index_path, std::ifstream::binary);
        rd4.read(reinterpret_cast<char*>(feature_index), real_sample_num * sizeof(int));
        rd4.close();

        param_list[i]->batch_size = real_batch_size;
        param_list[i]->sample_feature_size = real_sample_num;
        param_list[i]->feature_weight = feature_weight;
        param_list[i]->feature_field = feature_field;
        param_list[i]->feature_index = feature_index;

        std::cout << "real_sample_num=" << real_sample_num << std::endl;
        std::cout << "real_batch_size=" << real_batch_size << std::endl;
    }

    vector<void*> output_dev_list(example_count, nullptr);
    vector<void*> workspace_dev_list(example_count, nullptr);
    vector<void**> input_dev_list(example_count, nullptr);
    for (int i = 0; i < example_count; ++i) {
        void **input_dev = new void*[input_num];
        for (int j = 0; j < input_num; ++j) input_dev[j] = nullptr;
        input_dev_list[i] = input_dev;
    }
    PrepareInput2(input_dev_list, workspace_dev_list, output_dev_list, param_list);

    int* sample_count_list = new int[example_count];
    int* batch_size_list = new int[example_count];
    int* sample_count_prefix_sum_vec = new int[example_count];
    int* sample_count_vec = new int[example_count];
    int* batch_size_prefix_sum_vec = new int[example_count];
    int* batch_size_vec = new int[example_count];
    multi_input_param* multi_param = new multi_input_param();
    multi_param->example_count = example_count;
    multi_param->embedding_size = embedding_size;
    multi_param->fw_field_num = fw_field_num;
    multi_param->field_num = field_num;
    multi_param->feature_weight_size = 0;
    multi_param->feature_field_size = 0;
    multi_param->feature_index_size = 0;
    multi_param->sample_feature_size = 0;
    multi_param->batch_size = 0;
    multi_param->input_size = 0;
    multi_param->output_size = 0;
    multi_param->workspace_size = 0;
    int sample_count_sum = 0;
    int max_sample_count = 0;
    int batch_size_sum = 0;
    int max_batch_size = 0;
    for (int i = 0; i < example_count; ++i) {
        sample_count_list[i] = param_list[i]->sample_feature_size;
        sample_count_vec[i] = param_list[i]->sample_feature_size;
        batch_size_list[i] = param_list[i]->batch_size;
        batch_size_vec[i] = param_list[i]->batch_size;
        sample_count_prefix_sum_vec[i] = sample_count_sum;
        batch_size_prefix_sum_vec[i] = batch_size_sum;
        sample_count_sum += param_list[i]->sample_feature_size;
        max_sample_count = std::max(max_sample_count, param_list[i]->sample_feature_size);
        batch_size_sum += param_list[i]->batch_size;
        max_batch_size = std::max(max_batch_size, param_list[i]->batch_size);
        multi_param->feature_weight_size += param_list[i]->feature_weight_size;
        multi_param->feature_field_size += param_list[i]->feature_field_size;
        multi_param->feature_index_size += param_list[i]->feature_index_size;
        multi_param->sample_feature_size += param_list[i]->sample_feature_size;
        multi_param->batch_size += param_list[i]->batch_size;
        multi_param->input_size += param_list[i]->input_size;
        multi_param->output_size += param_list[i]->output_size;
        multi_param->workspace_size = std::max(multi_param->workspace_size, param_list[i]->workspace_size);
    }
    multi_param->sample_count_list = sample_count_list;
    multi_param->batch_size_list = batch_size_list;

    int total_batch_size = multi_param->batch_size;
    uint64_t workspace_size_v2 = (total_batch_size + example_count) * sizeof(int32_t) + example_count * DATA_ALIGN_BYTE_COUNT;
    workspace_size_v2 = ALIGN_UP(workspace_size_v2, DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 += example_count * (2 * (fw_field_num + 1) * sizeof(int32_t) + DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 = ALIGN_UP(workspace_size_v2, DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 += example_count * (fw_weight_size * sizeof(float) + DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 = ALIGN_UP(workspace_size_v2, DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 += (total_batch_size * (field_num + 1) * fw_field_num * embedding_size * sizeof(float) + example_count * DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 = ALIGN_UP(workspace_size_v2, DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 += example_count * ((field_num + 1) * fw_field_num * embedding_size * sizeof(float) + DATA_ALIGN_BYTE_COUNT);
    workspace_size_v2 = ALIGN_UP(workspace_size_v2, DATA_ALIGN_BYTE_COUNT);

    void **all_input_dev = new void*[input_num];
    for (int j = 0; j < input_num; ++j) all_input_dev[j] = nullptr;
    void **all_input_dev_v2 = new void*[input_num + 4];
    for (int j = 0; j < input_num + 4; ++j) all_input_dev_v2[j] = nullptr;
    void *input_dev_void = nullptr;
    void *output_dev_void = nullptr;
    void *workspace_dev_void = nullptr;
    void *workspace_dev_void_v2 = nullptr;
    void *sample_count_prefix_sum_vec_dev = nullptr;
    void *sample_count_vec_dev = nullptr;
    void *batch_size_prefix_sum_vec_dev = nullptr;
    void *batch_size_vec_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&input_dev_void, multi_param->input_size));
    CUDA_CHECK(cudaMalloc(&output_dev_void, multi_param->output_size));
    CUDA_CHECK(cudaMalloc(&workspace_dev_void, multi_param->workspace_size));
    CUDA_CHECK(cudaMalloc(&workspace_dev_void_v2, workspace_size_v2));
    CUDA_CHECK(cudaMalloc(&sample_count_prefix_sum_vec_dev, example_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sample_count_vec_dev, example_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&batch_size_prefix_sum_vec_dev, example_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&batch_size_vec_dev, example_count * sizeof(int)));
    float* feature_weight_dev = (float*)input_dev_void;
    int* feature_field_dev = (int*)(feature_weight_dev + multi_param->feature_weight_size);
    int* feature_index_dev = (int*)(feature_field_dev + multi_param->feature_field_size);
    all_input_dev[0] = feature_weight_dev;
    all_input_dev[1] = feature_field_dev;
    all_input_dev[2] = feature_index_dev;
    all_input_dev_v2[0] = feature_weight_dev;
    all_input_dev_v2[1] = feature_field_dev;
    all_input_dev_v2[2] = feature_index_dev;
    for (int i = 0; i < example_count; ++i) {
        cudaMemcpy((void*)feature_weight_dev, (void*)input_dev_list[i][0], param_list[i]->feature_weight_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy((void*)feature_field_dev, (void*)input_dev_list[i][1], param_list[i]->feature_field_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy((void*)feature_index_dev, (void*)input_dev_list[i][2], param_list[i]->feature_index_size, cudaMemcpyDeviceToDevice);
        feature_weight_dev += param_list[i]->feature_weight_size / sizeof(float);
        feature_field_dev += param_list[i]->feature_field_size / sizeof(int);
        feature_index_dev += param_list[i]->feature_index_size / sizeof(int);
    }
    cudaMemcpy((void*)sample_count_prefix_sum_vec_dev, (void*)sample_count_prefix_sum_vec, example_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)sample_count_vec_dev, (void*)sample_count_vec, example_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)batch_size_prefix_sum_vec_dev, (void*)batch_size_prefix_sum_vec, example_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)batch_size_vec_dev, (void*)batch_size_vec, example_count * sizeof(int), cudaMemcpyHostToDevice);
    all_input_dev_v2[3] = sample_count_prefix_sum_vec_dev;
    all_input_dev_v2[4] = sample_count_vec_dev;
    all_input_dev_v2[5] = batch_size_prefix_sum_vec_dev;
    all_input_dev_v2[6] = batch_size_vec_dev;

    float* all_example_output_host1 = new float[multi_param->output_size / sizeof(float)];
    float* all_example_output_host2 = new float[multi_param->output_size / sizeof(float)];
    memset(all_example_output_host1, 0, multi_param->output_size);
    memset(all_example_output_host2, 0, multi_param->output_size);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float* output_addr_i = all_example_output_host1;
    for (int i = 0; i < example_count; ++i) {
        CUDA_CHECK(cudaMemset(output_dev_list[i], 23, param_list[i]->output_size));
        nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemory<float>(
            stream, input_dev_list[i], static_cast<float*>(output_dev_list[i]), workspace_dev_list[i], fw_field_num,
            param_list[i]->sample_feature_size, field_num, embedding_size, param_list[i]->batch_size);
            cudaMemcpy((void *)output_addr_i, output_dev_list[i], param_list[i]->output_size, cudaMemcpyDeviceToHost);
            output_addr_i += param_list[i]->output_size / sizeof(float);
    }

    CUDA_CHECK(cudaMemset(output_dev_void, 99, multi_param->output_size));
//    nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemoryMultiExample<float>(
//        stream, all_input_dev, static_cast<float*>(output_dev_void), workspace_dev_void, multi_param->sample_count_list,
//        multi_param->batch_size_list, example_count, fw_field_num, field_num, embedding_size);
    nvinfer1::sparse_fipnn_shared_multi::functor::ComputeSparseFipnnSharedMemoryMultiExample(
        stream, all_input_dev_v2, static_cast<float*>(output_dev_void), workspace_dev_void_v2,
        multi_param->batch_size, example_count, max_sample_count, max_batch_size,
        fw_field_num, field_num, embedding_size);

    cudaMemcpy((void*)all_example_output_host2, output_dev_void, multi_param->output_size, cudaMemcpyDeviceToHost);
    CompareResult(all_example_output_host1, all_example_output_host2, multi_param->output_size / sizeof(float));
    std::cout << "multi_param->batch_size=" << multi_param->batch_size << std::endl;
    std::cout << "multi_param->sample_feature_size=" << multi_param->sample_feature_size << std::endl;


    int ite = 1000;
    {
        cudaEvent_t estart;
        cudaEvent_t estop;
        float usems;
        cudaEventCreate(&estart);
        cudaEventRecord(estart, stream);

        for (int n = 0; n < ite; ++n) {

//            for (int i = 0; i < example_count; ++i) {
//                nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemory<float>(
//                    stream, input_dev_list[i], static_cast<float*>(output_dev_list[i]), workspace_dev_list[i], fw_field_num,
//                    param_list[i]->sample_feature_size, field_num, embedding_size, param_list[i]->batch_size);
//            }
            nvinfer1::sparse_fipnn_shared::functor::ComputeSparseFipnnSharedMemoryMultiExample<float>(
                stream, all_input_dev, static_cast<float*>(output_dev_void), workspace_dev_void, multi_param->sample_count_list,
                multi_param->batch_size_list, example_count, fw_field_num, field_num, embedding_size);
//            nvinfer1::sparse_fipnn_shared_multi::functor::ComputeSparseFipnnSharedMemoryMultiExample(
//                stream, all_input_dev_v2, static_cast<float*>(output_dev_void), workspace_dev_void_v2,
//                multi_param->batch_size, example_count, max_sample_count, max_batch_size,
//                fw_field_num, field_num, embedding_size);
        }

        cudaEventCreate(&estop);
        cudaEventRecord(estop, stream);
        cudaEventSynchronize(estop);
        cudaEventElapsedTime(&usems, estart, estop);
        printf("[INFO] %.1f us ( %d iterations)\n", usems * 1000 / ite, ite);
    }


    CUDA_CHECK(cudaStreamDestroy(stream));
    DeleteInput2(input_dev_list, workspace_dev_list, output_dev_list);
    cudaFree(all_input_dev[0]);
    cudaFree(workspace_dev_void);
    cudaFree(workspace_dev_void_v2);
    cudaFree(output_dev_void);
    cudaFree(sample_count_prefix_sum_vec_dev);
    cudaFree(sample_count_vec_dev);
    cudaFree(batch_size_prefix_sum_vec_dev);
    cudaFree(batch_size_vec_dev);
    for (int i = 0; i < example_count; ++i) {
        delete param_list[i];
    }
    delete multi_param;
    delete [] batch_sample_num;
    delete [] all_input_dev;
    delete [] sample_count_prefix_sum_vec;
    delete [] sample_count_vec;
    delete [] batch_size_prefix_sum_vec;
    delete [] batch_size_vec;
    delete [] all_example_output_host1;
    delete [] all_example_output_host2;
}

int main() {
    InferenceMultiExample();

    return 0;
}
