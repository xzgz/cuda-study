#include "cuda_runtime.h"
#include <stdio.h>

using namespace std;

__global__ void transformKernel(float* output, int width, int height);

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
const int cycle_count = 1000;

__global__ void transformKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x<0 || x>width || y<0 || y>height)
        return;

    // 纹理读取
    // 一定要偏移0.5像素，原因是CUDA在采样时偏移了0.5像素
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering
    output[y * width + x % width] = tex2D(texRef, x+0.5f, y+0.5f);
    // for (int i = 0; i < cycle_count; ++i) {
    //     output[y * width + (x + i) % width] = tex2D(texRef, x+0.5f, y+0.5f);
    // }
}


int main()
{
    //实验数据
	int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);

	float h_data[width*height];

        for(int y=0;y<height;y++)
	{
	  for(int x=0;x<width;x++)
	  {
	    h_data[y*width+x] = x;
	  }
        }

    // for (int y = 0; y<height; y++)
	// {
	// 	for (int x = 0; x<width; x++)
	// 	{
	// 		printf("%f ", h_data[y*width + x]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");

    // 设备内存声明，此处以cudaArray为例
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);

    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 数据拷贝
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size,cudaMemcpyHostToDevice);

    // 设定纹理参考的属性
    texRef.addressMode[0] = cudaAddressModeBorder;
    texRef.addressMode[1] = cudaAddressModeBorder;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = 0;

    // 纹理绑定
    cudaBindTextureToArray(texRef, cuArray);

    // 保存结果
    float* output;
    cudaMalloc(&output, size);

    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);
    dim3 dimBlock(32, 32);
    dim3 dimGrid( max( (width+dimBlock.x-1)/dimBlock.x,1 ),
                  max( (height+dimBlock.y-1)/dimBlock.y,1) );
    for (int i = 0; i < cycle_count; ++i) {
        transformKernel<<<dimGrid, dimBlock>>>(output, width, height);
    }
    // transformKernel<<<dimGrid, dimBlock>>>(output, width, height);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Processing time: %f (ms), %f (GB/s)\n", msecTotal, float(width * height * 4 * 1e+3 * 2 * cycle_count) / float(1024 * 1024 * 1024 * msecTotal));

    cudaMemcpy(h_data, output, size, cudaMemcpyDeviceToHost);

	// for (int y = 0; y<height; y++)
	// {
	// 	for (int x = 0; x<width; x++)
	// 	{
	// 		printf("%f ", h_data[y*width + x]);
	// 	}
	// 	printf("\n");
	// }

    // 释放内存
    cudaFreeArray(cuArray);
    cudaFree(output);

    return 0;
}
