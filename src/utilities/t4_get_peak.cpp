#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>


#define CHECK_CUDA(x, str) \
  if((x) != cudaSuccess) \
  { \
    fprintf(stderr, str); \
    exit(EXIT_FAILURE); \
  }

int main(void)
{
  int gpu_index = 0;
  cudaDeviceProp prop;

  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_index), "cudaGetDeviceProperties error");
  printf("GPU Name = %s\n", prop.name);
  printf("Compute Capability = %d.%d\n", prop.major, prop.minor); // 鑾峰緱 SM 鐗堟湰
  printf("GPU SMs = %d\n", prop.multiProcessorCount); // 鑾峰緱 SM 鏁扮洰
  printf("GPU SM clock rate = %.3f GHz\n", prop.clockRate/1e6); // prop.clockRate 鍗曚綅涓� kHz锛岄櫎浠� 1e6 涔嬪悗鍗曚綅涓� GHz
  printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate/1e6); // 鍚屼笂
  if((prop.major == 7) && (prop.minor == 5)) // SM 8.0锛屽嵆 A100
  {
    // 鏍规嵁鍏紡璁＄畻宄板€煎悶鍚愶紝鍏朵腑 64銆�32銆�256銆�256 鏄粠琛ㄤ腑鏌ュ埌
    printf("-----------CUDA Core Performance------------\n");
    printf("FP32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 64 * 2);
    printf("FP64 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 2 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 128 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 256 * 2);

    printf("-----------Tensor Core Dense Performance------------\n");
    printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 512 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 1024 * 2); 
    printf("INT4 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 2048 * 2);
    printf("INT1 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate/1e6) * 8192 * 2);
  }
  return 0;
}
