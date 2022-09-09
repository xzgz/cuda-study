#include <stdio.h>
// includes CUDA Runtime
#include <cuda_runtime.h>

#include "../include/common/common.h"

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData;

__global__ void checkGlobalVariable()
{
  // display the original value
  printf("Device: the value of the global variable is %f\n", devData);

  // alter the value
  devData += 2.0f;
}

__global__ void checkGlobalMemoryVariable(float *dptr)
{
  // display the original value
  printf("Device: The value of the global memory variable is %f\n", *dptr);

  // alter the value
  *dptr += 2.0f;
}

int main(void)
{
  // initialize the global variable
  float value = 3.14f;
  CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
  printf("Host:   copied %f to the global variable\n", value);
  printf("value address: %p\n", &value);
  printf("devData address: %p\n", &devData);

  // invoke the kernel
  checkGlobalVariable<<<1, 1>>>();

  // copy the global variable back to the host
  CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
  printf("Host:   the value changed by the checkGlobalVariable kernel to %f\n", value);

  value = 6;
  float *dptr = NULL;
  printf("value address: %p\n", &value);
  printf("devData address: %p\n", &devData);
  printf("dptr address: %p\n", dptr);
  CHECK(cudaGetSymbolAddress((void **)&dptr, devData));
  CHECK(cudaMemcpy((void *)dptr, &value, sizeof(float), cudaMemcpyHostToDevice));
  printf("Host:   copied %f to the global memory variable\n", value);
  printf("value address: %p\n", &value);
  printf("devData address: %p\n", &devData);
  printf("dptr address: %p\n", dptr);

  // invoke the kernel
  checkGlobalMemoryVariable<<<1, 1>>>(dptr);

  // copy the global variable back to the host
  CHECK(cudaMemcpy((void *)&value, dptr, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Host:   the value changed by the checkGlobalMemoryVariable kernel to %f\n", value);

  // invoke the kernel
  checkGlobalVariable<<<1, 1>>>();

  // copy the global variable back to the host
  CHECK(cudaMemcpy((void *)&value, dptr, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Host:   the value changed by the checkGlobalVariable kernel to %f\n", value);
  printf("value address: %p\n", &value);
  printf("devData address: %p\n", &devData);
  printf("dptr address: %p\n", dptr);

  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
