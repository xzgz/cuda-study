#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernels.fatbin.c"
extern void __device_stub__Z18cuda_kernel_warmupPf(float *);
extern void __device_stub__Z19cuda_kernel_sgemm_0PfS_S_mmmff(float *, float *, float *, size_t, size_t, size_t, float, float);
extern void __device_stub__Z19cuda_kernel_sgemm_1PfS_S_mmmff(float *, float *, float *, size_t, size_t, size_t, float, float);
extern void __device_stub__Z19cuda_kernel_sgemm_2PfS_S_mmmff(float *, float *, float *, size_t, size_t, size_t, float, float);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18cuda_kernel_warmupPf(float *__par0){__cudaLaunchPrologue(1);__cudaSetupArgSimple(__par0, 0UL);__cudaLaunch(((char *)((void ( *)(float *))cuda_kernel_warmup)));}
# 7 "../kernels.cu"
void cuda_kernel_warmup( float *__cuda_0)
# 8 "../kernels.cu"
{__device_stub__Z18cuda_kernel_warmupPf( __cuda_0);



}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19cuda_kernel_sgemm_0PfS_S_mmmff( float *__par0,  float *__par1,  float *__par2,  size_t __par3,  size_t __par4,  size_t __par5,  float __par6,  float __par7) {  __cudaLaunchPrologue(8); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 48UL); __cudaSetupArgSimple(__par7, 52UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_0))); }
# 15 "../kernels.cu"
void cuda_kernel_sgemm_0( float *__cuda_0,float *__cuda_1,float *__cuda_2,size_t __cuda_3,size_t __cuda_4,size_t __cuda_5,float __cuda_6,float __cuda_7)
# 19 "../kernels.cu"
{__device_stub__Z19cuda_kernel_sgemm_0PfS_S_mmmff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 34 "../kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19cuda_kernel_sgemm_1PfS_S_mmmff( float *__par0,  float *__par1,  float *__par2,  size_t __par3,  size_t __par4,  size_t __par5,  float __par6,  float __par7) {  __cudaLaunchPrologue(8); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 48UL); __cudaSetupArgSimple(__par7, 52UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_1))); }
# 37 "../kernels.cu"
void cuda_kernel_sgemm_1( float *__cuda_0,float *__cuda_1,float *__cuda_2,size_t __cuda_3,size_t __cuda_4,size_t __cuda_5,float __cuda_6,float __cuda_7)
# 41 "../kernels.cu"
{__device_stub__Z19cuda_kernel_sgemm_1PfS_S_mmmff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 84 "../kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19cuda_kernel_sgemm_2PfS_S_mmmff( float *__par0,  float *__par1,  float *__par2,  size_t __par3,  size_t __par4,  size_t __par5,  float __par6,  float __par7) {  __cudaLaunchPrologue(8); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 48UL); __cudaSetupArgSimple(__par7, 52UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_2))); }
# 87 "../kernels.cu"
void cuda_kernel_sgemm_2( float *__cuda_0,float *__cuda_1,float *__cuda_2,size_t __cuda_3,size_t __cuda_4,size_t __cuda_5,float __cuda_6,float __cuda_7)
# 91 "../kernels.cu"
{__device_stub__Z19cuda_kernel_sgemm_2PfS_S_mmmff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 134 "../kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_2), _Z19cuda_kernel_sgemm_2PfS_S_mmmff, (-1)); __cudaRegisterEntry(__T0, ((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_1), _Z19cuda_kernel_sgemm_1PfS_S_mmmff, (-1)); __cudaRegisterEntry(__T0, ((void ( *)(float *, float *, float *, size_t, size_t, size_t, float, float))cuda_kernel_sgemm_0), _Z19cuda_kernel_sgemm_0PfS_S_mmmff, (-1)); __cudaRegisterEntry(__T0, ((void ( *)(float *))cuda_kernel_warmup), _Z18cuda_kernel_warmupPf, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
