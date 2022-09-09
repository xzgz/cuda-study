#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "cudaTensorCoreGemm.fatbin.c"
extern void __device_stub__Z12compute_gemmPK6__halfS1_PKfPfff(const half *, const half *, const float *, float *, float, float);
extern void __device_stub__Z16simple_wmma_gemmP6__halfS0_PfS1_iiiff(half *, half *, float *, float *, int, int, int, float, float);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z12compute_gemmPK6__halfS1_PKfPfff(const half *__par0, const half *__par1, const float *__par2, float *__par3, float __par4, float __par5){__cudaLaunchPrologue(6);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 36UL);__cudaLaunch(((char *)((void ( *)(const half *, const half *, const float *, float *, float, float))compute_gemm)));}
# 178 "../cudaTensorCoreGemm.cu"
void compute_gemm( const half *__cuda_0,const half *__cuda_1,const float *__cuda_2,float *__cuda_3,float __cuda_4,float __cuda_5)
# 179 "../cudaTensorCoreGemm.cu"
{__device_stub__Z12compute_gemmPK6__halfS1_PKfPfff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 350 "../cudaTensorCoreGemm.cu"
}
# 1 "cudaTensorCoreGemm.cudafe1.stub.c"
void __device_stub__Z16simple_wmma_gemmP6__halfS0_PfS1_iiiff( half *__par0,  half *__par1,  float *__par2,  float *__par3,  int __par4,  int __par5,  int __par6,  float __par7,  float __par8) {  __cudaLaunchPrologue(9); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaLaunch(((char *)((void ( *)(half *, half *, float *, float *, int, int, int, float, float))simple_wmma_gemm))); }
# 360 "../cudaTensorCoreGemm.cu"
void simple_wmma_gemm( half *__cuda_0,half *__cuda_1,float *__cuda_2,float *__cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,float __cuda_7,float __cuda_8)
# 361 "../cudaTensorCoreGemm.cu"
{__device_stub__Z16simple_wmma_gemmP6__halfS0_PfS1_iiiff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8);
# 413 "../cudaTensorCoreGemm.cu"
}
# 1 "cudaTensorCoreGemm.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T13) {  __nv_dummy_param_ref(__T13); __nv_save_fatbinhandle_for_managed_rt(__T13); __cudaRegisterEntry(__T13, ((void ( *)(half *, half *, float *, float *, int, int, int, float, float))simple_wmma_gemm), _Z16simple_wmma_gemmP6__halfS0_PfS1_iiiff, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(const half *, const half *, const float *, float *, float, float))compute_gemm), _Z12compute_gemmPK6__halfS1_PKfPfff, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
