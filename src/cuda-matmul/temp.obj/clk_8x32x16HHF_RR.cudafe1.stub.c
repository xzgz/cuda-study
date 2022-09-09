#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "clk_8x32x16HHF_RR.fatbin.c"
extern void __device_stub__Z12wmma_exampleP6__halfS0_PfS1_(atype *, btype *, ctype *, dtype *);
static void __device_stub__Z7convertI6__halffEvPT_PT0_i(atype *, host_type *, int);
static void __device_stub__Z7convertIffEvPT_PT0_i(ctype *, host_type *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z12wmma_exampleP6__halfS0_PfS1_(atype *__par0, btype *__par1, ctype *__par2, dtype *__par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(atype *, btype *, ctype *, dtype *))wmma_example)));}
# 192 "../clk_8x32x16HHF_RR.cu"
void wmma_example( atype *__cuda_0,btype *__cuda_1,ctype *__cuda_2,dtype *__cuda_3)
# 193 "../clk_8x32x16HHF_RR.cu"
{__device_stub__Z12wmma_exampleP6__halfS0_PfS1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 208 "../clk_8x32x16HHF_RR.cu"
}
# 1 "clk_8x32x16HHF_RR.cudafe1.stub.c"
static void __device_stub__Z7convertI6__halffEvPT_PT0_i( atype *__par0,  host_type *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(atype *, host_type *, int))convert<    ::__half, float> ))); }
template<> __specialization_static void __wrapper__device_stub_convert< ::atype,float>( ::atype *&__cuda_0,::host_type *&__cuda_1,int &__cuda_2){__device_stub__Z7convertI6__halffEvPT_PT0_i( (::atype *&)__cuda_0,(::host_type *&)__cuda_1,(int &)__cuda_2);}
static void __device_stub__Z7convertIffEvPT_PT0_i( ctype *__par0,  host_type *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(ctype *, host_type *, int))convert<float, float> ))); }
template<> __specialization_static void __wrapper__device_stub_convert<float,float>( ::ctype *&__cuda_0,::host_type *&__cuda_1,int &__cuda_2){__device_stub__Z7convertIffEvPT_PT0_i( (::ctype *&)__cuda_0,(::host_type *&)__cuda_1,(int &)__cuda_2);}
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(ctype *, host_type *, int))convert<float, float> ), _Z7convertIffEvPT_PT0_i, (-1)); __cudaRegisterEntry(__T3, ((void ( *)(atype *, host_type *, int))convert<    ::__half, float> ), _Z7convertI6__halffEvPT_PT0_i, (-1)); __cudaRegisterEntry(__T3, ((void ( *)(atype *, btype *, ctype *, dtype *))wmma_example), _Z12wmma_exampleP6__halfS0_PfS1_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
