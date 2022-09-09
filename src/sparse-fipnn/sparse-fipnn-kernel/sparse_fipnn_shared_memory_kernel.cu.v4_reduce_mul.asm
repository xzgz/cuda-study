
Fatbin elf code:
================
arch = sm_75
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Fatbin ptx code:
================
arch = sm_75
code version = [7,0]
producer = <unknown>
host = linux
compile_size = 64bit
compressed








.version 7.0
.target sm_75
.address_size 64


.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust6system6detail10sequential3seqE[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust6system3cpp3parE[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust8cuda_cub3parE[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_1E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_2E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_3E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_4E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_5E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_6E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_7E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_8E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders2_9E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust12placeholders3_10E[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust3seqE[1];
.global .align 1 .b8 _ZN76_INTERNAL_54_tmpxft_0000271f_00000000_7_sparse_fipnn_kernel_cpp1_ii_b298f1786thrust6deviceE[1];
.extern .shared .align 4 .b8 _ZN8nvinfer112sparse_fipnn9smem_poolE[];

.visible .entry _ZN3cub11EmptyKernelIvEEvv(

)
{



ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3_(
.param .u64 _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_2,
.param .u64 _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_4
)
{
.reg .pred %p<21>;
.reg .b32 %r<62>;
.reg .b64 %rd<38>;


ld.param.u64 %rd10, [_ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_0];
ld.param.u32 %r35, [_ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_1];
ld.param.u32 %r36, [_ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_2];
ld.param.u64 %rd11, [_ZN8nvinfer112sparse_fipnn20ComputeBatchBoundaryEPKiiiPiS3__param_3];
cvta.to.global.u64 %rd1, %rd11;
cvta.to.global.u64 %rd2, %rd10;
mov.u32 %r37, %ntid.x;
mov.u32 %r38, %ctaid.x;
mov.u32 %r39, %tid.x;
mad.lo.s32 %r1, %r37, %r38, %r39;
setp.ge.s32	%p1, %r1, %r35;
@%p1 bra BB1_32;

setp.gt.s32	%p2, %r1, 0;
@%p2 bra BB1_22;
bra.uni BB1_2;

BB1_22:
mul.wide.s32 %rd27, %r1, 4;
add.s64 %rd28, %rd2, %rd27;
ld.global.u32 %r23, [%rd28];
ld.global.u32 %r24, [%rd28+-4];
setp.le.s32	%p15, %r23, %r24;
@%p15 bra BB1_32;

sub.s32 %r25, %r23, %r24;
and.b32 %r26, %r25, 3;
setp.eq.s32	%p16, %r26, 0;
@%p16 bra BB1_29;

setp.eq.s32	%p17, %r26, 1;
@%p17 bra BB1_28;

setp.eq.s32	%p18, %r26, 2;
@%p18 bra BB1_27;

mul.wide.s32 %rd29, %r23, 4;
add.s64 %rd30, %rd1, %rd29;
st.global.u32 [%rd30], %r1;
add.s32 %r23, %r23, -1;

BB1_27:
mul.wide.s32 %rd31, %r23, 4;
add.s64 %rd32, %rd1, %rd31;
st.global.u32 [%rd32], %r1;
add.s32 %r23, %r23, -1;

BB1_28:
mul.wide.s32 %rd33, %r23, 4;
add.s64 %rd34, %rd1, %rd33;
st.global.u32 [%rd34], %r1;
add.s32 %r23, %r23, -1;

BB1_29:
setp.lt.u32	%p19, %r25, 4;
@%p19 bra BB1_32;

mul.wide.s32 %rd35, %r23, 4;
add.s64 %rd37, %rd1, %rd35;

BB1_31:
st.global.u32 [%rd37], %r1;
st.global.u32 [%rd37+-4], %r1;
st.global.u32 [%rd37+-8], %r1;
st.global.u32 [%rd37+-12], %r1;
add.s64 %rd37, %rd37, -16;
add.s32 %r23, %r23, -4;
setp.gt.s32	%p20, %r23, %r24;
@%p20 bra BB1_31;
bra.uni BB1_32;

BB1_2:
ld.global.u32 %r2, [%rd2];
setp.lt.s32	%p3, %r2, 0;
@%p3 bra BB1_12;

add.s32 %r3, %r2, 1;
and.b32 %r43, %r3, 3;
mov.u32 %r52, 0;
setp.eq.s32	%p4, %r43, 0;
@%p4 bra BB1_9;

setp.eq.s32	%p5, %r43, 1;
mov.u32 %r51, %r52;
@%p5 bra BB1_8;

setp.eq.s32	%p6, %r43, 2;
mov.u32 %r50, %r52;
@%p6 bra BB1_7;

mov.u32 %r45, 0;
st.global.u32 [%rd1], %r45;
mov.u32 %r50, 1;

BB1_7:
mul.wide.u32 %rd12, %r50, 4;
add.s64 %rd13, %rd1, %rd12;
st.global.u32 [%rd13], %r52;
add.s32 %r51, %r50, 1;

BB1_8:
mul.wide.s32 %rd14, %r51, 4;
add.s64 %rd15, %rd1, %rd14;
st.global.u32 [%rd15], %r52;
add.s32 %r52, %r51, 1;

BB1_9:
setp.lt.u32	%p7, %r3, 4;
@%p7 bra BB1_12;

add.s32 %r53, %r52, -1;
mul.wide.s32 %rd16, %r52, 4;
add.s64 %rd36, %rd1, %rd16;

BB1_11:
mov.u64 %rd17, 0;
st.global.u32 [%rd36+4], %rd17;
st.global.u32 [%rd36], %rd17;
st.global.u32 [%rd36+12], %rd17;
st.global.u32 [%rd36+8], %rd17;
add.s64 %rd36, %rd36, 16;
add.s32 %r53, %r53, 4;
setp.lt.s32	%p8, %r53, %r2;
@%p8 bra BB1_11;

BB1_12:
add.s32 %r48, %r35, -1;
mul.wide.s32 %rd18, %r48, 4;
add.s64 %rd19, %rd2, %rd18;
add.s32 %r54, %r36, -1;
ld.global.u32 %r13, [%rd19];
mul.wide.s32 %rd20, %r36, 4;
add.s64 %rd6, %rd1, %rd20;
setp.le.s32	%p9, %r54, %r13;
@%p9 bra BB1_21;

sub.s32 %r14, %r54, %r13;
and.b32 %r49, %r14, 3;
setp.eq.s32	%p10, %r49, 0;
@%p10 bra BB1_19;

setp.eq.s32	%p11, %r49, 1;
@%p11 bra BB1_18;

setp.eq.s32	%p12, %r49, 2;
@%p12 bra BB1_17;

st.global.u32 [%rd6+-4], %r35;
add.s32 %r54, %r36, -2;

BB1_17:
mul.wide.s32 %rd21, %r54, 4;
add.s64 %rd22, %rd1, %rd21;
st.global.u32 [%rd22], %r35;
add.s32 %r54, %r54, -1;

BB1_18:
mul.wide.s32 %rd23, %r54, 4;
add.s64 %rd24, %rd1, %rd23;
st.global.u32 [%rd24], %r35;
add.s32 %r54, %r54, -1;

BB1_19:
setp.lt.u32	%p13, %r14, 4;
@%p13 bra BB1_21;

BB1_20:
mul.wide.s32 %rd25, %r54, 4;
add.s64 %rd26, %rd1, %rd25;
st.global.u32 [%rd26], %r35;
st.global.u32 [%rd26+-4], %r35;
st.global.u32 [%rd26+-8], %r35;
st.global.u32 [%rd26+-12], %r35;
add.s32 %r54, %r54, -4;
setp.gt.s32	%p14, %r54, %r13;
@%p14 bra BB1_20;

BB1_21:
st.global.u32 [%rd6], %r35;

BB1_32:
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2_(
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_2,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_6,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_7,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_8
)
{
.reg .pred %p<19>;
.reg .f32 %f<6>;
.reg .b32 %r<56>;
.reg .b64 %rd<42>;


ld.param.u32 %r22, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_0];
ld.param.u32 %r23, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_1];
ld.param.u32 %r24, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_2];
ld.param.u64 %rd18, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_3];
ld.param.u64 %rd13, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_4];
ld.param.u64 %rd14, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_5];
ld.param.u64 %rd15, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_6];
ld.param.u64 %rd16, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_7];
ld.param.u64 %rd17, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartIfLi32EEEviiiPiPKT_PKiPS3_S8_S2__param_8];
mov.u32 %r25, %ctaid.x;
shl.b32 %r26, %r25, 5;
mov.u32 %r27, %tid.y;
add.s32 %r50, %r26, %r27;
cvta.to.global.u64 %rd19, %rd18;
ld.global.u32 %r2, [%rd19];
setp.ge.s32	%p1, %r50, %r2;
@%p1 bra BB2_19;

add.s32 %r28, %r22, 31;
shr.s32 %r29, %r28, 31;
shr.u32 %r30, %r29, 27;
add.s32 %r31, %r28, %r30;
and.b32 %r3, %r31, -32;
cvta.to.global.u64 %rd20, %rd14;
cvta.to.global.u64 %rd23, %rd17;
cvta.to.global.u64 %rd28, %rd15;
cvta.to.global.u64 %rd34, %rd16;

BB2_2:
shl.b32 %r32, %r50, 1;
mul.wide.s32 %rd21, %r32, 4;
add.s64 %rd22, %rd20, %rd21;
ld.global.u32 %r5, [%rd22];
add.s32 %r6, %r5, -1;
ld.global.u32 %r7, [%rd22+4];
add.s32 %r8, %r7, -1;
setp.lt.s32	%p2, %r8, 0;
setp.ge.s32	%p3, %r8, %r24;
or.pred %p4, %p2, %p3;
setp.lt.s32	%p5, %r6, 0;
or.pred %p6, %p4, %p5;
setp.ge.s32	%p7, %r6, %r23;
or.pred %p8, %p6, %p7;
@%p8 bra BB2_18;

mov.u32 %r54, %tid.x;
setp.ne.s32	%p9, %r54, 0;
@%p9 bra BB2_5;

mul.wide.s32 %rd24, %r8, 4;
add.s64 %rd25, %rd23, %rd24;
add.s32 %r49, %r5, -1;
st.global.u32 [%rd25], %r49;

BB2_5:
setp.lt.s32	%p10, %r23, 1;
@%p10 bra BB2_13;

mov.u32 %r51, 0;

BB2_7:
setp.lt.s32	%p11, %r22, 1;
@%p11 bra BB2_12;

mad.lo.s32 %r36, %r23, %r50, %r51;
mov.u32 %r52, %tid.x;
mad.lo.s32 %r37, %r22, %r36, %r52;
cvta.to.global.u64 %rd26, %rd13;
mul.wide.s32 %rd27, %r37, 4;
add.s64 %rd39, %rd26, %rd27;
mad.lo.s32 %r38, %r24, %r51, %r7;
add.s32 %r39, %r38, -1;
mul.lo.s32 %r40, %r22, %r39;
mul.wide.s32 %rd29, %r52, 4;
add.s64 %rd30, %rd28, %rd29;
mul.wide.s32 %rd31, %r40, 4;
add.s64 %rd38, %rd30, %rd31;
mov.u32 %r53, 0;

BB2_9:
setp.ge.s32	%p12, %r52, %r22;
@%p12 bra BB2_11;

ld.global.f32 %f1, [%rd39];
atom.global.add.f32 %f2, [%rd38], %f1;

BB2_11:
add.s32 %r53, %r53, 32;
add.s64 %rd39, %rd39, 128;
add.s32 %r52, %r52, 32;
add.s64 %rd38, %rd38, 128;
setp.lt.s32	%p13, %r53, %r3;
@%p13 bra BB2_9;

BB2_12:
add.s32 %r51, %r51, 1;
setp.lt.s32	%p14, %r51, %r23;
@%p14 bra BB2_7;

BB2_13:
setp.lt.s32	%p15, %r22, 1;
@%p15 bra BB2_18;

mad.lo.s32 %r42, %r23, %r50, %r5;
add.s32 %r43, %r42, -1;
mad.lo.s32 %r44, %r22, %r43, %r54;
cvta.to.global.u64 %rd32, %rd13;
mul.wide.s32 %rd33, %r44, 4;
add.s64 %rd41, %rd32, %rd33;
mul.lo.s32 %r46, %r22, %r8;
mul.wide.s32 %rd35, %r54, 4;
add.s64 %rd36, %rd34, %rd35;
mul.wide.s32 %rd37, %r46, 4;
add.s64 %rd40, %rd36, %rd37;
mov.u32 %r55, 0;

BB2_15:
setp.ge.s32	%p16, %r54, %r22;
@%p16 bra BB2_17;

ld.global.f32 %f3, [%rd41];
mul.f32 %f4, %f3, %f3;
atom.global.add.f32 %f5, [%rd40], %f4;

BB2_17:
add.s32 %r55, %r55, 32;
add.s64 %rd41, %rd41, 128;
add.s32 %r54, %r54, 32;
add.s64 %rd40, %rd40, 128;
setp.lt.s32	%p17, %r55, %r3;
@%p17 bra BB2_15;

BB2_18:
mov.u32 %r47, %nctaid.x;
shl.b32 %r48, %r47, 5;
add.s32 %r50, %r50, %r48;
setp.lt.s32	%p18, %r50, %r2;
@%p18 bra BB2_2;

BB2_19:
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3_(
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_2,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_6
)
{
.reg .f32 %f<4>;
.reg .b32 %r<20>;
.reg .b64 %rd<18>;


ld.param.u32 %r1, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_1];
ld.param.u32 %r2, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_2];
ld.param.u32 %r3, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_3];
ld.param.u64 %rd1, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_4];
ld.param.u64 %rd2, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_5];
ld.param.u64 %rd3, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_6];
cvta.to.global.u64 %rd4, %rd1;
cvta.to.global.u64 %rd5, %rd3;
cvta.to.global.u64 %rd6, %rd2;
mov.u32 %r4, %tid.x;
mov.u32 %r5, %ctaid.x;
rem.u32 %r6, %r5, %r3;
mad.lo.s32 %r7, %r6, %r1, %r4;
div.u32 %r8, %r5, %r3;
mul.wide.s32 %rd7, %r7, 4;
add.s64 %rd8, %rd6, %rd7;
ld.global.f32 %f1, [%rd8];
add.s32 %r9, %r2, 1;
mul.lo.s32 %r10, %r9, %r1;
mul.lo.s32 %r11, %r10, %r3;
mul.lo.s32 %r12, %r11, %r8;
mul.lo.s32 %r13, %r2, %r1;
mad.lo.s32 %r14, %r13, %r3, %r12;
add.s32 %r15, %r14, %r7;
mul.wide.s32 %rd9, %r15, 4;
add.s64 %rd10, %rd5, %rd9;
st.global.f32 [%rd10], %f1;
add.s64 %rd11, %rd4, %rd7;
ld.global.f32 %f2, [%rd11];
mul.lo.s32 %r16, %r3, %r1;
add.s32 %r17, %r7, %r16;
mul.wide.s32 %rd12, %r17, 4;
add.s64 %rd13, %rd4, %rd12;
ld.global.f32 %f3, [%rd13];
add.s32 %r18, %r12, %r7;
mul.wide.s32 %rd14, %r18, 4;
add.s64 %rd15, %rd5, %rd14;
st.global.f32 [%rd15], %f2;
add.s32 %r19, %r18, %r16;
mul.wide.s32 %rd16, %r19, 4;
add.s64 %rd17, %rd5, %rd16;
st.global.f32 [%rd17], %f3;
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4_(
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_2,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_6,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_7,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_8,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_9,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_10,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_11
)
{
.local .align 16 .b8 __local_depot4[48];
.reg .b64 %SP;
.reg .b64 %SPL;
.reg .pred %p<93>;
.reg .f32 %f<148>;
.reg .b32 %r<438>;
.reg .b64 %rd<134>;


mov.u64 %SPL, __local_depot4;
ld.param.u32 %r116, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_0];
ld.param.u32 %r117, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_1];
ld.param.u32 %r118, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_2];
ld.param.u64 %rd47, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_3];
ld.param.u64 %rd48, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_7];
ld.param.u64 %rd49, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_8];
ld.param.u64 %rd46, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_9];
ld.param.u64 %rd50, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_10];
ld.param.u64 %rd51, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuIfLi32EEEviiiPiS2_PT_S4_S2_PKS3_PKiS4_S4__param_11];
cvta.to.global.u64 %rd1, %rd48;
cvta.to.global.u64 %rd2, %rd49;
cvta.to.global.u64 %rd3, %rd51;
cvta.to.global.u64 %rd4, %rd50;
add.u64 %rd5, %SPL, 0;
add.s32 %r119, %r118, 1;
mul.lo.s32 %r120, %r119, %r118;
shr.u32 %r121, %r120, 31;
add.s32 %r122, %r120, %r121;
shr.s32 %r123, %r122, 1;
mov.u32 %r1, %ctaid.x;
mul.lo.s32 %r2, %r123, %r1;
add.s32 %r124, %r117, 1;
mul.lo.s32 %r125, %r124, %r116;
mul.lo.s32 %r3, %r125, %r118;
mul.lo.s32 %r4, %r3, %r1;
mul.lo.s32 %r126, %r117, %r116;
mad.lo.s32 %r5, %r126, %r118, %r4;
cvta.to.global.u64 %rd53, %rd47;
mul.wide.s32 %rd54, %r1, 4;
add.s64 %rd55, %rd53, %rd54;
ld.global.u32 %r6, [%rd55];
ld.global.u32 %r7, [%rd55+4];
mov.u32 %r127, %ntid.x;
mov.u32 %r424, %tid.y;
mov.u32 %r9, %tid.x;
mad.lo.s32 %r10, %r127, %r424, %r9;
setp.lt.s32	%p1, %r118, 1;
@%p1 bra BB4_23;

add.s32 %r132, %r118, -1;
shr.u32 %r133, %r132, 10;
add.s32 %r11, %r133, 1;
and.b32 %r131, %r11, 3;
mov.u32 %r396, 0;
setp.eq.s32	%p2, %r131, 0;
@%p2 bra BB4_12;

setp.eq.s32	%p3, %r131, 1;
@%p3 bra BB4_9;

setp.eq.s32	%p4, %r131, 2;
@%p4 bra BB4_6;

mov.u32 %r396, 1024;
setp.ge.s32	%p5, %r10, %r118;
@%p5 bra BB4_6;

mul.wide.s32 %rd56, %r10, 4;
add.s64 %rd57, %rd1, %rd56;
ld.global.u32 %r136, [%rd57];
shl.b32 %r137, %r10, 2;
mov.u32 %r138, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r139, %r138, %r137;
st.shared.u32 [%r139], %r136;

BB4_6:
add.s32 %r13, %r10, %r396;
setp.ge.s32	%p6, %r13, %r118;
@%p6 bra BB4_8;

mul.wide.s32 %rd58, %r13, 4;
add.s64 %rd59, %rd1, %rd58;
ld.global.u32 %r140, [%rd59];
shl.b32 %r141, %r13, 2;
mov.u32 %r142, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r143, %r142, %r141;
st.shared.u32 [%r143], %r140;

BB4_8:
add.s32 %r396, %r396, 1024;

BB4_9:
add.s32 %r16, %r10, %r396;
setp.ge.s32	%p7, %r16, %r118;
@%p7 bra BB4_11;

mul.wide.s32 %rd60, %r16, 4;
add.s64 %rd61, %rd1, %rd60;
ld.global.u32 %r144, [%rd61];
shl.b32 %r145, %r16, 2;
mov.u32 %r146, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r147, %r146, %r145;
st.shared.u32 [%r147], %r144;

BB4_11:
add.s32 %r396, %r396, 1024;

BB4_12:
setp.lt.u32	%p8, %r11, 4;
@%p8 bra BB4_23;

add.s32 %r399, %r10, %r396;
shl.b32 %r148, %r399, 2;
mov.u32 %r149, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r400, %r149, %r148;
mul.wide.s32 %rd62, %r399, 4;
add.s64 %rd124, %rd1, %rd62;

BB4_14:
setp.ge.s32	%p9, %r399, %r118;
@%p9 bra BB4_16;

ld.global.u32 %r150, [%rd124];
st.shared.u32 [%r400], %r150;

BB4_16:
add.s32 %r151, %r399, 1024;
setp.ge.s32	%p10, %r151, %r118;
@%p10 bra BB4_18;

ld.global.u32 %r152, [%rd124+4096];
st.shared.u32 [%r400+4096], %r152;

BB4_18:
add.s32 %r153, %r399, 2048;
setp.ge.s32	%p11, %r153, %r118;
@%p11 bra BB4_20;

ld.global.u32 %r154, [%rd124+8192];
st.shared.u32 [%r400+8192], %r154;

BB4_20:
add.s32 %r155, %r399, 3072;
setp.ge.s32	%p12, %r155, %r118;
@%p12 bra BB4_22;

ld.global.u32 %r156, [%rd124+12288];
st.shared.u32 [%r400+12288], %r156;

BB4_22:
add.s32 %r396, %r396, 4096;
add.s64 %rd124, %rd124, 16384;
add.s32 %r399, %r399, 4096;
setp.lt.s32	%p13, %r396, %r118;
add.s32 %r400, %r400, 16384;
@%p13 bra BB4_14;

BB4_23:
bar.sync 0;
add.s32 %r402, %r424, %r6;
setp.ge.s32	%p14, %r402, %r7;
@%p14 bra BB4_42;

cvta.to.global.u64 %rd9, %rd46;
add.s32 %r157, %r116, 31;
shr.s32 %r158, %r157, 31;
shr.u32 %r159, %r158, 27;
add.s32 %r160, %r157, %r159;
and.b32 %r28, %r160, -32;
mad.lo.s32 %r161, %r3, %r1, %r9;
mul.wide.s32 %rd63, %r161, 4;
add.s64 %rd10, %rd3, %rd63;
add.s32 %r162, %r5, %r9;
mul.wide.s32 %rd64, %r162, 4;
add.s64 %rd11, %rd3, %rd64;

BB4_25:
shl.b32 %r163, %r402, 1;
mul.wide.s32 %rd65, %r163, 4;
add.s64 %rd66, %rd9, %rd65;
add.s32 %r164, %r163, 1;
mul.wide.s32 %rd67, %r164, 4;
add.s64 %rd68, %rd9, %rd67;
ld.global.u32 %r30, [%rd68];
add.s32 %r165, %r30, -1;
setp.lt.s32	%p15, %r165, 0;
setp.ge.s32	%p16, %r165, %r118;
or.pred %p17, %p15, %p16;
ld.global.u32 %r31, [%rd66];
add.s32 %r166, %r31, -1;
setp.lt.s32	%p18, %r166, 0;
or.pred %p19, %p17, %p18;
setp.ge.s32	%p20, %r166, %r117;
or.pred %p21, %p19, %p20;
@%p21 bra BB4_41;

setp.ne.s32	%p22, %r9, 0;
@%p22 bra BB4_28;

shl.b32 %r167, %r30, 2;
mov.u32 %r168, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r169, %r167, %r168;
add.s32 %r395, %r31, -1;
st.shared.u32 [%r169+-4], %r395;

BB4_28:
setp.lt.s32	%p23, %r117, 1;
@%p23 bra BB4_36;

mul.lo.s32 %r32, %r117, %r402;
mov.u32 %r403, 0;

BB4_30:
setp.lt.s32	%p24, %r116, 1;
@%p24 bra BB4_35;

add.s32 %r173, %r32, %r403;
mad.lo.s32 %r174, %r116, %r173, %r9;
mul.wide.s32 %rd69, %r174, 4;
add.s64 %rd126, %rd2, %rd69;
mad.lo.s32 %r175, %r118, %r403, %r165;
mul.lo.s32 %r176, %r116, %r175;
mul.wide.s32 %rd70, %r176, 4;
add.s64 %rd125, %rd10, %rd70;
mov.u32 %r405, 0;
mov.u32 %r404, %r9;

BB4_32:
setp.ge.s32	%p25, %r404, %r116;
@%p25 bra BB4_34;

ld.global.f32 %f51, [%rd126];
atom.global.add.f32 %f52, [%rd125], %f51;

BB4_34:
add.s32 %r405, %r405, 32;
add.s64 %rd126, %rd126, 128;
add.s32 %r404, %r404, 32;
add.s64 %rd125, %rd125, 128;
setp.lt.s32	%p26, %r405, %r28;
@%p26 bra BB4_32;

BB4_35:
add.s32 %r403, %r403, 1;
setp.lt.s32	%p27, %r403, %r117;
@%p27 bra BB4_30;

BB4_36:
setp.lt.s32	%p28, %r116, 1;
@%p28 bra BB4_41;

mad.lo.s32 %r178, %r117, %r402, %r31;
add.s32 %r179, %r178, -1;
mad.lo.s32 %r180, %r116, %r179, %r9;
mul.wide.s32 %rd71, %r180, 4;
add.s64 %rd128, %rd2, %rd71;
mul.lo.s32 %r182, %r116, %r165;
mul.wide.s32 %rd72, %r182, 4;
add.s64 %rd127, %rd11, %rd72;
mov.u32 %r407, 0;
mov.u32 %r406, %r9;

BB4_38:
setp.ge.s32	%p29, %r406, %r116;
@%p29 bra BB4_40;

ld.global.f32 %f53, [%rd128];
mul.f32 %f54, %f53, %f53;
atom.global.add.f32 %f55, [%rd127], %f54;

BB4_40:
add.s32 %r407, %r407, 32;
add.s64 %rd128, %rd128, 128;
add.s32 %r406, %r406, 32;
add.s64 %rd127, %rd127, 128;
setp.lt.s32	%p30, %r407, %r28;
@%p30 bra BB4_38;

BB4_41:
add.s32 %r402, %r402, 32;
setp.lt.s32	%p31, %r402, %r7;
@%p31 bra BB4_25;

BB4_42:
bar.sync 0;
or.b32 %r183, %r9, %r424;
add.s32 %r184, %r118, %r118;
shl.b32 %r185, %r184, 2;
mov.u32 %r186, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r45, %r186, %r185;
setp.ne.s32	%p32, %r183, 0;
@%p32 bra BB4_76;

mov.u32 %r418, 0;
@%p1 bra BB4_75;

and.b32 %r194, %r118, 3;
mov.u32 %r409, 0;
setp.eq.s32	%p34, %r194, 0;
@%p34 bra BB4_45;

setp.eq.s32	%p35, %r194, 1;
@%p35 bra BB4_47;
bra.uni BB4_48;

BB4_47:
mov.u32 %r412, %r409;
bra.uni BB4_56;

BB4_45:
mov.u32 %r418, %r409;
bra.uni BB4_60;

BB4_48:
setp.eq.s32	%p36, %r194, 2;
@%p36 bra BB4_49;
bra.uni BB4_50;

BB4_49:
mov.u32 %r408, %r409;
bra.uni BB4_52;

BB4_50:
ld.shared.u32 %r197, [_ZN8nvinfer112sparse_fipnn9smem_poolE];
mov.u32 %r408, 1;
setp.lt.s32	%p37, %r197, 0;
@%p37 bra BB4_52;

shl.b32 %r200, %r118, 2;
add.s32 %r202, %r186, %r200;
mov.u32 %r203, 0;
st.shared.u32 [%r202], %r203;
mov.u32 %r408, 1;
mov.u32 %r409, %r408;

BB4_52:
shl.b32 %r204, %r408, 2;
add.s32 %r206, %r186, %r204;
ld.shared.u32 %r207, [%r206];
setp.lt.s32	%p38, %r207, 0;
@%p38 bra BB4_53;

add.s32 %r412, %r409, 1;
add.s32 %r208, %r409, %r118;
shl.b32 %r209, %r208, 2;
add.s32 %r211, %r186, %r209;
st.shared.u32 [%r211], %r408;
bra.uni BB4_55;

BB4_53:
mov.u32 %r412, %r409;

BB4_55:
add.s32 %r409, %r408, 1;

BB4_56:
shl.b32 %r212, %r409, 2;
add.s32 %r214, %r186, %r212;
ld.shared.u32 %r215, [%r214];
setp.lt.s32	%p39, %r215, 0;
@%p39 bra BB4_57;

add.s32 %r418, %r412, 1;
add.s32 %r216, %r412, %r118;
shl.b32 %r217, %r216, 2;
add.s32 %r219, %r186, %r217;
st.shared.u32 [%r219], %r409;
bra.uni BB4_59;

BB4_57:
mov.u32 %r418, %r412;

BB4_59:
add.s32 %r409, %r409, 1;

BB4_60:
setp.lt.u32	%p40, %r118, 4;
@%p40 bra BB4_75;

shl.b32 %r220, %r409, 2;
add.s32 %r416, %r186, %r220;

BB4_62:
ld.shared.u32 %r222, [%r416];
setp.lt.s32	%p41, %r222, 0;
@%p41 bra BB4_63;

add.s32 %r419, %r418, 1;
add.s32 %r223, %r418, %r118;
shl.b32 %r224, %r223, 2;
add.s32 %r226, %r186, %r224;
st.shared.u32 [%r226], %r409;
bra.uni BB4_65;

BB4_63:
mov.u32 %r419, %r418;

BB4_65:
ld.shared.u32 %r227, [%r416+4];
setp.lt.s32	%p42, %r227, 0;
@%p42 bra BB4_66;

add.s32 %r420, %r419, 1;
add.s32 %r228, %r419, %r118;
shl.b32 %r229, %r228, 2;
add.s32 %r231, %r186, %r229;
add.s32 %r232, %r409, 1;
st.shared.u32 [%r231], %r232;
bra.uni BB4_68;

BB4_66:
mov.u32 %r420, %r419;

BB4_68:
ld.shared.u32 %r233, [%r416+8];
setp.lt.s32	%p43, %r233, 0;
@%p43 bra BB4_69;

add.s32 %r421, %r420, 1;
add.s32 %r234, %r420, %r118;
shl.b32 %r235, %r234, 2;
add.s32 %r237, %r186, %r235;
add.s32 %r238, %r409, 2;
st.shared.u32 [%r237], %r238;
bra.uni BB4_71;

BB4_69:
mov.u32 %r421, %r420;

BB4_71:
ld.shared.u32 %r239, [%r416+12];
setp.lt.s32	%p44, %r239, 0;
@%p44 bra BB4_72;

add.s32 %r418, %r421, 1;
add.s32 %r240, %r421, %r118;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r186, %r241;
add.s32 %r244, %r409, 3;
st.shared.u32 [%r243], %r244;
bra.uni BB4_74;

BB4_72:
mov.u32 %r418, %r421;

BB4_74:
add.s32 %r409, %r409, 4;
setp.lt.s32	%p45, %r409, %r118;
add.s32 %r416, %r416, 16;
@%p45 bra BB4_62;

BB4_75:
st.shared.u32 [%r45], %r418;

BB4_76:
bar.sync 0;
add.s32 %r245, %r116, 31;
shr.s32 %r246, %r245, 31;
shr.u32 %r247, %r246, 27;
add.s32 %r248, %r245, %r247;
and.b32 %r73, %r248, -32;
ld.shared.u32 %r74, [%r45];
setp.ge.s32	%p46, %r424, %r74;
@%p46 bra BB4_141;

setp.gt.s32	%p47, %r73, 32;
add.s32 %r249, %r73, -1;
shr.u32 %r250, %r249, 5;
add.s32 %r251, %r250, 1;
selp.b32	%r75, %r251, 1, %p47;
and.b32 %r77, %r75, 3;
mul.wide.s32 %rd73, %r4, 4;
add.s64 %rd24, %rd3, %rd73;
mul.lo.s32 %r253, %r1, %r118;
mul.lo.s32 %r254, %r253, %r116;
mul.lo.s32 %r256, %r254, %r124;
mul.wide.s32 %rd74, %r256, 4;
add.s64 %rd27, %rd3, %rd74;
mul.lo.s32 %r257, %r118, %r117;
mad.lo.s32 %r258, %r257, %r116, %r256;
mul.wide.s32 %rd75, %r258, 4;
add.s64 %rd28, %rd3, %rd75;

BB4_78:
add.s32 %r259, %r424, %r118;
shl.b32 %r260, %r259, 2;
add.s32 %r262, %r186, %r260;
ld.shared.u32 %r79, [%r262];
shl.b32 %r263, %r79, 2;
add.s32 %r264, %r186, %r263;
ld.shared.u32 %r80, [%r264];
add.s32 %r265, %r79, 2;
add.s32 %r266, %r79, 1;
mul.lo.s32 %r267, %r265, %r266;
shr.u32 %r268, %r267, 31;
add.s32 %r269, %r267, %r268;
shr.s32 %r270, %r269, 1;
sub.s32 %r81, %r270, %r266;
mov.f32 %f147, 0f00000000;
st.local.v4.f32 [%rd5], {%f147, %f147, %f147, %f147};
st.local.v4.f32 [%rd5+16], {%f147, %f147, %f147, %f147};
st.local.v4.f32 [%rd5+32], {%f147, %f147, %f147, %f147};
setp.lt.s32	%p48, %r116, 1;
@%p48 bra BB4_101;

mad.lo.s32 %r82, %r79, %r116, %r9;
add.s32 %r274, %r79, %r118;
mad.lo.s32 %r83, %r274, %r116, %r9;
mov.u32 %r425, 0;
setp.eq.s32	%p49, %r77, 0;
@%p49 bra BB4_90;

setp.eq.s32	%p50, %r77, 1;
@%p50 bra BB4_87;

setp.eq.s32	%p51, %r77, 2;
@%p51 bra BB4_84;

mov.u32 %r425, 32;
setp.ge.s32	%p52, %r9, %r116;
@%p52 bra BB4_84;

add.s32 %r277, %r82, %r4;
mul.wide.s32 %rd78, %r277, 4;
add.s64 %rd79, %rd3, %rd78;
ld.global.f32 %f57, [%rd79];
st.local.f32 [%rd5], %f57;
add.s32 %r278, %r83, %r4;
mul.wide.s32 %rd80, %r278, 4;
add.s64 %rd81, %rd3, %rd80;
ld.global.f32 %f58, [%rd81];
st.local.f32 [%rd5+24], %f58;

BB4_84:
add.s32 %r279, %r425, %r9;
setp.ge.s32	%p53, %r279, %r116;
@%p53 bra BB4_86;

add.s32 %r280, %r82, %r425;
add.s32 %r281, %r280, %r4;
mul.wide.s32 %rd82, %r281, 4;
add.s64 %rd83, %rd3, %rd82;
ld.global.f32 %f59, [%rd83];
shr.u32 %r282, %r425, 5;
mul.wide.u32 %rd84, %r282, 4;
add.s64 %rd85, %rd5, %rd84;
st.local.f32 [%rd85], %f59;
add.s32 %r283, %r83, %r425;
add.s32 %r284, %r283, %r4;
mul.wide.s32 %rd86, %r284, 4;
add.s64 %rd87, %rd3, %rd86;
ld.global.f32 %f60, [%rd87];
st.local.f32 [%rd85+24], %f60;

BB4_86:
add.s32 %r425, %r425, 32;

BB4_87:
add.s32 %r285, %r425, %r9;
setp.ge.s32	%p54, %r285, %r116;
@%p54 bra BB4_89;

add.s32 %r286, %r82, %r425;
add.s32 %r287, %r286, %r4;
mul.wide.s32 %rd88, %r287, 4;
add.s64 %rd89, %rd3, %rd88;
ld.global.f32 %f61, [%rd89];
shr.s32 %r288, %r425, 31;
shr.u32 %r289, %r288, 27;
add.s32 %r290, %r425, %r289;
shr.s32 %r291, %r290, 5;
mul.wide.s32 %rd90, %r291, 4;
add.s64 %rd91, %rd5, %rd90;
st.local.f32 [%rd91], %f61;
add.s32 %r292, %r83, %r425;
add.s32 %r293, %r292, %r4;
mul.wide.s32 %rd92, %r293, 4;
add.s64 %rd93, %rd3, %rd92;
ld.global.f32 %f62, [%rd93];
st.local.f32 [%rd91+24], %f62;

BB4_89:
add.s32 %r425, %r425, 32;

BB4_90:
setp.lt.u32	%p55, %r75, 4;
@%p55 bra BB4_101;

add.s32 %r428, %r9, %r425;
mad.lo.s32 %r294, %r116, %r79, %r428;
mul.wide.s32 %rd94, %r294, 4;
add.s64 %rd129, %rd27, %rd94;
add.s32 %r295, %r118, %r79;
mad.lo.s32 %r296, %r116, %r295, %r428;
mul.wide.s32 %rd95, %r296, 4;
add.s64 %rd130, %rd27, %rd95;

BB4_92:
setp.ge.s32	%p56, %r428, %r116;
@%p56 bra BB4_94;

ld.global.f32 %f63, [%rd129];
shr.s32 %r297, %r425, 31;
shr.u32 %r298, %r297, 27;
add.s32 %r299, %r425, %r298;
shr.s32 %r300, %r299, 5;
mul.wide.s32 %rd96, %r300, 4;
add.s64 %rd97, %rd5, %rd96;
st.local.f32 [%rd97], %f63;
ld.global.f32 %f64, [%rd130];
st.local.f32 [%rd97+24], %f64;

BB4_94:
add.s32 %r301, %r428, 32;
setp.ge.s32	%p57, %r301, %r116;
@%p57 bra BB4_96;

ld.global.f32 %f65, [%rd129+128];
add.s32 %r302, %r425, 32;
shr.s32 %r303, %r302, 31;
shr.u32 %r304, %r303, 27;
add.s32 %r305, %r302, %r304;
shr.s32 %r306, %r305, 5;
mul.wide.s32 %rd98, %r306, 4;
add.s64 %rd99, %rd5, %rd98;
st.local.f32 [%rd99], %f65;
ld.global.f32 %f66, [%rd130+128];
st.local.f32 [%rd99+24], %f66;

BB4_96:
add.s32 %r307, %r428, 64;
setp.ge.s32	%p58, %r307, %r116;
@%p58 bra BB4_98;

ld.global.f32 %f67, [%rd129+256];
add.s32 %r308, %r425, 64;
shr.s32 %r309, %r308, 31;
shr.u32 %r310, %r309, 27;
add.s32 %r311, %r308, %r310;
shr.s32 %r312, %r311, 5;
mul.wide.s32 %rd100, %r312, 4;
add.s64 %rd101, %rd5, %rd100;
st.local.f32 [%rd101], %f67;
ld.global.f32 %f68, [%rd130+256];
st.local.f32 [%rd101+24], %f68;

BB4_98:
add.s32 %r313, %r428, 96;
setp.ge.s32	%p59, %r313, %r116;
@%p59 bra BB4_100;

ld.global.f32 %f69, [%rd129+384];
add.s32 %r314, %r425, 96;
shr.s32 %r315, %r314, 31;
shr.u32 %r316, %r315, 27;
add.s32 %r317, %r314, %r316;
shr.s32 %r318, %r317, 5;
mul.wide.s32 %rd102, %r318, 4;
add.s64 %rd103, %rd5, %rd102;
st.local.f32 [%rd103], %f69;
ld.global.f32 %f70, [%rd130+384];
st.local.f32 [%rd103+24], %f70;

BB4_100:
add.s32 %r425, %r425, 128;
add.s32 %r428, %r428, 128;
setp.lt.s32	%p60, %r425, %r73;
add.s64 %rd129, %rd129, 512;
add.s64 %rd130, %rd130, 512;
@%p60 bra BB4_92;

BB4_101:
setp.lt.s32	%p61, %r424, 1;
@%p61 bra BB4_113;

mul.lo.s32 %r94, %r118, %r80;
mov.u32 %r430, 0;

BB4_103:
add.s32 %r320, %r430, %r118;
shl.b32 %r321, %r320, 2;
add.s32 %r323, %r186, %r321;
ld.shared.u32 %r96, [%r323];
mov.f32 %f128, 0f00000000;
@%p48 bra BB4_110;

shl.b32 %r325, %r96, 2;
add.s32 %r327, %r186, %r325;
ld.shared.u32 %r97, [%r327];
add.s32 %r328, %r94, %r96;
mad.lo.s32 %r329, %r116, %r328, %r9;
mul.wide.s32 %rd104, %r329, 4;
add.s64 %rd131, %rd24, %rd104;
mov.f32 %f72, 0f00000000;
mov.u32 %r432, 0;
mov.u32 %r431, %r9;
mov.f32 %f128, %f72;

BB4_105:
setp.ge.s32	%p63, %r431, %r116;
mov.f32 %f126, %f72;
mov.f32 %f127, %f72;
@%p63 bra BB4_109;

setp.eq.s32	%p64, %r97, 0;
ld.global.f32 %f126, [%rd131];
shr.s32 %r330, %r432, 31;
shr.u32 %r331, %r330, 27;
add.s32 %r332, %r432, %r331;
shr.s32 %r333, %r332, 5;
mul.wide.s32 %rd105, %r333, 4;
add.s64 %rd38, %rd5, %rd105;
@%p64 bra BB4_108;
bra.uni BB4_107;

BB4_108:
ld.local.f32 %f127, [%rd38];
bra.uni BB4_109;

BB4_107:
ld.local.f32 %f127, [%rd38+24];

BB4_109:
fma.rn.f32 %f128, %f126, %f127, %f128;
add.s32 %r431, %r431, 32;
add.s64 %rd131, %rd131, 128;
add.s32 %r432, %r432, 32;
setp.lt.s32	%p65, %r432, %r73;
@%p65 bra BB4_105;

BB4_110:
bar.warp.sync -1;
mov.b32 %r334, %f128;
mov.u32 %r335, 31;
mov.u32 %r336, 1;
mov.u32 %r337, -1;
shfl.sync.bfly.b32 %r338|%p66, %r334, %r336, %r335, %r337;
mov.b32 %f75, %r338;
add.f32 %f76, %f128, %f75;
mov.b32 %r339, %f76;
mov.u32 %r340, 2;
shfl.sync.bfly.b32 %r341|%p67, %r339, %r340, %r335, %r337;
mov.b32 %f77, %r341;
add.f32 %f78, %f76, %f77;
mov.b32 %r342, %f78;
mov.u32 %r343, 4;
shfl.sync.bfly.b32 %r344|%p68, %r342, %r343, %r335, %r337;
mov.b32 %f79, %r344;
add.f32 %f80, %f78, %f79;
mov.b32 %r345, %f80;
mov.u32 %r346, 8;
shfl.sync.bfly.b32 %r347|%p69, %r345, %r346, %r335, %r337;
mov.b32 %f81, %r347;
add.f32 %f82, %f80, %f81;
mov.b32 %r348, %f82;
mov.u32 %r349, 16;
shfl.sync.bfly.b32 %r350|%p70, %r348, %r349, %r335, %r337;
mov.b32 %f83, %r350;
add.f32 %f9, %f82, %f83;
setp.ne.s32	%p71, %r9, 0;
@%p71 bra BB4_112;

add.s32 %r351, %r96, %r81;
add.s32 %r352, %r351, %r2;
mul.wide.s32 %rd106, %r352, 4;
add.s64 %rd107, %rd4, %rd106;
st.global.f32 [%rd107], %f9;

BB4_112:
bar.warp.sync -1;
add.s32 %r430, %r430, 1;
setp.lt.s32	%p72, %r430, %r424;
@%p72 bra BB4_103;

BB4_113:
@%p48 bra BB4_138;

mad.lo.s32 %r356, %r80, %r118, %r79;
mad.lo.s32 %r103, %r356, %r116, %r9;
mad.lo.s32 %r104, %r79, %r116, %r9;
mov.f32 %f147, 0f00000000;
mov.u32 %r435, 0;
setp.eq.s32	%p74, %r77, 0;
@%p74 bra BB4_127;

setp.eq.s32	%p75, %r77, 1;
@%p75 bra BB4_116;
bra.uni BB4_117;

BB4_116:
mov.f32 %f134, %f147;
bra.uni BB4_124;

BB4_117:
setp.eq.s32	%p76, %r77, 2;
mov.f32 %f131, %f147;
@%p76 bra BB4_121;

mov.f32 %f129, 0f00000000;
setp.ge.s32	%p77, %r9, %r116;
mov.f32 %f130, %f129;
@%p77 bra BB4_120;

add.s32 %r357, %r103, %r4;
mul.wide.s32 %rd108, %r357, 4;
add.s64 %rd109, %rd3, %rd108;
ld.global.f32 %f129, [%rd109];
add.s32 %r358, %r104, %r5;
mul.wide.s32 %rd110, %r358, 4;
add.s64 %rd111, %rd3, %rd110;
ld.global.f32 %f130, [%rd111];

BB4_120:
mul.f32 %f90, %f129, %f129;
sub.f32 %f91, %f90, %f130;
fma.rn.f32 %f131, %f91, 0f3F000000, 0f00000000;
mov.u32 %r435, 32;

BB4_121:
add.s32 %r360, %r435, %r9;
setp.ge.s32	%p78, %r360, %r116;
mov.f32 %f132, %f147;
mov.f32 %f133, %f147;
@%p78 bra BB4_123;

add.s32 %r361, %r103, %r435;
add.s32 %r362, %r361, %r4;
mul.wide.s32 %rd112, %r362, 4;
add.s64 %rd113, %rd3, %rd112;
ld.global.f32 %f132, [%rd113];
add.s32 %r363, %r104, %r435;
add.s32 %r364, %r363, %r5;
mul.wide.s32 %rd114, %r364, 4;
add.s64 %rd115, %rd3, %rd114;
ld.global.f32 %f133, [%rd115];

BB4_123:
mul.f32 %f94, %f132, %f132;
sub.f32 %f95, %f94, %f133;
fma.rn.f32 %f134, %f95, 0f3F000000, %f131;
add.s32 %r435, %r435, 32;

BB4_124:
add.s32 %r365, %r435, %r9;
setp.ge.s32	%p79, %r365, %r116;
mov.f32 %f136, %f147;
@%p79 bra BB4_126;

add.s32 %r366, %r103, %r435;
add.s32 %r367, %r366, %r4;
mul.wide.s32 %rd116, %r367, 4;
add.s64 %rd117, %rd3, %rd116;
ld.global.f32 %f147, [%rd117];
add.s32 %r368, %r104, %r435;
add.s32 %r369, %r368, %r5;
mul.wide.s32 %rd118, %r369, 4;
add.s64 %rd119, %rd3, %rd118;
ld.global.f32 %f136, [%rd119];

BB4_126:
mul.f32 %f98, %f147, %f147;
sub.f32 %f99, %f98, %f136;
fma.rn.f32 %f147, %f99, 0f3F000000, %f134;
add.s32 %r435, %r435, 32;

BB4_127:
setp.lt.u32	%p80, %r75, 4;
@%p80 bra BB4_138;

add.s32 %r436, %r9, %r435;
mad.lo.s32 %r370, %r118, %r80, %r79;
mad.lo.s32 %r371, %r116, %r370, %r436;
mul.wide.s32 %rd120, %r371, 4;
add.s64 %rd132, %rd27, %rd120;
mad.lo.s32 %r372, %r116, %r79, %r436;
mul.wide.s32 %rd121, %r372, 4;
add.s64 %rd133, %rd28, %rd121;

BB4_129:
mov.f32 %f145, 0f00000000;
setp.ge.s32	%p81, %r436, %r116;
mov.f32 %f139, %f145;
mov.f32 %f140, %f145;
@%p81 bra BB4_131;

ld.global.f32 %f139, [%rd132];
ld.global.f32 %f140, [%rd133];

BB4_131:
mul.f32 %f104, %f139, %f139;
sub.f32 %f105, %f104, %f140;
fma.rn.f32 %f33, %f105, 0f3F000000, %f147;
add.s32 %r373, %r436, 32;
setp.ge.s32	%p82, %r373, %r116;
mov.f32 %f141, %f145;
mov.f32 %f142, %f145;
@%p82 bra BB4_133;

ld.global.f32 %f141, [%rd132+128];
ld.global.f32 %f142, [%rd133+128];

BB4_133:
mul.f32 %f108, %f141, %f141;
sub.f32 %f109, %f108, %f142;
fma.rn.f32 %f38, %f109, 0f3F000000, %f33;
add.s32 %r374, %r436, 64;
setp.ge.s32	%p83, %r374, %r116;
mov.f32 %f143, %f145;
mov.f32 %f144, %f145;
@%p83 bra BB4_135;

ld.global.f32 %f143, [%rd132+256];
ld.global.f32 %f144, [%rd133+256];

BB4_135:
mul.f32 %f112, %f143, %f143;
sub.f32 %f113, %f112, %f144;
fma.rn.f32 %f43, %f113, 0f3F000000, %f38;
add.s32 %r375, %r436, 96;
setp.ge.s32	%p84, %r375, %r116;
mov.f32 %f146, %f145;
@%p84 bra BB4_137;

ld.global.f32 %f145, [%rd132+384];
ld.global.f32 %f146, [%rd133+384];

BB4_137:
mul.f32 %f114, %f145, %f145;
sub.f32 %f115, %f114, %f146;
fma.rn.f32 %f147, %f115, 0f3F000000, %f43;
add.s32 %r436, %r436, 128;
add.s32 %r435, %r435, 128;
setp.lt.s32	%p85, %r435, %r73;
add.s64 %rd132, %rd132, 512;
add.s64 %rd133, %rd133, 512;
@%p85 bra BB4_129;

BB4_138:
bar.warp.sync -1;
mov.b32 %r376, %f147;
mov.u32 %r377, 31;
mov.u32 %r378, 1;
mov.u32 %r379, -1;
shfl.sync.bfly.b32 %r380|%p86, %r376, %r378, %r377, %r379;
mov.b32 %f116, %r380;
add.f32 %f117, %f147, %f116;
mov.b32 %r381, %f117;
mov.u32 %r382, 2;
shfl.sync.bfly.b32 %r383|%p87, %r381, %r382, %r377, %r379;
mov.b32 %f118, %r383;
add.f32 %f119, %f117, %f118;
mov.b32 %r384, %f119;
mov.u32 %r385, 4;
shfl.sync.bfly.b32 %r386|%p88, %r384, %r385, %r377, %r379;
mov.b32 %f120, %r386;
add.f32 %f121, %f119, %f120;
mov.b32 %r387, %f121;
mov.u32 %r388, 8;
shfl.sync.bfly.b32 %r389|%p89, %r387, %r388, %r377, %r379;
mov.b32 %f122, %r389;
add.f32 %f123, %f121, %f122;
mov.b32 %r390, %f123;
mov.u32 %r391, 16;
shfl.sync.bfly.b32 %r392|%p90, %r390, %r391, %r377, %r379;
mov.b32 %f124, %r392;
add.f32 %f50, %f123, %f124;
setp.ne.s32	%p91, %r9, 0;
@%p91 bra BB4_140;

add.s32 %r393, %r81, %r79;
add.s32 %r394, %r393, %r2;
mul.wide.s32 %rd122, %r394, 4;
add.s64 %rd123, %rd4, %rd122;
st.global.f32 [%rd123], %f50;

BB4_140:
bar.warp.sync -1;
add.s32 %r424, %r424, 32;
setp.lt.s32	%p92, %r424, %r74;
@%p92 bra BB4_78;

BB4_141:
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3_(
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_2,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_6,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_7,
.param .u64 _ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_8
)
{
.reg .pred %p<19>;
.reg .b16 %rs<10>;
.reg .f32 %f<3>;
.reg .b32 %r<56>;
.reg .b64 %rd<42>;


ld.param.u32 %r22, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_0];
ld.param.u32 %r23, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_1];
ld.param.u32 %r24, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_2];
ld.param.u64 %rd18, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_3];
ld.param.u64 %rd13, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_4];
ld.param.u64 %rd14, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_5];
ld.param.u64 %rd15, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_6];
ld.param.u64 %rd16, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_7];
ld.param.u64 %rd17, [_ZN8nvinfer112sparse_fipnn17ProcessCommonPartI6__halfLi32EEEviiiPiPKT_PKiPS4_S9_S3__param_8];
cvta.to.global.u64 %rd19, %rd18;
mov.u32 %r25, %ctaid.x;
shl.b32 %r26, %r25, 5;
mov.u32 %r27, %tid.y;
add.s32 %r50, %r26, %r27;
ld.global.u32 %r2, [%rd19];
setp.ge.s32	%p1, %r50, %r2;
@%p1 bra BB5_19;

add.s32 %r28, %r22, 31;
shr.s32 %r29, %r28, 31;
shr.u32 %r30, %r29, 27;
add.s32 %r31, %r28, %r30;
and.b32 %r3, %r31, -32;
cvta.to.global.u64 %rd20, %rd14;
cvta.to.global.u64 %rd23, %rd17;

BB5_2:
shl.b32 %r32, %r50, 1;
mul.wide.s32 %rd21, %r32, 4;
add.s64 %rd22, %rd20, %rd21;
ld.global.u32 %r5, [%rd22];
add.s32 %r6, %r5, -1;
ld.global.u32 %r7, [%rd22+4];
add.s32 %r8, %r7, -1;
setp.lt.s32	%p2, %r8, 0;
setp.ge.s32	%p3, %r8, %r24;
or.pred %p4, %p2, %p3;
setp.lt.s32	%p5, %r6, 0;
or.pred %p6, %p4, %p5;
setp.ge.s32	%p7, %r6, %r23;
or.pred %p8, %p6, %p7;
@%p8 bra BB5_18;

mov.u32 %r54, %tid.x;
setp.ne.s32	%p9, %r54, 0;
@%p9 bra BB5_5;

mul.wide.s32 %rd24, %r8, 4;
add.s64 %rd25, %rd23, %rd24;
add.s32 %r49, %r5, -1;
st.global.u32 [%rd25], %r49;

BB5_5:
setp.lt.s32	%p10, %r23, 1;
@%p10 bra BB5_13;

mov.u32 %r51, 0;

BB5_7:
setp.lt.s32	%p11, %r22, 1;
@%p11 bra BB5_12;

mad.lo.s32 %r36, %r23, %r50, %r51;
mov.u32 %r52, %tid.x;
mad.lo.s32 %r37, %r22, %r36, %r52;
cvta.to.global.u64 %rd26, %rd13;
mul.wide.s32 %rd27, %r37, 2;
add.s64 %rd39, %rd26, %rd27;
mad.lo.s32 %r38, %r24, %r51, %r7;
add.s32 %r39, %r38, -1;
mul.lo.s32 %r40, %r22, %r39;
mul.wide.s32 %rd28, %r52, 2;
add.s64 %rd29, %rd15, %rd28;
mul.wide.s32 %rd30, %r40, 2;
add.s64 %rd38, %rd29, %rd30;
mov.u32 %r53, 0;

BB5_9:
setp.ge.s32	%p12, %r52, %r22;
@%p12 bra BB5_11;

ld.global.u16 %rs2, [%rd39];

	{ atom.add.noftz.f16 %rs1,[%rd38],%rs2; }



BB5_11:
add.s32 %r53, %r53, 32;
add.s64 %rd39, %rd39, 64;
add.s32 %r52, %r52, 32;
add.s64 %rd38, %rd38, 64;
setp.lt.s32	%p13, %r53, %r3;
@%p13 bra BB5_9;

BB5_12:
add.s32 %r51, %r51, 1;
setp.lt.s32	%p14, %r51, %r23;
@%p14 bra BB5_7;

BB5_13:
setp.lt.s32	%p15, %r22, 1;
@%p15 bra BB5_18;

mad.lo.s32 %r42, %r23, %r50, %r5;
add.s32 %r43, %r42, -1;
mad.lo.s32 %r44, %r22, %r43, %r54;
cvta.to.global.u64 %rd32, %rd13;
mul.wide.s32 %rd33, %r44, 2;
add.s64 %rd41, %rd32, %rd33;
mul.lo.s32 %r46, %r22, %r8;
mul.wide.s32 %rd34, %r54, 2;
add.s64 %rd35, %rd16, %rd34;
mul.wide.s32 %rd36, %r46, 2;
add.s64 %rd40, %rd35, %rd36;
mov.u32 %r55, 0;

BB5_15:
setp.ge.s32	%p16, %r54, %r22;
@%p16 bra BB5_17;

ld.global.u16 %rs4, [%rd41];

	{mul.f16 %rs3,%rs4,%rs4;
}

	
	{ cvt.f32.f16 %f1, %rs3;}


	
	{ cvt.rn.f16.f32 %rs7, %f1;}


	
	{ atom.add.noftz.f16 %rs8,[%rd40],%rs7; }



BB5_17:
add.s32 %r55, %r55, 32;
add.s64 %rd41, %rd41, 64;
add.s32 %r54, %r54, 32;
add.s64 %rd40, %rd40, 64;
setp.lt.s32	%p17, %r55, %r3;
@%p17 bra BB5_15;

BB5_18:
mov.u32 %r47, %nctaid.x;
shl.b32 %r48, %r47, 5;
add.s32 %r50, %r50, %r48;
setp.lt.s32	%p18, %r50, %r2;
@%p18 bra BB5_2;

BB5_19:
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4_(
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_2,
.param .u32 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_6
)
{
.reg .b16 %rs<4>;
.reg .b32 %r<20>;
.reg .b64 %rd<18>;


ld.param.u32 %r1, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_1];
ld.param.u32 %r2, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_2];
ld.param.u32 %r3, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_3];
ld.param.u64 %rd1, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_4];
ld.param.u64 %rd2, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_5];
ld.param.u64 %rd3, [_ZN8nvinfer112sparse_fipnn19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_6];
cvta.to.global.u64 %rd4, %rd1;
cvta.to.global.u64 %rd5, %rd3;
cvta.to.global.u64 %rd6, %rd2;
mov.u32 %r4, %tid.x;
mov.u32 %r5, %ctaid.x;
rem.u32 %r6, %r5, %r3;
mad.lo.s32 %r7, %r6, %r1, %r4;
div.u32 %r8, %r5, %r3;
mul.wide.s32 %rd7, %r7, 2;
add.s64 %rd8, %rd6, %rd7;
ld.global.u16 %rs1, [%rd8];
add.s32 %r9, %r2, 1;
mul.lo.s32 %r10, %r9, %r1;
mul.lo.s32 %r11, %r10, %r3;
mul.lo.s32 %r12, %r11, %r8;
mul.lo.s32 %r13, %r2, %r1;
mad.lo.s32 %r14, %r13, %r3, %r12;
add.s32 %r15, %r14, %r7;
mul.wide.s32 %rd9, %r15, 2;
add.s64 %rd10, %rd5, %rd9;
st.global.u16 [%rd10], %rs1;
add.s64 %rd11, %rd4, %rd7;
ld.global.u16 %rs2, [%rd11];
mul.lo.s32 %r16, %r3, %r1;
add.s32 %r17, %r7, %r16;
mul.wide.s32 %rd12, %r17, 2;
add.s64 %rd13, %rd4, %rd12;
ld.global.u16 %rs3, [%rd13];
add.s32 %r18, %r12, %r7;
mul.wide.s32 %rd14, %r18, 2;
add.s64 %rd15, %rd5, %rd14;
st.global.u16 [%rd15], %rs2;
add.s32 %r19, %r18, %r16;
mul.wide.s32 %rd16, %r19, 2;
add.s64 %rd17, %rd5, %rd16;
st.global.u16 [%rd17], %rs3;
ret;
}


.visible .entry _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5_(
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_0,
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_1,
.param .u32 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_2,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_3,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_4,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_5,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_6,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_7,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_8,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_9,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_10,
.param .u64 _ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_11
)
{
.local .align 8 .b8 __local_depot7[24];
.reg .b64 %SP;
.reg .b64 %SPL;
.reg .pred %p<83>;
.reg .b16 %rs<260>;
.reg .f32 %f<3>;
.reg .b32 %r<505>;
.reg .f64 %fd<8>;
.reg .b64 %rd<136>;


mov.u64 %SPL, __local_depot7;
ld.param.u32 %r116, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_0];
ld.param.u32 %r117, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_1];
ld.param.u32 %r118, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_2];
ld.param.u64 %rd48, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_3];
ld.param.u64 %rd49, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_7];
ld.param.u64 %rd50, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_8];
ld.param.u64 %rd46, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_9];
ld.param.u64 %rd51, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_10];
ld.param.u64 %rd47, [_ZN8nvinfer112sparse_fipnn14SparseFIPNNGpuI6__halfLi32EEEviiiPiS3_PT_S5_S3_PKS4_PKiS5_S5__param_11];
cvta.to.global.u64 %rd1, %rd49;
cvta.to.global.u64 %rd2, %rd50;
cvta.to.global.u64 %rd3, %rd47;
cvta.to.global.u64 %rd4, %rd51;
add.u64 %rd5, %SPL, 0;
add.s32 %r119, %r118, 1;
mul.lo.s32 %r120, %r119, %r118;
shr.u32 %r121, %r120, 31;
add.s32 %r122, %r120, %r121;
shr.s32 %r123, %r122, 1;
mov.u32 %r1, %ctaid.x;
mul.lo.s32 %r2, %r123, %r1;
add.s32 %r124, %r117, 1;
mul.lo.s32 %r125, %r124, %r116;
mul.lo.s32 %r3, %r125, %r118;
mul.lo.s32 %r4, %r3, %r1;
mul.lo.s32 %r126, %r117, %r116;
mad.lo.s32 %r5, %r126, %r118, %r4;
cvta.to.global.u64 %rd53, %rd48;
mul.wide.s32 %rd54, %r1, 4;
add.s64 %rd55, %rd53, %rd54;
ld.global.u32 %r6, [%rd55];
ld.global.u32 %r7, [%rd55+4];
mov.u32 %r127, %ntid.x;
mov.u32 %r491, %tid.y;
mov.u32 %r9, %tid.x;
mad.lo.s32 %r10, %r127, %r491, %r9;
setp.lt.s32	%p1, %r118, 1;
@%p1 bra BB7_23;

add.s32 %r132, %r118, -1;
shr.u32 %r133, %r132, 10;
add.s32 %r11, %r133, 1;
and.b32 %r131, %r11, 3;
mov.u32 %r463, 0;
setp.eq.s32	%p2, %r131, 0;
@%p2 bra BB7_12;

setp.eq.s32	%p3, %r131, 1;
@%p3 bra BB7_9;

setp.eq.s32	%p4, %r131, 2;
@%p4 bra BB7_6;

mov.u32 %r463, 1024;
setp.ge.s32	%p5, %r10, %r118;
@%p5 bra BB7_6;

mul.wide.s32 %rd56, %r10, 4;
add.s64 %rd57, %rd1, %rd56;
ld.global.u32 %r136, [%rd57];
shl.b32 %r137, %r10, 2;
mov.u32 %r138, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r139, %r138, %r137;
st.shared.u32 [%r139], %r136;

BB7_6:
add.s32 %r13, %r10, %r463;
setp.ge.s32	%p6, %r13, %r118;
@%p6 bra BB7_8;

mul.wide.s32 %rd58, %r13, 4;
add.s64 %rd59, %rd1, %rd58;
ld.global.u32 %r140, [%rd59];
shl.b32 %r141, %r13, 2;
mov.u32 %r142, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r143, %r142, %r141;
st.shared.u32 [%r143], %r140;

BB7_8:
add.s32 %r463, %r463, 1024;

BB7_9:
add.s32 %r16, %r10, %r463;
setp.ge.s32	%p7, %r16, %r118;
@%p7 bra BB7_11;

mul.wide.s32 %rd60, %r16, 4;
add.s64 %rd61, %rd1, %rd60;
ld.global.u32 %r144, [%rd61];
shl.b32 %r145, %r16, 2;
mov.u32 %r146, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r147, %r146, %r145;
st.shared.u32 [%r147], %r144;

BB7_11:
add.s32 %r463, %r463, 1024;

BB7_12:
setp.lt.u32	%p8, %r11, 4;
@%p8 bra BB7_23;

add.s32 %r466, %r10, %r463;
shl.b32 %r148, %r466, 2;
mov.u32 %r149, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r467, %r149, %r148;
mul.wide.s32 %rd62, %r466, 4;
add.s64 %rd126, %rd1, %rd62;

BB7_14:
setp.ge.s32	%p9, %r466, %r118;
@%p9 bra BB7_16;

ld.global.u32 %r150, [%rd126];
st.shared.u32 [%r467], %r150;

BB7_16:
add.s32 %r151, %r466, 1024;
setp.ge.s32	%p10, %r151, %r118;
@%p10 bra BB7_18;

ld.global.u32 %r152, [%rd126+4096];
st.shared.u32 [%r467+4096], %r152;

BB7_18:
add.s32 %r153, %r466, 2048;
setp.ge.s32	%p11, %r153, %r118;
@%p11 bra BB7_20;

ld.global.u32 %r154, [%rd126+8192];
st.shared.u32 [%r467+8192], %r154;

BB7_20:
add.s32 %r155, %r466, 3072;
setp.ge.s32	%p12, %r155, %r118;
@%p12 bra BB7_22;

ld.global.u32 %r156, [%rd126+12288];
st.shared.u32 [%r467+12288], %r156;

BB7_22:
add.s32 %r463, %r463, 4096;
add.s64 %rd126, %rd126, 16384;
add.s32 %r466, %r466, 4096;
setp.lt.s32	%p13, %r463, %r118;
add.s32 %r467, %r467, 16384;
@%p13 bra BB7_14;

BB7_23:
bar.sync 0;
add.s32 %r469, %r491, %r6;
setp.ge.s32	%p14, %r469, %r7;
@%p14 bra BB7_42;

cvta.to.global.u64 %rd9, %rd46;
add.s32 %r157, %r116, 31;
shr.s32 %r158, %r157, 31;
shr.u32 %r159, %r158, 27;
add.s32 %r160, %r157, %r159;
and.b32 %r28, %r160, -32;
mad.lo.s32 %r161, %r3, %r1, %r9;
mul.wide.s32 %rd63, %r161, 2;
add.s64 %rd10, %rd47, %rd63;
add.s32 %r162, %r5, %r9;
mul.wide.s32 %rd64, %r162, 2;
add.s64 %rd11, %rd47, %rd64;

BB7_25:
shl.b32 %r163, %r469, 1;
mul.wide.s32 %rd65, %r163, 4;
add.s64 %rd66, %rd9, %rd65;
add.s32 %r164, %r163, 1;
mul.wide.s32 %rd67, %r164, 4;
add.s64 %rd68, %rd9, %rd67;
ld.global.u32 %r30, [%rd68];
add.s32 %r165, %r30, -1;
setp.lt.s32	%p15, %r165, 0;
setp.ge.s32	%p16, %r165, %r118;
or.pred %p17, %p15, %p16;
ld.global.u32 %r31, [%rd66];
add.s32 %r166, %r31, -1;
setp.lt.s32	%p18, %r166, 0;
or.pred %p19, %p17, %p18;
setp.ge.s32	%p20, %r166, %r117;
or.pred %p21, %p19, %p20;
@%p21 bra BB7_41;

setp.ne.s32	%p22, %r9, 0;
@%p22 bra BB7_28;

shl.b32 %r167, %r30, 2;
mov.u32 %r168, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r169, %r167, %r168;
add.s32 %r462, %r31, -1;
st.shared.u32 [%r169+-4], %r462;

BB7_28:
setp.lt.s32	%p23, %r117, 1;
@%p23 bra BB7_36;

mul.lo.s32 %r32, %r117, %r469;
mov.u32 %r470, 0;

BB7_30:
setp.lt.s32	%p24, %r116, 1;
@%p24 bra BB7_35;

add.s32 %r173, %r32, %r470;
mad.lo.s32 %r174, %r116, %r173, %r9;
mul.wide.s32 %rd69, %r174, 2;
add.s64 %rd128, %rd2, %rd69;
mad.lo.s32 %r175, %r118, %r470, %r165;
mul.lo.s32 %r176, %r116, %r175;
mul.wide.s32 %rd70, %r176, 2;
add.s64 %rd127, %rd10, %rd70;
mov.u32 %r472, 0;
mov.u32 %r471, %r9;

BB7_32:
setp.ge.s32	%p25, %r471, %r116;
@%p25 bra BB7_34;

ld.global.u16 %rs54, [%rd128];

	{ atom.add.noftz.f16 %rs53,[%rd127],%rs54; }



BB7_34:
add.s32 %r472, %r472, 32;
add.s64 %rd128, %rd128, 64;
add.s32 %r471, %r471, 32;
add.s64 %rd127, %rd127, 64;
setp.lt.s32	%p26, %r472, %r28;
@%p26 bra BB7_32;

BB7_35:
add.s32 %r470, %r470, 1;
setp.lt.s32	%p27, %r470, %r117;
@%p27 bra BB7_30;

BB7_36:
setp.lt.s32	%p28, %r116, 1;
@%p28 bra BB7_41;

mad.lo.s32 %r178, %r117, %r469, %r31;
add.s32 %r179, %r178, -1;
mad.lo.s32 %r180, %r116, %r179, %r9;
mul.wide.s32 %rd72, %r180, 2;
add.s64 %rd130, %rd2, %rd72;
mul.lo.s32 %r182, %r116, %r165;
mul.wide.s32 %rd73, %r182, 2;
add.s64 %rd129, %rd11, %rd73;
mov.u32 %r474, 0;
mov.u32 %r473, %r9;

BB7_38:
setp.ge.s32	%p29, %r473, %r116;
@%p29 bra BB7_40;

ld.global.u16 %rs56, [%rd130];

	{mul.f16 %rs55,%rs56,%rs56;
}

	
	{ cvt.f32.f16 %f1, %rs55;}


	
	{ cvt.rn.f16.f32 %rs59, %f1;}


	
	{ atom.add.noftz.f16 %rs60,[%rd129],%rs59; }



BB7_40:
add.s32 %r474, %r474, 32;
add.s64 %rd130, %rd130, 64;
add.s32 %r473, %r473, 32;
add.s64 %rd129, %rd129, 64;
setp.lt.s32	%p30, %r474, %r28;
@%p30 bra BB7_38;

BB7_41:
add.s32 %r469, %r469, 32;
setp.lt.s32	%p31, %r469, %r7;
@%p31 bra BB7_25;

BB7_42:
bar.sync 0;
or.b32 %r183, %r9, %r491;
add.s32 %r184, %r118, %r118;
shl.b32 %r185, %r184, 2;
mov.u32 %r186, _ZN8nvinfer112sparse_fipnn9smem_poolE;
add.s32 %r45, %r186, %r185;
setp.ne.s32	%p32, %r183, 0;
@%p32 bra BB7_76;

mov.u32 %r485, 0;
@%p1 bra BB7_75;

and.b32 %r194, %r118, 3;
mov.u32 %r476, 0;
setp.eq.s32	%p34, %r194, 0;
@%p34 bra BB7_45;

setp.eq.s32	%p35, %r194, 1;
@%p35 bra BB7_47;
bra.uni BB7_48;

BB7_47:
mov.u32 %r479, %r476;
bra.uni BB7_56;

BB7_45:
mov.u32 %r485, %r476;
bra.uni BB7_60;

BB7_48:
setp.eq.s32	%p36, %r194, 2;
@%p36 bra BB7_49;
bra.uni BB7_50;

BB7_49:
mov.u32 %r475, %r476;
bra.uni BB7_52;

BB7_50:
ld.shared.u32 %r197, [_ZN8nvinfer112sparse_fipnn9smem_poolE];
mov.u32 %r475, 1;
setp.lt.s32	%p37, %r197, 0;
@%p37 bra BB7_52;

shl.b32 %r200, %r118, 2;
add.s32 %r202, %r186, %r200;
mov.u32 %r203, 0;
st.shared.u32 [%r202], %r203;
mov.u32 %r475, 1;
mov.u32 %r476, %r475;

BB7_52:
shl.b32 %r204, %r475, 2;
add.s32 %r206, %r186, %r204;
ld.shared.u32 %r207, [%r206];
setp.lt.s32	%p38, %r207, 0;
@%p38 bra BB7_53;

add.s32 %r479, %r476, 1;
add.s32 %r208, %r476, %r118;
shl.b32 %r209, %r208, 2;
add.s32 %r211, %r186, %r209;
st.shared.u32 [%r211], %r475;
bra.uni BB7_55;

BB7_53:
mov.u32 %r479, %r476;

BB7_55:
add.s32 %r476, %r475, 1;

BB7_56:
shl.b32 %r212, %r476, 2;
add.s32 %r214, %r186, %r212;
ld.shared.u32 %r215, [%r214];
setp.lt.s32	%p39, %r215, 0;
@%p39 bra BB7_57;

add.s32 %r485, %r479, 1;
add.s32 %r216, %r479, %r118;
shl.b32 %r217, %r216, 2;
add.s32 %r219, %r186, %r217;
st.shared.u32 [%r219], %r476;
bra.uni BB7_59;

BB7_57:
mov.u32 %r485, %r479;

BB7_59:
add.s32 %r476, %r476, 1;

BB7_60:
setp.lt.u32	%p40, %r118, 4;
@%p40 bra BB7_75;

shl.b32 %r220, %r476, 2;
add.s32 %r483, %r186, %r220;

BB7_62:
ld.shared.u32 %r222, [%r483];
setp.lt.s32	%p41, %r222, 0;
@%p41 bra BB7_63;

add.s32 %r486, %r485, 1;
add.s32 %r223, %r485, %r118;
shl.b32 %r224, %r223, 2;
add.s32 %r226, %r186, %r224;
st.shared.u32 [%r226], %r476;
bra.uni BB7_65;

BB7_63:
mov.u32 %r486, %r485;

BB7_65:
ld.shared.u32 %r227, [%r483+4];
setp.lt.s32	%p42, %r227, 0;
@%p42 bra BB7_66;

add.s32 %r487, %r486, 1;
add.s32 %r228, %r486, %r118;
shl.b32 %r229, %r228, 2;
add.s32 %r231, %r186, %r229;
add.s32 %r232, %r476, 1;
st.shared.u32 [%r231], %r232;
bra.uni BB7_68;

BB7_66:
mov.u32 %r487, %r486;

BB7_68:
ld.shared.u32 %r233, [%r483+8];
setp.lt.s32	%p43, %r233, 0;
@%p43 bra BB7_69;

add.s32 %r488, %r487, 1;
add.s32 %r234, %r487, %r118;
shl.b32 %r235, %r234, 2;
add.s32 %r237, %r186, %r235;
add.s32 %r238, %r476, 2;
st.shared.u32 [%r237], %r238;
bra.uni BB7_71;

BB7_69:
mov.u32 %r488, %r487;

BB7_71:
ld.shared.u32 %r239, [%r483+12];
setp.lt.s32	%p44, %r239, 0;
@%p44 bra BB7_72;

add.s32 %r485, %r488, 1;
add.s32 %r240, %r488, %r118;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r186, %r241;
add.s32 %r244, %r476, 3;
st.shared.u32 [%r243], %r244;
bra.uni BB7_74;

BB7_72:
mov.u32 %r485, %r488;

BB7_74:
add.s32 %r476, %r476, 4;
setp.lt.s32	%p45, %r476, %r118;
add.s32 %r483, %r483, 16;
@%p45 bra BB7_62;

BB7_75:
st.shared.u32 [%r45], %r485;

BB7_76:
bar.sync 0;
add.s32 %r245, %r116, 31;
shr.s32 %r246, %r245, 31;
shr.u32 %r247, %r246, 27;
add.s32 %r248, %r245, %r247;
and.b32 %r73, %r248, -32;
ld.shared.u32 %r74, [%r45];
setp.ge.s32	%p46, %r491, %r74;
@%p46 bra BB7_143;

setp.gt.s32	%p47, %r73, 32;
add.s32 %r249, %r73, -1;
shr.u32 %r250, %r249, 5;
add.s32 %r251, %r250, 1;
selp.b32	%r75, %r251, 1, %p47;
and.b32 %r77, %r75, 3;
mul.wide.s32 %rd75, %r4, 2;
add.s64 %rd24, %rd3, %rd75;
mul.lo.s32 %r253, %r1, %r118;
mul.lo.s32 %r254, %r253, %r116;
mul.lo.s32 %r256, %r254, %r124;
mul.wide.s32 %rd76, %r256, 2;
add.s64 %rd27, %rd3, %rd76;
mul.lo.s32 %r257, %r118, %r117;
mad.lo.s32 %r258, %r257, %r116, %r256;
mul.wide.s32 %rd77, %r258, 2;
add.s64 %rd28, %rd3, %rd77;

	{mov.u32 %r336, WARP_SZ;
}

	shl.b32 %r375, %r336, 8;
add.s32 %r376, %r375, -8192;
or.b32 %r340, %r376, 31;

BB7_78:
mov.u16 %rs259, 0;
st.local.v4.u16 [%rd5], {%rs259, %rs259, %rs259, %rs259};
st.local.v4.u16 [%rd5+8], {%rs259, %rs259, %rs259, %rs259};
st.local.v4.u16 [%rd5+16], {%rs259, %rs259, %rs259, %rs259};
add.s32 %r260, %r491, %r118;
shl.b32 %r261, %r260, 2;
add.s32 %r263, %r186, %r261;
ld.shared.u32 %r79, [%r263];
shl.b32 %r264, %r79, 2;
add.s32 %r265, %r186, %r264;
ld.shared.u32 %r80, [%r265];
add.s32 %r266, %r79, 2;
add.s32 %r267, %r79, 1;
mul.lo.s32 %r268, %r266, %r267;
shr.u32 %r269, %r268, 31;
add.s32 %r270, %r268, %r269;
shr.s32 %r271, %r270, 1;
sub.s32 %r81, %r271, %r267;
mov.u32 %r259, 0;

	cvt.rn.f16.s32 %rs62, %r259;

	st.local.u16 [%rd5], %rs62;
setp.lt.s32	%p48, %r116, 1;
@%p48 bra BB7_101;

mad.lo.s32 %r82, %r79, %r116, %r9;
add.s32 %r275, %r79, %r118;
mad.lo.s32 %r83, %r275, %r116, %r9;
mov.u32 %r492, 0;
setp.eq.s32	%p49, %r77, 0;
@%p49 bra BB7_90;

setp.eq.s32	%p50, %r77, 1;
@%p50 bra BB7_87;

setp.eq.s32	%p51, %r77, 2;
@%p51 bra BB7_84;

mov.u32 %r492, 32;
setp.ge.s32	%p52, %r9, %r116;
@%p52 bra BB7_84;

add.s32 %r278, %r82, %r4;
mul.wide.s32 %rd80, %r278, 2;
add.s64 %rd81, %rd3, %rd80;
ld.global.u16 %rs64, [%rd81];
st.local.u16 [%rd5], %rs64;
add.s32 %r279, %r83, %r4;
mul.wide.s32 %rd82, %r279, 2;
add.s64 %rd83, %rd3, %rd82;
ld.global.u16 %rs65, [%rd83];
st.local.u16 [%rd5+12], %rs65;

BB7_84:
add.s32 %r280, %r492, %r9;
setp.ge.s32	%p53, %r280, %r116;
@%p53 bra BB7_86;

shr.u32 %r281, %r492, 5;
add.s32 %r282, %r82, %r492;
add.s32 %r283, %r282, %r4;
mul.wide.s32 %rd84, %r283, 2;
add.s64 %rd85, %rd3, %rd84;
ld.global.u16 %rs66, [%rd85];
mul.wide.u32 %rd86, %r281, 2;
add.s64 %rd87, %rd5, %rd86;
st.local.u16 [%rd87], %rs66;
add.s32 %r284, %r83, %r492;
add.s32 %r285, %r284, %r4;
mul.wide.s32 %rd88, %r285, 2;
add.s64 %rd89, %rd3, %rd88;
ld.global.u16 %rs67, [%rd89];
st.local.u16 [%rd87+12], %rs67;

BB7_86:
add.s32 %r492, %r492, 32;

BB7_87:
add.s32 %r286, %r492, %r9;
setp.ge.s32	%p54, %r286, %r116;
@%p54 bra BB7_89;

shr.s32 %r287, %r492, 31;
shr.u32 %r288, %r287, 27;
add.s32 %r289, %r492, %r288;
shr.s32 %r290, %r289, 5;
add.s32 %r291, %r82, %r492;
add.s32 %r292, %r291, %r4;
mul.wide.s32 %rd90, %r292, 2;
add.s64 %rd91, %rd3, %rd90;
ld.global.u16 %rs68, [%rd91];
mul.wide.s32 %rd92, %r290, 2;
add.s64 %rd93, %rd5, %rd92;
st.local.u16 [%rd93], %rs68;
add.s32 %r293, %r83, %r492;
add.s32 %r294, %r293, %r4;
mul.wide.s32 %rd94, %r294, 2;
add.s64 %rd95, %rd3, %rd94;
ld.global.u16 %rs69, [%rd95];
st.local.u16 [%rd93+12], %rs69;

BB7_89:
add.s32 %r492, %r492, 32;

BB7_90:
setp.lt.u32	%p55, %r75, 4;
@%p55 bra BB7_101;

add.s32 %r495, %r9, %r492;
mad.lo.s32 %r295, %r116, %r79, %r495;
mul.wide.s32 %rd96, %r295, 2;
add.s64 %rd131, %rd27, %rd96;
add.s32 %r296, %r118, %r79;
mad.lo.s32 %r297, %r116, %r296, %r495;
mul.wide.s32 %rd97, %r297, 2;
add.s64 %rd132, %rd27, %rd97;

BB7_92:
setp.ge.s32	%p56, %r495, %r116;
@%p56 bra BB7_94;

shr.s32 %r298, %r492, 31;
shr.u32 %r299, %r298, 27;
add.s32 %r300, %r492, %r299;
shr.s32 %r301, %r300, 5;
ld.global.u16 %rs70, [%rd131];
mul.wide.s32 %rd98, %r301, 2;
add.s64 %rd99, %rd5, %rd98;
st.local.u16 [%rd99], %rs70;
ld.global.u16 %rs71, [%rd132];
st.local.u16 [%rd99+12], %rs71;

BB7_94:
add.s32 %r302, %r495, 32;
setp.ge.s32	%p57, %r302, %r116;
@%p57 bra BB7_96;

add.s32 %r303, %r492, 32;
shr.s32 %r304, %r303, 31;
shr.u32 %r305, %r304, 27;
add.s32 %r306, %r303, %r305;
shr.s32 %r307, %r306, 5;
ld.global.u16 %rs72, [%rd131+64];
mul.wide.s32 %rd100, %r307, 2;
add.s64 %rd101, %rd5, %rd100;
st.local.u16 [%rd101], %rs72;
ld.global.u16 %rs73, [%rd132+64];
st.local.u16 [%rd101+12], %rs73;

BB7_96:
add.s32 %r308, %r495, 64;
setp.ge.s32	%p58, %r308, %r116;
@%p58 bra BB7_98;

add.s32 %r309, %r492, 64;
shr.s32 %r310, %r309, 31;
shr.u32 %r311, %r310, 27;
add.s32 %r312, %r309, %r311;
shr.s32 %r313, %r312, 5;
ld.global.u16 %rs74, [%rd131+128];
mul.wide.s32 %rd102, %r313, 2;
add.s64 %rd103, %rd5, %rd102;
st.local.u16 [%rd103], %rs74;
ld.global.u16 %rs75, [%rd132+128];
st.local.u16 [%rd103+12], %rs75;

BB7_98:
add.s32 %r314, %r495, 96;
setp.ge.s32	%p59, %r314, %r116;
@%p59 bra BB7_100;

add.s32 %r315, %r492, 96;
shr.s32 %r316, %r315, 31;
shr.u32 %r317, %r316, 27;
add.s32 %r318, %r315, %r317;
shr.s32 %r319, %r318, 5;
ld.global.u16 %rs76, [%rd131+192];
mul.wide.s32 %rd104, %r319, 2;
add.s64 %rd105, %rd5, %rd104;
st.local.u16 [%rd105], %rs76;
ld.global.u16 %rs77, [%rd132+192];
st.local.u16 [%rd105+12], %rs77;

BB7_100:
add.s32 %r492, %r492, 128;
add.s32 %r495, %r495, 128;
setp.lt.s32	%p60, %r492, %r73;
add.s64 %rd131, %rd131, 256;
add.s64 %rd132, %rd132, 256;
@%p60 bra BB7_92;

BB7_101:
setp.lt.s32	%p61, %r491, 1;
@%p61 bra BB7_113;

mul.lo.s32 %r94, %r118, %r80;
mov.u32 %r497, 0;

BB7_103:
add.s32 %r321, %r497, %r118;
shl.b32 %r322, %r321, 2;
add.s32 %r324, %r186, %r322;
ld.shared.u32 %r96, [%r324];
mov.u16 %rs239, %rs62;
@%p48 bra BB7_110;

shl.b32 %r326, %r96, 2;
add.s32 %r328, %r186, %r326;
ld.shared.u32 %r97, [%r328];
add.s32 %r329, %r94, %r96;
mad.lo.s32 %r330, %r116, %r329, %r9;
mul.wide.s32 %rd106, %r330, 2;
add.s64 %rd133, %rd24, %rd106;
mov.u32 %r499, 0;
mov.u32 %r498, %r9;
mov.u16 %rs239, %rs62;

BB7_105:
setp.ge.s32	%p63, %r498, %r116;
mov.u16 %rs237, %rs62;
mov.u16 %rs238, %rs62;
@%p63 bra BB7_109;

ld.global.u16 %rs238, [%rd133];
shr.s32 %r331, %r499, 31;
shr.u32 %r332, %r331, 27;
add.s32 %r333, %r499, %r332;
shr.s32 %r334, %r333, 5;
mul.wide.s32 %rd107, %r334, 2;
add.s64 %rd38, %rd5, %rd107;
setp.eq.s32	%p64, %r97, 0;
@%p64 bra BB7_108;
bra.uni BB7_107;

BB7_108:
ld.local.u16 %rs237, [%rd38];
bra.uni BB7_109;

BB7_107:
ld.local.u16 %rs237, [%rd38+12];

BB7_109:

	{mul.f16 %rs78,%rs238,%rs237;
}

	
	{add.f16 %rs239,%rs239,%rs78;
}

	add.s32 %r498, %r498, 32;
add.s64 %rd133, %rd133, 64;
add.s32 %r499, %r499, 32;
setp.lt.s32	%p65, %r499, %r73;
@%p65 bra BB7_105;

BB7_110:
bar.warp.sync -1;

	{ mov.b32 %r335, {%rs239,%rs239};}


	mov.u32 %r363, 8;
mov.u32 %r339, 1;
mov.u32 %r373, -1;

	{shfl.sync.bfly.b32 %r337,%r335,%r339,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r337;
mov.b16 %rs86, low;}

	
	{add.f16 %rs87,%rs239,%rs86;
}

	
	{ mov.b32 %r343, {%rs87,%rs87};}


	mov.u32 %r347, 2;

	{shfl.sync.bfly.b32 %r345,%r343,%r347,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r345;
mov.b16 %rs92, low;}

	
	{add.f16 %rs93,%rs87,%rs92;
}

	
	{ mov.b32 %r351, {%rs93,%rs93};}


	mov.u32 %r355, 4;

	{shfl.sync.bfly.b32 %r353,%r351,%r355,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r353;
mov.b16 %rs98, low;}

	
	{add.f16 %rs99,%rs93,%rs98;
}

	
	{ mov.b32 %r359, {%rs99,%rs99};}


	
	{shfl.sync.bfly.b32 %r361,%r359,%r363,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r361;
mov.b16 %rs104, low;}

	
	{add.f16 %rs105,%rs99,%rs104;
}

	
	{ mov.b32 %r367, {%rs105,%rs105};}


	mov.u32 %r371, 16;

	{shfl.sync.bfly.b32 %r369,%r367,%r371,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r369;
mov.b16 %rs110, low;}

	
	{add.f16 %rs111,%rs105,%rs110;
}

	setp.ne.s32	%p66, %r9, 0;
@%p66 bra BB7_112;

add.s32 %r385, %r96, %r81;
add.s32 %r386, %r385, %r2;
mul.wide.s32 %rd108, %r386, 2;
add.s64 %rd109, %rd4, %rd108;
st.global.u16 [%rd109], %rs111;

BB7_112:
bar.warp.sync -1;
add.s32 %r497, %r497, 1;
setp.lt.s32	%p67, %r497, %r491;
@%p67 bra BB7_103;

BB7_113:
@%p48 bra BB7_114;

mad.lo.s32 %r390, %r80, %r118, %r79;
mad.lo.s32 %r103, %r390, %r116, %r9;
mad.lo.s32 %r104, %r79, %r116, %r9;
setp.eq.s32	%p69, %r77, 0;
@%p69 bra BB7_116;

setp.eq.s32	%p70, %r77, 1;
@%p70 bra BB7_118;
bra.uni BB7_119;

BB7_118:
mov.u16 %rs245, %rs62;
bra.uni BB7_126;

BB7_114:
mov.u16 %rs259, %rs62;
bra.uni BB7_140;

BB7_116:
mov.u16 %rs248, %rs62;
bra.uni BB7_129;

BB7_119:
setp.eq.s32	%p71, %r77, 2;
mov.u16 %rs242, %rs62;
@%p71 bra BB7_123;

setp.ge.s32	%p72, %r9, %r116;
mov.u16 %rs240, %rs62;
mov.u16 %rs241, %rs62;
@%p72 bra BB7_122;

add.s32 %r391, %r103, %r4;
mul.wide.s32 %rd110, %r391, 2;
add.s64 %rd111, %rd3, %rd110;
ld.global.u16 %rs241, [%rd111];
add.s32 %r392, %r104, %r5;
mul.wide.s32 %rd112, %r392, 2;
add.s64 %rd113, %rd3, %rd112;
ld.global.u16 %rs240, [%rd113];

BB7_122:
mov.f64 %fd1, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs115, %fd1;}


	
	{mul.f16 %rs116,%rs241,%rs241;
}

	
	{sub.f16 %rs119,%rs116,%rs240;
}

	
	{mul.f16 %rs122,%rs115,%rs119;
}

	
	{add.f16 %rs242,%rs62,%rs122;
}

	mov.u32 %r259, 32;

BB7_123:
add.s32 %r394, %r259, %r9;
setp.ge.s32	%p73, %r394, %r116;
mov.u16 %rs243, %rs62;
mov.u16 %rs244, %rs62;
@%p73 bra BB7_125;

add.s32 %r395, %r103, %r259;
add.s32 %r396, %r395, %r4;
mul.wide.s32 %rd114, %r396, 2;
add.s64 %rd115, %rd3, %rd114;
ld.global.u16 %rs244, [%rd115];
add.s32 %r397, %r104, %r259;
add.s32 %r398, %r397, %r5;
mul.wide.s32 %rd116, %r398, 2;
add.s64 %rd117, %rd3, %rd116;
ld.global.u16 %rs243, [%rd117];

BB7_125:
mov.f64 %fd2, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs128, %fd2;}


	
	{mul.f16 %rs129,%rs244,%rs244;
}

	
	{sub.f16 %rs132,%rs129,%rs243;
}

	
	{mul.f16 %rs135,%rs128,%rs132;
}

	
	{add.f16 %rs245,%rs242,%rs135;
}

	add.s32 %r259, %r259, 32;

BB7_126:
add.s32 %r399, %r259, %r9;
setp.ge.s32	%p74, %r399, %r116;
mov.u16 %rs246, %rs62;
mov.u16 %rs247, %rs62;
@%p74 bra BB7_128;

add.s32 %r400, %r103, %r259;
add.s32 %r401, %r400, %r4;
mul.wide.s32 %rd118, %r401, 2;
add.s64 %rd119, %rd3, %rd118;
ld.global.u16 %rs247, [%rd119];
add.s32 %r402, %r104, %r259;
add.s32 %r403, %r402, %r5;
mul.wide.s32 %rd120, %r403, 2;
add.s64 %rd121, %rd3, %rd120;
ld.global.u16 %rs246, [%rd121];

BB7_128:
mov.f64 %fd3, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs141, %fd3;}


	
	{mul.f16 %rs142,%rs247,%rs247;
}

	
	{sub.f16 %rs145,%rs142,%rs246;
}

	
	{mul.f16 %rs148,%rs141,%rs145;
}

	
	{add.f16 %rs248,%rs245,%rs148;
}

	add.s32 %r259, %r259, 32;
mov.u16 %rs259, %rs248;

BB7_129:
setp.lt.u32	%p75, %r75, 4;
@%p75 bra BB7_140;

add.s32 %r503, %r9, %r259;
mad.lo.s32 %r404, %r118, %r80, %r79;
mad.lo.s32 %r405, %r116, %r404, %r503;
mul.wide.s32 %rd122, %r405, 2;
add.s64 %rd134, %rd27, %rd122;
mad.lo.s32 %r406, %r116, %r79, %r503;
mul.wide.s32 %rd123, %r406, 2;
add.s64 %rd135, %rd28, %rd123;
mov.u16 %rs259, %rs248;

BB7_131:
setp.ge.s32	%p76, %r503, %r116;
mov.u16 %rs251, %rs62;
mov.u16 %rs252, %rs62;
@%p76 bra BB7_133;

ld.global.u16 %rs252, [%rd134];
ld.global.u16 %rs251, [%rd135];

BB7_133:
mov.f64 %fd4, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs154, %fd4;}


	
	{mul.f16 %rs155,%rs252,%rs252;
}

	
	{sub.f16 %rs158,%rs155,%rs251;
}

	
	{mul.f16 %rs161,%rs154,%rs158;
}

	
	{add.f16 %rs164,%rs259,%rs161;
}

	add.s32 %r407, %r503, 32;
setp.ge.s32	%p77, %r407, %r116;
mov.u16 %rs253, %rs62;
mov.u16 %rs254, %rs62;
@%p77 bra BB7_135;

ld.global.u16 %rs254, [%rd134+64];
ld.global.u16 %rs253, [%rd135+64];

BB7_135:

	{ cvt.rn.f16.f64 %rs167, %fd4;}


	
	{mul.f16 %rs168,%rs254,%rs254;
}

	
	{sub.f16 %rs171,%rs168,%rs253;
}

	
	{mul.f16 %rs174,%rs167,%rs171;
}

	
	{add.f16 %rs177,%rs164,%rs174;
}

	add.s32 %r408, %r503, 64;
setp.ge.s32	%p78, %r408, %r116;
mov.u16 %rs255, %rs62;
mov.u16 %rs256, %rs62;
@%p78 bra BB7_137;

ld.global.u16 %rs256, [%rd134+128];
ld.global.u16 %rs255, [%rd135+128];

BB7_137:

	{ cvt.rn.f16.f64 %rs180, %fd4;}


	
	{mul.f16 %rs181,%rs256,%rs256;
}

	
	{sub.f16 %rs184,%rs181,%rs255;
}

	
	{mul.f16 %rs187,%rs180,%rs184;
}

	
	{add.f16 %rs190,%rs177,%rs187;
}

	add.s32 %r409, %r503, 96;
setp.ge.s32	%p79, %r409, %r116;
mov.u16 %rs257, %rs62;
mov.u16 %rs258, %rs62;
@%p79 bra BB7_139;

ld.global.u16 %rs258, [%rd134+192];
ld.global.u16 %rs257, [%rd135+192];

BB7_139:

	{ cvt.rn.f16.f64 %rs193, %fd4;}


	
	{mul.f16 %rs194,%rs258,%rs258;
}

	
	{sub.f16 %rs197,%rs194,%rs257;
}

	
	{mul.f16 %rs200,%rs193,%rs197;
}

	
	{add.f16 %rs259,%rs190,%rs200;
}

	add.s32 %r503, %r503, 128;
add.s32 %r259, %r259, 128;
setp.lt.s32	%p80, %r259, %r73;
add.s64 %rd134, %rd134, 256;
add.s64 %rd135, %rd135, 256;
@%p80 bra BB7_131;

BB7_140:
bar.warp.sync -1;

	{ mov.b32 %r410, {%rs259,%rs259};}


	mov.u32 %r438, 8;
mov.u32 %r414, 1;
mov.u32 %r448, -1;

	{shfl.sync.bfly.b32 %r412,%r410,%r414,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r412;
mov.b16 %rs208, low;}

	
	{add.f16 %rs209,%rs259,%rs208;
}

	
	{ mov.b32 %r418, {%rs209,%rs209};}


	mov.u32 %r422, 2;

	{shfl.sync.bfly.b32 %r420,%r418,%r422,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r420;
mov.b16 %rs214, low;}

	
	{add.f16 %rs215,%rs209,%rs214;
}

	
	{ mov.b32 %r426, {%rs215,%rs215};}


	mov.u32 %r430, 4;

	{shfl.sync.bfly.b32 %r428,%r426,%r430,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r428;
mov.b16 %rs220, low;}

	
	{add.f16 %rs221,%rs215,%rs220;
}

	
	{ mov.b32 %r434, {%rs221,%rs221};}


	
	{shfl.sync.bfly.b32 %r436,%r434,%r438,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r436;
mov.b16 %rs226, low;}

	
	{add.f16 %rs227,%rs221,%rs226;
}

	
	{ mov.b32 %r442, {%rs227,%rs227};}


	mov.u32 %r446, 16;

	{shfl.sync.bfly.b32 %r444,%r442,%r446,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r444;
mov.b16 %rs232, low;}

	
	{add.f16 %rs233,%rs227,%rs232;
}

	setp.ne.s32	%p81, %r9, 0;
@%p81 bra BB7_142;

add.s32 %r460, %r81, %r79;
add.s32 %r461, %r460, %r2;
mul.wide.s32 %rd124, %r461, 2;
add.s64 %rd125, %rd4, %rd124;
st.global.u16 [%rd125], %rs233;

BB7_142:
bar.warp.sync -1;
add.s32 %r491, %r491, 32;
setp.lt.s32	%p82, %r491, %r74;
@%p82 bra BB7_78;

BB7_143:
ret;
}



Fatbin elf code:
================
arch = sm_75
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Fatbin ptx code:
================
arch = sm_75
code version = [7,0]
producer = <unknown>
host = linux
compile_size = 64bit
compressed








.version 7.0
.target sm_75
.address_size 64


.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust6system6detail10sequential3seqE[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust6system3cpp3parE[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust8cuda_cub3parE[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_1E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_2E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_3E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_4E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_5E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_6E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_7E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_8E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders2_9E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust12placeholders3_10E[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust3seqE[1];
.global .align 1 .b8 _ZN90_INTERNAL_68_tmpxft_00005687_00000000_7_sparse_fipnn_shared_memory_kernel_cpp1_ii_7a104e8f6thrust6deviceE[1];
.extern .shared .align 4 .b8 _ZN8nvinfer119sparse_fipnn_shared9smem_poolE[];

.visible .entry _ZN3cub11EmptyKernelIvEEvv(

)
{



ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT_(
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_2,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_5
)
{
.reg .pred %p<22>;
.reg .b32 %r<64>;
.reg .b64 %rd<40>;


ld.param.u64 %rd10, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_0];
ld.param.u32 %r35, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_1];
ld.param.u32 %r36, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_2];
ld.param.u64 %rd11, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryIfEEvPKiiiPiiPT__param_3];
cvta.to.global.u64 %rd1, %rd10;
cvta.to.global.u64 %rd2, %rd11;
mov.u32 %r37, %ntid.x;
mov.u32 %r38, %ctaid.x;
mov.u32 %r39, %tid.x;
mad.lo.s32 %r1, %r37, %r38, %r39;
add.s32 %r40, %r36, 1;
setp.ge.s32	%p1, %r1, %r40;
@%p1 bra BB1_2;

mul.wide.s32 %rd12, %r1, 4;
add.s64 %rd13, %rd2, %rd12;
mov.u32 %r41, 0;
st.global.u32 [%rd13], %r41;

BB1_2:
bar.sync 0;
setp.ge.s32	%p2, %r1, %r35;
@%p2 bra BB1_34;

setp.gt.s32	%p3, %r1, 0;
@%p3 bra BB1_24;
bra.uni BB1_4;

BB1_24:
mul.wide.s32 %rd29, %r1, 4;
add.s64 %rd30, %rd1, %rd29;
ld.global.u32 %r23, [%rd30];
ld.global.u32 %r24, [%rd30+-4];
setp.le.s32	%p16, %r23, %r24;
@%p16 bra BB1_34;

sub.s32 %r25, %r23, %r24;
and.b32 %r26, %r25, 3;
setp.eq.s32	%p17, %r26, 0;
@%p17 bra BB1_31;

setp.eq.s32	%p18, %r26, 1;
@%p18 bra BB1_30;

setp.eq.s32	%p19, %r26, 2;
@%p19 bra BB1_29;

mul.wide.s32 %rd31, %r23, 4;
add.s64 %rd32, %rd2, %rd31;
st.global.u32 [%rd32], %r1;
add.s32 %r23, %r23, -1;

BB1_29:
mul.wide.s32 %rd33, %r23, 4;
add.s64 %rd34, %rd2, %rd33;
st.global.u32 [%rd34], %r1;
add.s32 %r23, %r23, -1;

BB1_30:
mul.wide.s32 %rd35, %r23, 4;
add.s64 %rd36, %rd2, %rd35;
st.global.u32 [%rd36], %r1;
add.s32 %r23, %r23, -1;

BB1_31:
setp.lt.u32	%p20, %r25, 4;
@%p20 bra BB1_34;

mul.wide.s32 %rd37, %r23, 4;
add.s64 %rd39, %rd2, %rd37;

BB1_33:
st.global.u32 [%rd39], %r1;
st.global.u32 [%rd39+-4], %r1;
st.global.u32 [%rd39+-8], %r1;
st.global.u32 [%rd39+-12], %r1;
add.s64 %rd39, %rd39, -16;
add.s32 %r23, %r23, -4;
setp.gt.s32	%p21, %r23, %r24;
@%p21 bra BB1_33;
bra.uni BB1_34;

BB1_4:
ld.global.u32 %r2, [%rd1];
setp.lt.s32	%p4, %r2, 0;
@%p4 bra BB1_14;

add.s32 %r3, %r2, 1;
and.b32 %r45, %r3, 3;
mov.u32 %r54, 0;
setp.eq.s32	%p5, %r45, 0;
@%p5 bra BB1_11;

setp.eq.s32	%p6, %r45, 1;
mov.u32 %r53, %r54;
@%p6 bra BB1_10;

setp.eq.s32	%p7, %r45, 2;
mov.u32 %r52, %r54;
@%p7 bra BB1_9;

mov.u32 %r47, 0;
st.global.u32 [%rd2], %r47;
mov.u32 %r52, 1;

BB1_9:
mul.wide.u32 %rd14, %r52, 4;
add.s64 %rd15, %rd2, %rd14;
st.global.u32 [%rd15], %r54;
add.s32 %r53, %r52, 1;

BB1_10:
mul.wide.s32 %rd16, %r53, 4;
add.s64 %rd17, %rd2, %rd16;
st.global.u32 [%rd17], %r54;
add.s32 %r54, %r53, 1;

BB1_11:
setp.lt.u32	%p8, %r3, 4;
@%p8 bra BB1_14;

add.s32 %r55, %r54, -1;
mul.wide.s32 %rd18, %r54, 4;
add.s64 %rd38, %rd2, %rd18;

BB1_13:
mov.u64 %rd19, 0;
st.global.u32 [%rd38+4], %rd19;
st.global.u32 [%rd38], %rd19;
st.global.u32 [%rd38+12], %rd19;
st.global.u32 [%rd38+8], %rd19;
add.s64 %rd38, %rd38, 16;
add.s32 %r55, %r55, 4;
setp.lt.s32	%p9, %r55, %r2;
@%p9 bra BB1_13;

BB1_14:
add.s32 %r50, %r35, -1;
mul.wide.s32 %rd20, %r50, 4;
add.s64 %rd21, %rd1, %rd20;
add.s32 %r56, %r36, -1;
ld.global.u32 %r13, [%rd21];
mul.wide.s32 %rd22, %r36, 4;
add.s64 %rd6, %rd2, %rd22;
setp.le.s32	%p10, %r56, %r13;
@%p10 bra BB1_23;

sub.s32 %r14, %r56, %r13;
and.b32 %r51, %r14, 3;
setp.eq.s32	%p11, %r51, 0;
@%p11 bra BB1_21;

setp.eq.s32	%p12, %r51, 1;
@%p12 bra BB1_20;

setp.eq.s32	%p13, %r51, 2;
@%p13 bra BB1_19;

st.global.u32 [%rd6+-4], %r35;
add.s32 %r56, %r36, -2;

BB1_19:
mul.wide.s32 %rd23, %r56, 4;
add.s64 %rd24, %rd2, %rd23;
st.global.u32 [%rd24], %r35;
add.s32 %r56, %r56, -1;

BB1_20:
mul.wide.s32 %rd25, %r56, 4;
add.s64 %rd26, %rd2, %rd25;
st.global.u32 [%rd26], %r35;
add.s32 %r56, %r56, -1;

BB1_21:
setp.lt.u32	%p14, %r14, 4;
@%p14 bra BB1_23;

BB1_22:
mul.wide.s32 %rd27, %r56, 4;
add.s64 %rd28, %rd2, %rd27;
st.global.u32 [%rd28], %r35;
st.global.u32 [%rd28+-4], %r35;
st.global.u32 [%rd28+-8], %r35;
st.global.u32 [%rd28+-12], %r35;
add.s32 %r56, %r56, -4;
setp.gt.s32	%p15, %r56, %r13;
@%p15 bra BB1_22;

BB1_23:
st.global.u32 [%rd6], %r35;

BB1_34:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_4,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_11
)
{
.reg .pred %p<76>;
.reg .f32 %f<81>;
.reg .b32 %r<464>;
.reg .b64 %rd<120>;


ld.param.u32 %r136, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_0];
ld.param.u32 %r137, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_1];
ld.param.u32 %r138, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_2];
ld.param.u32 %r139, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_3];
ld.param.u32 %r140, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_4];
ld.param.u32 %r141, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_5];
ld.param.u64 %rd20, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_6];
ld.param.u64 %rd21, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_7];
ld.param.u64 %rd17, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_8];
ld.param.u64 %rd18, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_9];
ld.param.u64 %rd19, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_10];
ld.param.u64 %rd22, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_11];
cvta.to.global.u64 %rd1, %rd21;
cvta.to.global.u64 %rd2, %rd22;
add.s32 %r1, %r139, 1;
mul.lo.s32 %r142, %r1, %r139;
shr.u32 %r143, %r142, 31;
add.s32 %r144, %r142, %r143;
shr.s32 %r2, %r144, 1;
cvt.s64.s32	%rd23, %r139;
add.s64 %rd3, %rd23, 1;
add.s64 %rd4, %rd3, %rd3;
mul.lo.s32 %r145, %r138, %r137;
mul.lo.s32 %r146, %r145, %r139;
cvt.s64.s32	%rd24, %r146;
add.s64 %rd5, %rd4, %rd24;
mul.lo.s32 %r147, %r139, %r137;
cvt.s64.s32	%rd25, %r147;
add.s64 %rd6, %rd5, %rd25;
mov.u32 %r3, %ntid.x;
mov.u32 %r453, %tid.y;
mov.u32 %r5, %tid.x;
mad.lo.s32 %r459, %r3, %r453, %r5;
mov.u32 %r7, %ntid.y;
mul.lo.s32 %r8, %r3, %r7;
mov.u32 %r9, %ctaid.y;
div.u32 %r10, %r137, %r3;
cvta.to.global.u64 %rd26, %rd20;
ld.global.u32 %r11, [%rd26];
setp.ge.s32	%p1, %r459, %r140;
@%p1 bra BB2_5;

mov.u32 %r414, %r459;

BB2_2:
cvt.u64.u32	%rd27, %r414;
add.s64 %rd28, %rd27, %rd4;
cvt.u32.u64	%r148, %rd28;
shl.b32 %r149, %r148, 2;
mov.u32 %r150, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r151, %r150, %r149;
mov.u32 %r152, 0;
st.shared.u32 [%r151], %r152;
setp.ge.s32	%p2, %r414, %r139;
@%p2 bra BB2_4;

shl.b32 %r153, %r414, 2;
add.s32 %r155, %r150, %r153;
mov.u32 %r156, -1;
st.shared.u32 [%r155], %r156;

BB2_4:
add.s32 %r414, %r414, %r8;
setp.lt.s32	%p3, %r414, %r140;
@%p3 bra BB2_2;

BB2_5:
setp.ge.s32	%p4, %r459, %r2;
@%p4 bra BB2_8;

mov.u32 %r415, %r459;

BB2_7:
cvt.u64.u32	%rd29, %r415;
add.s64 %rd30, %rd29, %rd6;
cvt.u32.u64	%r157, %rd30;
shl.b32 %r158, %r157, 2;
mov.u32 %r159, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r160, %r159, %r158;
mov.u32 %r161, 0;
st.shared.u32 [%r160], %r161;
add.s32 %r415, %r415, %r8;
setp.lt.s32	%p5, %r415, %r2;
@%p5 bra BB2_7;

BB2_8:
bar.sync 0;
setp.ge.s32	%p6, %r453, %r11;
@%p6 bra BB2_27;

cvta.to.global.u64 %rd7, %rd17;
cvt.u64.u32	%rd31, %r5;
add.s64 %rd8, %rd31, %rd4;
add.s64 %rd9, %rd31, %rd5;
mad.lo.s32 %r16, %r9, %r137, %r5;
mov.u32 %r416, %r453;

BB2_10:
shl.b32 %r162, %r416, 1;
mul.wide.s32 %rd32, %r162, 4;
add.s64 %rd33, %rd7, %rd32;
ld.global.u32 %r163, [%rd33+4];
add.s32 %r18, %r163, -1;
setp.lt.s32	%p7, %r18, 0;
setp.ge.s32	%p8, %r18, %r139;
or.pred %p9, %p7, %p8;
ld.global.u32 %r19, [%rd33];
add.s32 %r164, %r19, -1;
setp.lt.s32	%p10, %r164, 0;
or.pred %p11, %p9, %p10;
setp.ge.s32	%p12, %r164, %r138;
or.pred %p13, %p11, %p12;
@%p13 bra BB2_26;

setp.ne.s32	%p14, %r5, 0;
@%p14 bra BB2_13;

shl.b32 %r165, %r18, 2;
mov.u32 %r166, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r167, %r166, %r165;
add.s32 %r407, %r19, -1;
st.shared.u32 [%r167], %r407;

BB2_13:
setp.lt.s32	%p15, %r138, 1;
@%p15 bra BB2_21;

mul.lo.s32 %r20, %r138, %r416;
mov.u32 %r417, 0;

BB2_15:
setp.lt.s32	%p16, %r10, 1;
@%p16 bra BB2_20;

mad.lo.s32 %r172, %r417, %r139, %r18;
mul.lo.s32 %r173, %r172, %r137;
cvt.u64.u32	%rd34, %r173;
add.s64 %rd10, %rd8, %rd34;
add.s32 %r174, %r20, %r417;
mad.lo.s32 %r418, %r136, %r174, %r16;
mov.u32 %r420, 0;
mov.u32 %r419, %r16;
mov.u32 %r421, %r420;

BB2_17:
setp.ge.u32	%p17, %r419, %r136;
@%p17 bra BB2_19;

cvt.u64.u32	%rd35, %r420;
add.s64 %rd36, %rd10, %rd35;
cvt.u32.u64	%r175, %rd36;
shl.b32 %r176, %r175, 2;
mov.u32 %r177, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r178, %r177, %r176;
mul.wide.s32 %rd37, %r418, 4;
add.s64 %rd38, %rd1, %rd37;
ld.global.f32 %f17, [%rd38];
atom.shared.add.f32 %f18, [%r178], %f17;

BB2_19:
add.s32 %r421, %r421, 1;
add.s32 %r420, %r420, %r3;
add.s32 %r419, %r419, %r3;
add.s32 %r418, %r418, %r3;
setp.lt.s32	%p18, %r421, %r10;
@%p18 bra BB2_17;

BB2_20:
add.s32 %r417, %r417, 1;
setp.lt.s32	%p19, %r417, %r138;
@%p19 bra BB2_15;

BB2_21:
setp.lt.s32	%p20, %r10, 1;
@%p20 bra BB2_26;

mul.lo.s32 %r181, %r18, %r137;
cvt.u64.u32	%rd39, %r181;
add.s64 %rd11, %rd9, %rd39;
mad.lo.s32 %r182, %r138, %r416, %r19;
add.s32 %r183, %r182, -1;
mad.lo.s32 %r422, %r136, %r183, %r16;
mov.u32 %r424, 0;
mov.u32 %r423, %r16;
mov.u32 %r425, %r424;

BB2_23:
setp.ge.u32	%p21, %r423, %r136;
@%p21 bra BB2_25;

cvt.u64.u32	%rd40, %r424;
add.s64 %rd41, %rd11, %rd40;
cvt.u32.u64	%r184, %rd41;
shl.b32 %r185, %r184, 2;
mov.u32 %r186, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r187, %r186, %r185;
mul.wide.s32 %rd42, %r422, 4;
add.s64 %rd43, %rd1, %rd42;
ld.global.f32 %f19, [%rd43];
mul.f32 %f20, %f19, %f19;
atom.shared.add.f32 %f21, [%r187], %f20;

BB2_25:
add.s32 %r425, %r425, 1;
add.s32 %r424, %r424, %r3;
add.s32 %r423, %r423, %r3;
add.s32 %r422, %r422, %r3;
setp.lt.s32	%p22, %r425, %r10;
@%p22 bra BB2_23;

BB2_26:
add.s32 %r416, %r416, %r7;
setp.lt.s32	%p23, %r416, %r11;
@%p23 bra BB2_10;

BB2_27:
bar.sync 0;
or.b32 %r188, %r5, %r453;
cvt.u64.u32	%rd44, %r139;
add.s64 %rd45, %rd3, %rd44;
cvt.u32.u64	%r189, %rd45;
shl.b32 %r190, %r189, 2;
mov.u32 %r191, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r42, %r191, %r190;
setp.ne.s32	%p24, %r188, 0;
@%p24 bra BB2_61;

mov.u32 %r436, 0;
setp.lt.s32	%p25, %r139, 1;
@%p25 bra BB2_60;

and.b32 %r199, %r139, 3;
mov.u32 %r427, 0;
setp.eq.s32	%p26, %r199, 0;
@%p26 bra BB2_30;

setp.eq.s32	%p27, %r199, 1;
@%p27 bra BB2_32;
bra.uni BB2_33;

BB2_32:
mov.u32 %r430, %r427;
bra.uni BB2_41;

BB2_30:
mov.u32 %r436, %r427;
bra.uni BB2_45;

BB2_33:
setp.eq.s32	%p28, %r199, 2;
@%p28 bra BB2_34;
bra.uni BB2_35;

BB2_34:
mov.u32 %r426, %r427;
bra.uni BB2_37;

BB2_35:
ld.shared.u32 %r202, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r426, 1;
setp.lt.s32	%p29, %r202, 0;
@%p29 bra BB2_37;

shl.b32 %r205, %r1, 2;
add.s32 %r207, %r191, %r205;
mov.u32 %r208, 0;
st.shared.u32 [%r207], %r208;
mov.u32 %r426, 1;
mov.u32 %r427, %r426;

BB2_37:
shl.b32 %r209, %r426, 2;
add.s32 %r211, %r191, %r209;
ld.shared.u32 %r212, [%r211];
setp.lt.s32	%p30, %r212, 0;
@%p30 bra BB2_38;

add.s32 %r430, %r427, 1;
cvt.u64.u32	%rd46, %r427;
add.s64 %rd47, %rd46, %rd3;
cvt.u32.u64	%r213, %rd47;
shl.b32 %r214, %r213, 2;
add.s32 %r216, %r191, %r214;
st.shared.u32 [%r216], %r426;
bra.uni BB2_40;

BB2_38:
mov.u32 %r430, %r427;

BB2_40:
add.s32 %r427, %r426, 1;

BB2_41:
shl.b32 %r217, %r427, 2;
add.s32 %r219, %r191, %r217;
ld.shared.u32 %r220, [%r219];
setp.lt.s32	%p31, %r220, 0;
@%p31 bra BB2_42;

add.s32 %r436, %r430, 1;
cvt.u64.u32	%rd48, %r430;
add.s64 %rd49, %rd48, %rd3;
cvt.u32.u64	%r221, %rd49;
shl.b32 %r222, %r221, 2;
add.s32 %r224, %r191, %r222;
st.shared.u32 [%r224], %r427;
bra.uni BB2_44;

BB2_42:
mov.u32 %r436, %r430;

BB2_44:
add.s32 %r427, %r427, 1;

BB2_45:
setp.lt.u32	%p32, %r139, 4;
@%p32 bra BB2_60;

shl.b32 %r225, %r427, 2;
add.s32 %r434, %r191, %r225;

BB2_47:
ld.shared.u32 %r227, [%r434];
setp.lt.s32	%p33, %r227, 0;
@%p33 bra BB2_48;

add.s32 %r437, %r436, 1;
cvt.u64.u32	%rd50, %r436;
add.s64 %rd51, %rd50, %rd3;
cvt.u32.u64	%r228, %rd51;
shl.b32 %r229, %r228, 2;
add.s32 %r231, %r191, %r229;
st.shared.u32 [%r231], %r427;
bra.uni BB2_50;

BB2_48:
mov.u32 %r437, %r436;

BB2_50:
ld.shared.u32 %r232, [%r434+4];
setp.lt.s32	%p34, %r232, 0;
@%p34 bra BB2_51;

add.s32 %r438, %r437, 1;
cvt.u64.u32	%rd52, %r437;
add.s64 %rd53, %rd52, %rd3;
cvt.u32.u64	%r233, %rd53;
shl.b32 %r234, %r233, 2;
add.s32 %r236, %r191, %r234;
add.s32 %r237, %r427, 1;
st.shared.u32 [%r236], %r237;
bra.uni BB2_53;

BB2_51:
mov.u32 %r438, %r437;

BB2_53:
ld.shared.u32 %r238, [%r434+8];
setp.lt.s32	%p35, %r238, 0;
@%p35 bra BB2_54;

add.s32 %r439, %r438, 1;
cvt.u64.u32	%rd54, %r438;
add.s64 %rd55, %rd54, %rd3;
cvt.u32.u64	%r239, %rd55;
shl.b32 %r240, %r239, 2;
add.s32 %r242, %r191, %r240;
add.s32 %r243, %r427, 2;
st.shared.u32 [%r242], %r243;
bra.uni BB2_56;

BB2_54:
mov.u32 %r439, %r438;

BB2_56:
ld.shared.u32 %r244, [%r434+12];
setp.lt.s32	%p36, %r244, 0;
@%p36 bra BB2_57;

add.s32 %r436, %r439, 1;
cvt.u64.u32	%rd56, %r439;
add.s64 %rd57, %rd56, %rd3;
cvt.u32.u64	%r245, %rd57;
shl.b32 %r246, %r245, 2;
add.s32 %r248, %r191, %r246;
add.s32 %r249, %r427, 3;
st.shared.u32 [%r248], %r249;
bra.uni BB2_59;

BB2_57:
mov.u32 %r436, %r439;

BB2_59:
add.s32 %r427, %r427, 4;
setp.lt.s32	%p37, %r427, %r139;
add.s32 %r434, %r434, 16;
@%p37 bra BB2_47;

BB2_60:
st.shared.u32 [%r42], %r436;

BB2_61:
bar.sync 0;
ld.shared.u32 %r70, [%r42];
setp.ge.s32	%p38, %r453, %r70;
@%p38 bra BB2_81;

mad.lo.s32 %r71, %r9, %r137, %r5;
mov.u32 %r442, %r453;

BB2_63:
cvt.u64.u32	%rd58, %r442;
add.s64 %rd59, %rd58, %rd3;
cvt.u32.u64	%r250, %rd59;
shl.b32 %r251, %r250, 2;
add.s32 %r253, %r191, %r251;
ld.shared.u32 %r73, [%r253];
shl.b32 %r254, %r73, 2;
add.s32 %r255, %r191, %r254;
ld.shared.u32 %r74, [%r255];
add.s32 %r256, %r73, 1;
mul.lo.s32 %r257, %r256, %r73;
shr.u32 %r258, %r257, 31;
add.s32 %r259, %r257, %r258;
shr.s32 %r75, %r259, 1;
mov.f32 %f76, 0f00000000;
setp.lt.s32	%p39, %r10, 1;
@%p39 bra BB2_68;

mad.lo.s32 %r445, %r137, %r73, %r5;
mad.lo.s32 %r261, %r139, %r74, %r73;
mad.lo.s32 %r444, %r137, %r261, %r5;
mov.f32 %f23, 0f00000000;
mov.u32 %r446, 0;
mov.u32 %r443, %r71;
mov.f32 %f76, %f23;

BB2_65:
setp.ge.u32	%p40, %r443, %r136;
mov.f32 %f74, %f23;
mov.f32 %f75, %f23;
@%p40 bra BB2_67;

cvt.u64.u32	%rd60, %r444;
add.s64 %rd61, %rd60, %rd4;
cvt.u32.u64	%r262, %rd61;
shl.b32 %r263, %r262, 2;
add.s32 %r265, %r191, %r263;
ld.shared.f32 %f74, [%r265];
cvt.u64.u32	%rd62, %r445;
add.s64 %rd63, %rd62, %rd5;
cvt.u32.u64	%r266, %rd63;
shl.b32 %r267, %r266, 2;
add.s32 %r268, %r191, %r267;
ld.shared.f32 %f75, [%r268];

BB2_67:
mul.f32 %f26, %f74, %f74;
sub.f32 %f27, %f26, %f75;
fma.rn.f32 %f76, %f27, 0f3F000000, %f76;
add.s32 %r445, %r445, %r3;
add.s32 %r444, %r444, %r3;
add.s32 %r443, %r443, %r3;
add.s32 %r446, %r446, 1;
setp.lt.s32	%p41, %r446, %r10;
@%p41 bra BB2_65;

BB2_68:
bar.warp.sync -1;
mov.b32 %r269, %f76;
mov.u32 %r270, 31;
mov.u32 %r271, 1;
mov.u32 %r272, -1;
shfl.sync.bfly.b32 %r273|%p42, %r269, %r271, %r270, %r272;
mov.b32 %f28, %r273;
add.f32 %f29, %f76, %f28;
mov.b32 %r274, %f29;
mov.u32 %r275, 2;
shfl.sync.bfly.b32 %r276|%p43, %r274, %r275, %r270, %r272;
mov.b32 %f30, %r276;
add.f32 %f31, %f29, %f30;
mov.b32 %r277, %f31;
mov.u32 %r278, 4;
shfl.sync.bfly.b32 %r279|%p44, %r277, %r278, %r270, %r272;
mov.b32 %f32, %r279;
add.f32 %f33, %f31, %f32;
mov.b32 %r280, %f33;
mov.u32 %r281, 8;
shfl.sync.bfly.b32 %r282|%p45, %r280, %r281, %r270, %r272;
mov.b32 %f34, %r282;
add.f32 %f35, %f33, %f34;
mov.b32 %r283, %f35;
mov.u32 %r284, 16;
shfl.sync.bfly.b32 %r285|%p46, %r283, %r284, %r270, %r272;
mov.b32 %f36, %r285;
add.f32 %f8, %f35, %f36;
setp.ne.s32	%p47, %r5, 0;
@%p47 bra BB2_70;

add.s32 %r286, %r75, %r73;
cvt.u64.u32	%rd64, %r286;
add.s64 %rd65, %rd64, %rd6;
cvt.u32.u64	%r287, %rd65;
shl.b32 %r288, %r287, 2;
add.s32 %r290, %r191, %r288;
st.shared.f32 [%r290], %f8;

BB2_70:
bar.warp.sync -1;
setp.lt.s32	%p48, %r442, 1;
@%p48 bra BB2_80;

mul.lo.s32 %r86, %r139, %r74;
mov.u32 %r447, 0;

BB2_72:
cvt.u64.u32	%rd66, %r447;
add.s64 %rd67, %rd66, %rd3;
cvt.u32.u64	%r292, %rd67;
shl.b32 %r293, %r292, 2;
add.s32 %r295, %r191, %r293;
ld.shared.u32 %r88, [%r295];
mov.f32 %f80, 0f00000000;
@%p39 bra BB2_77;

shl.b32 %r297, %r88, 2;
add.s32 %r299, %r191, %r297;
ld.shared.u32 %r300, [%r299];
mad.lo.s32 %r301, %r139, %r300, %r73;
mad.lo.s32 %r450, %r137, %r301, %r5;
add.s32 %r302, %r86, %r88;
mad.lo.s32 %r449, %r137, %r302, %r5;
mov.f32 %f38, 0f00000000;
mov.u32 %r451, 0;
mov.u32 %r448, %r71;
mov.f32 %f80, %f38;

BB2_74:
setp.ge.u32	%p50, %r448, %r136;
mov.f32 %f78, %f38;
mov.f32 %f79, %f38;
@%p50 bra BB2_76;

cvt.u64.u32	%rd68, %r449;
add.s64 %rd69, %rd68, %rd4;
cvt.u32.u64	%r303, %rd69;
shl.b32 %r304, %r303, 2;
add.s32 %r306, %r191, %r304;
ld.shared.f32 %f78, [%r306];
cvt.u64.u32	%rd70, %r450;
add.s64 %rd71, %rd70, %rd4;
cvt.u32.u64	%r307, %rd71;
shl.b32 %r308, %r307, 2;
add.s32 %r309, %r191, %r308;
ld.shared.f32 %f79, [%r309];

BB2_76:
fma.rn.f32 %f80, %f78, %f79, %f80;
add.s32 %r450, %r450, %r3;
add.s32 %r449, %r449, %r3;
add.s32 %r448, %r448, %r3;
add.s32 %r451, %r451, 1;
setp.lt.s32	%p51, %r451, %r10;
@%p51 bra BB2_74;

BB2_77:
bar.warp.sync -1;
mov.u32 %r413, 2;
mov.u32 %r412, 31;
mov.u32 %r411, 1;
mov.u32 %r410, -1;
mov.b32 %r310, %f80;
shfl.sync.bfly.b32 %r314|%p52, %r310, %r411, %r412, %r410;
mov.b32 %f41, %r314;
add.f32 %f42, %f80, %f41;
mov.b32 %r315, %f42;
shfl.sync.bfly.b32 %r317|%p53, %r315, %r413, %r412, %r410;
mov.b32 %f43, %r317;
add.f32 %f44, %f42, %f43;
mov.b32 %r318, %f44;
shfl.sync.bfly.b32 %r320|%p54, %r318, %r278, %r412, %r410;
mov.b32 %f45, %r320;
add.f32 %f46, %f44, %f45;
mov.b32 %r321, %f46;
shfl.sync.bfly.b32 %r323|%p55, %r321, %r281, %r412, %r410;
mov.b32 %f47, %r323;
add.f32 %f48, %f46, %f47;
mov.b32 %r324, %f48;
shfl.sync.bfly.b32 %r326|%p56, %r324, %r284, %r412, %r410;
mov.b32 %f49, %r326;
add.f32 %f16, %f48, %f49;
@%p47 bra BB2_79;

add.s32 %r327, %r88, %r75;
cvt.u64.u32	%rd72, %r327;
add.s64 %rd73, %rd72, %rd6;
cvt.u32.u64	%r328, %rd73;
shl.b32 %r329, %r328, 2;
add.s32 %r331, %r191, %r329;
st.shared.f32 [%r331], %f16;

BB2_79:
bar.warp.sync -1;
add.s32 %r447, %r447, 1;
setp.lt.s32	%p58, %r447, %r442;
@%p58 bra BB2_72;

BB2_80:
add.s32 %r442, %r442, %r7;
setp.lt.s32	%p59, %r442, %r70;
@%p59 bra BB2_63;

BB2_81:
cvta.to.global.u64 %rd12, %rd19;
bar.sync 0;
setp.ge.s32	%p60, %r459, %r1;
@%p60 bra BB2_84;

mov.u32 %r452, %r459;

BB2_83:
shl.b32 %r332, %r452, 2;
add.s32 %r334, %r191, %r332;
ld.shared.u32 %r335, [%r334];
mul.wide.s32 %rd74, %r452, 4;
add.s64 %rd75, %rd12, %rd74;
st.global.u32 [%rd75], %r335;
add.s32 %r336, %r452, %r139;
shl.b32 %r337, %r336, 2;
add.s32 %r338, %r191, %r337;
ld.shared.u32 %r339, [%r338+4];
mul.wide.s32 %rd76, %r336, 4;
add.s64 %rd77, %rd12, %rd76;
st.global.u32 [%rd77+4], %r339;
add.s32 %r452, %r452, %r8;
setp.lt.s32	%p61, %r452, %r1;
@%p61 bra BB2_83;

BB2_84:
ld.param.u32 %r408, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputIfEEviiiiiiPiPKT_PKiPS3_S2_S8__param_2];
add.s32 %r340, %r408, 1;
mul.lo.s32 %r103, %r340, %r139;
setp.ge.s32	%p62, %r453, %r103;
@%p62 bra BB2_92;

mov.u32 %r409, %ctaid.y;
mad.lo.s32 %r104, %r409, %r137, %r5;
cvta.to.global.u64 %rd13, %rd18;

BB2_86:
setp.lt.s32	%p63, %r10, 1;
@%p63 bra BB2_91;

mad.lo.s32 %r456, %r137, %r453, %r5;
mad.lo.s32 %r455, %r136, %r453, %r104;
mov.u32 %r457, 0;
mov.u32 %r454, %r104;

BB2_88:
setp.ge.s32	%p64, %r454, %r136;
@%p64 bra BB2_90;

cvt.u64.u32	%rd78, %r456;
add.s64 %rd79, %rd78, %rd4;
cvt.u32.u64	%r342, %rd79;
shl.b32 %r343, %r342, 2;
add.s32 %r345, %r191, %r343;
ld.shared.f32 %f50, [%r345];
mul.wide.s32 %rd80, %r455, 4;
add.s64 %rd81, %rd13, %rd80;
st.global.f32 [%rd81], %f50;

BB2_90:
add.s32 %r457, %r457, 1;
add.s32 %r456, %r456, %r3;
add.s32 %r455, %r455, %r3;
add.s32 %r454, %r454, %r3;
setp.lt.s32	%p65, %r457, %r10;
@%p65 bra BB2_88;

BB2_91:
add.s32 %r453, %r453, %r7;
setp.lt.s32	%p66, %r453, %r103;
@%p66 bra BB2_86;

BB2_92:
add.s32 %r346, %r141, 3;
shr.s32 %r347, %r346, 31;
shr.u32 %r348, %r347, 30;
add.s32 %r349, %r346, %r348;
shr.s32 %r117, %r349, 2;
setp.ge.s32	%p67, %r459, %r117;
@%p67 bra BB2_107;

shl.b32 %r351, %r8, 2;
neg.s32 %r118, %r351;
shl.b32 %r352, %r459, 2;
sub.s32 %r119, %r141, %r352;
mov.u32 %r458, 0;

BB2_94:
mad.lo.s32 %r122, %r118, %r458, %r119;
shl.b32 %r123, %r459, 2;
add.s32 %r124, %r123, 3;
setp.lt.s32	%p68, %r124, %r141;
@%p68 bra BB2_105;
bra.uni BB2_95;

BB2_105:
mul.wide.s32 %rd106, %r123, 4;
rem.s32 %r388, %r123, %r2;
cvt.u64.u32	%rd107, %r388;
add.s64 %rd108, %rd107, %rd6;
cvt.u32.u64	%r389, %rd108;
shl.b32 %r390, %r389, 2;
add.s32 %r392, %r191, %r390;
ld.shared.f32 %f65, [%r392];
add.s64 %rd109, %rd2, %rd106;
atom.global.add.f32 %f66, [%rd109], %f65;
add.s64 %rd110, %rd109, 4;
add.s32 %r393, %r123, 1;
rem.s32 %r394, %r393, %r2;
cvt.u64.u32	%rd111, %r394;
add.s64 %rd112, %rd111, %rd6;
cvt.u32.u64	%r395, %rd112;
shl.b32 %r396, %r395, 2;
add.s32 %r397, %r191, %r396;
ld.shared.f32 %f67, [%r397];
atom.global.add.f32 %f68, [%rd110], %f67;
add.s64 %rd113, %rd109, 8;
add.s32 %r398, %r123, 2;
rem.s32 %r399, %r398, %r2;
cvt.u64.u32	%rd114, %r399;
add.s64 %rd115, %rd114, %rd6;
cvt.u32.u64	%r400, %rd115;
shl.b32 %r401, %r400, 2;
add.s32 %r402, %r191, %r401;
ld.shared.f32 %f69, [%r402];
atom.global.add.f32 %f70, [%rd113], %f69;
add.s64 %rd116, %rd109, 12;
rem.s32 %r403, %r124, %r2;
cvt.u64.u32	%rd117, %r403;
add.s64 %rd118, %rd117, %rd6;
cvt.u32.u64	%r404, %rd118;
shl.b32 %r405, %r404, 2;
add.s32 %r406, %r191, %r405;
ld.shared.f32 %f71, [%r406];
atom.global.add.f32 %f72, [%rd116], %f71;
bra.uni BB2_106;

BB2_95:
setp.ge.s32	%p69, %r123, %r141;
@%p69 bra BB2_106;

and.b32 %r125, %r122, 3;
setp.eq.s32	%p70, %r125, 0;
@%p70 bra BB2_102;

setp.eq.s32	%p71, %r125, 1;
@%p71 bra BB2_101;

setp.eq.s32	%p72, %r125, 2;
@%p72 bra BB2_100;

mul.wide.s32 %rd82, %r123, 4;
add.s64 %rd83, %rd2, %rd82;
rem.s32 %r353, %r123, %r2;
cvt.u64.u32	%rd84, %r353;
add.s64 %rd85, %rd84, %rd6;
cvt.u32.u64	%r354, %rd85;
shl.b32 %r355, %r354, 2;
add.s32 %r357, %r191, %r355;
ld.shared.f32 %f51, [%r357];
atom.global.add.f32 %f52, [%rd83], %f51;
add.s32 %r123, %r123, 1;

BB2_100:
mul.wide.s32 %rd86, %r123, 4;
add.s64 %rd87, %rd2, %rd86;
rem.s32 %r358, %r123, %r2;
cvt.u64.u32	%rd88, %r358;
add.s64 %rd89, %rd88, %rd6;
cvt.u32.u64	%r359, %rd89;
shl.b32 %r360, %r359, 2;
add.s32 %r362, %r191, %r360;
ld.shared.f32 %f53, [%r362];
atom.global.add.f32 %f54, [%rd87], %f53;
add.s32 %r123, %r123, 1;

BB2_101:
mul.wide.s32 %rd90, %r123, 4;
add.s64 %rd91, %rd2, %rd90;
rem.s32 %r363, %r123, %r2;
cvt.u64.u32	%rd92, %r363;
add.s64 %rd93, %rd92, %rd6;
cvt.u32.u64	%r364, %rd93;
shl.b32 %r365, %r364, 2;
add.s32 %r367, %r191, %r365;
ld.shared.f32 %f55, [%r367];
atom.global.add.f32 %f56, [%rd91], %f55;
add.s32 %r123, %r123, 1;

BB2_102:
setp.lt.u32	%p73, %r122, 4;
@%p73 bra BB2_106;

mul.wide.s32 %rd94, %r123, 4;
add.s64 %rd119, %rd2, %rd94;

BB2_104:
rem.s32 %r368, %r123, %r2;
cvt.u64.u32	%rd95, %r368;
add.s64 %rd96, %rd95, %rd6;
cvt.u32.u64	%r369, %rd96;
shl.b32 %r370, %r369, 2;
add.s32 %r372, %r191, %r370;
ld.shared.f32 %f57, [%r372];
atom.global.add.f32 %f58, [%rd119], %f57;
add.s32 %r373, %r123, 1;
rem.s32 %r374, %r373, %r2;
cvt.u64.u32	%rd97, %r374;
add.s64 %rd98, %rd97, %rd6;
cvt.u32.u64	%r375, %rd98;
shl.b32 %r376, %r375, 2;
add.s32 %r377, %r191, %r376;
ld.shared.f32 %f59, [%r377];
add.s64 %rd99, %rd119, 4;
atom.global.add.f32 %f60, [%rd99], %f59;
add.s32 %r378, %r123, 2;
rem.s32 %r379, %r378, %r2;
cvt.u64.u32	%rd100, %r379;
add.s64 %rd101, %rd100, %rd6;
cvt.u32.u64	%r380, %rd101;
shl.b32 %r381, %r380, 2;
add.s32 %r382, %r191, %r381;
ld.shared.f32 %f61, [%r382];
add.s64 %rd102, %rd119, 8;
atom.global.add.f32 %f62, [%rd102], %f61;
add.s32 %r383, %r123, 3;
rem.s32 %r384, %r383, %r2;
cvt.u64.u32	%rd103, %r384;
add.s64 %rd104, %rd103, %rd6;
cvt.u32.u64	%r385, %rd104;
shl.b32 %r386, %r385, 2;
add.s32 %r387, %r191, %r386;
ld.shared.f32 %f63, [%r387];
add.s64 %rd105, %rd119, 12;
atom.global.add.f32 %f64, [%rd105], %f63;
add.s64 %rd119, %rd119, 16;
add.s32 %r123, %r123, 4;
setp.lt.s32	%p74, %r123, %r141;
@%p74 bra BB2_104;

BB2_106:
add.s32 %r459, %r459, %r8;
setp.lt.s32	%p75, %r459, %r117;
add.s32 %r458, %r458, 1;
@%p75 bra BB2_94;

BB2_107:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_11
)
{
.reg .pred %p<76>;
.reg .f32 %f<87>;
.reg .b32 %r<458>;
.reg .b64 %rd<54>;


ld.param.u32 %r152, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_0];
ld.param.u32 %r153, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_1];
ld.param.u32 %r154, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_2];
ld.param.u32 %r155, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_3];
ld.param.u64 %rd13, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_4];
ld.param.u64 %rd11, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_5];
ld.param.u64 %rd14, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_7];
ld.param.u64 %rd15, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_8];
ld.param.u64 %rd12, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_9];
ld.param.u64 %rd16, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2IfEEviiiiPiPT_S4_S2_PKS3_PKiS4_S4__param_10];
cvta.to.global.u64 %rd1, %rd14;
cvta.to.global.u64 %rd2, %rd15;
cvta.to.global.u64 %rd3, %rd16;
add.s32 %r1, %r155, 1;
mul.lo.s32 %r156, %r1, %r155;
shr.u32 %r157, %r156, 31;
add.s32 %r158, %r156, %r157;
shr.s32 %r159, %r158, 1;
shl.b32 %r2, %r1, 1;
mov.u32 %r160, %ctaid.x;
mul.lo.s32 %r161, %r159, %r160;
cvt.s64.s32	%rd4, %r161;
cvta.to.global.u64 %rd17, %rd13;
mul.wide.s32 %rd18, %r160, 4;
add.s64 %rd19, %rd17, %rd18;
ld.global.u32 %r3, [%rd19];
ld.global.u32 %r4, [%rd19+4];
mov.u32 %r5, %ntid.x;
mov.u32 %r451, %tid.y;
mov.u32 %r7, %tid.x;
mad.lo.s32 %r409, %r5, %r451, %r7;
mov.u32 %r9, %ntid.y;
mul.lo.s32 %r10, %r9, %r5;
div.u32 %r11, %r153, %r5;
mov.u32 %r12, %ctaid.y;
setp.ge.s32	%p1, %r409, %r1;
@%p1 bra BB3_2;

BB3_1:
shl.b32 %r162, %r409, 2;
mov.u32 %r163, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r164, %r163, %r162;
mov.u32 %r165, -1;
st.shared.u32 [%r164], %r165;
mul.wide.s32 %rd20, %r409, 4;
add.s64 %rd21, %rd1, %rd20;
ld.global.u32 %r166, [%rd21];
add.s32 %r167, %r409, %r155;
shl.b32 %r168, %r167, 2;
add.s32 %r169, %r163, %r168;
st.shared.u32 [%r169+4], %r166;
mul.wide.s32 %rd22, %r167, 4;
add.s64 %rd23, %rd1, %rd22;
ld.global.u32 %r170, [%rd23+4];
add.s32 %r171, %r167, %r2;
shl.b32 %r172, %r171, 2;
add.s32 %r173, %r163, %r172;
st.shared.u32 [%r173+4], %r170;
add.s32 %r409, %r409, %r10;
setp.lt.s32	%p2, %r409, %r1;
@%p2 bra BB3_1;

BB3_2:
add.s32 %r174, %r154, 1;
mul.lo.s32 %r15, %r174, %r155;
setp.ge.s32	%p3, %r451, %r15;
@%p3 bra BB3_10;

mad.lo.s32 %r175, %r155, 4, %r7;
shl.b32 %r176, %r175, 2;
mov.u32 %r177, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r178, %r176, %r177;
add.s32 %r16, %r178, 16;
shl.b32 %r17, %r5, 2;
mad.lo.s32 %r18, %r12, %r153, %r7;
cvta.to.global.u64 %rd5, %rd11;
mov.u32 %r410, %r451;

BB3_4:
setp.lt.s32	%p4, %r11, 1;
@%p4 bra BB3_9;

mul.lo.s32 %r180, %r153, %r410;
shl.b32 %r181, %r180, 2;
add.s32 %r413, %r16, %r181;
mad.lo.s32 %r412, %r152, %r410, %r18;
mov.u32 %r414, 0;
mov.u32 %r411, %r18;

BB3_6:
setp.ge.s32	%p5, %r411, %r152;
@%p5 bra BB3_8;

mul.wide.s32 %rd24, %r412, 4;
add.s64 %rd25, %rd5, %rd24;
ld.global.f32 %f25, [%rd25];
st.shared.f32 [%r413], %f25;

BB3_8:
add.s32 %r413, %r413, %r17;
add.s32 %r412, %r412, %r5;
add.s32 %r411, %r411, %r5;
add.s32 %r414, %r414, 1;
setp.lt.s32	%p6, %r414, %r11;
@%p6 bra BB3_6;

BB3_9:
add.s32 %r410, %r410, %r9;
setp.lt.s32	%p7, %r410, %r15;
@%p7 bra BB3_4;

BB3_10:
bar.sync 0;
add.s32 %r415, %r451, %r3;
setp.ge.s32	%p8, %r415, %r4;
@%p8 bra BB3_29;

cvta.to.global.u64 %rd6, %rd12;
mad.lo.s32 %r182, %r155, 4, %r7;
shl.b32 %r183, %r182, 2;
mov.u32 %r184, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r185, %r183, %r184;
add.s32 %r32, %r185, 16;
shl.b32 %r33, %r5, 2;
mad.lo.s32 %r34, %r12, %r153, %r7;
mad.lo.s32 %r186, %r154, %r153, 4;
mad.lo.s32 %r187, %r155, %r186, %r7;
shl.b32 %r188, %r187, 2;
add.s32 %r189, %r188, %r184;
add.s32 %r35, %r189, 16;

BB3_12:
shl.b32 %r190, %r415, 1;
mul.wide.s32 %rd26, %r190, 4;
add.s64 %rd27, %rd6, %rd26;
add.s32 %r191, %r190, 1;
mul.wide.s32 %rd28, %r191, 4;
add.s64 %rd29, %rd6, %rd28;
ld.global.u32 %r37, [%rd29];
add.s32 %r192, %r37, -1;
setp.lt.s32	%p9, %r192, 0;
setp.ge.s32	%p10, %r192, %r155;
or.pred %p11, %p9, %p10;
ld.global.u32 %r38, [%rd27];
add.s32 %r193, %r38, -1;
setp.lt.s32	%p12, %r193, 0;
or.pred %p13, %p11, %p12;
setp.ge.s32	%p14, %r193, %r154;
or.pred %p15, %p13, %p14;
@%p15 bra BB3_28;

setp.ne.s32	%p16, %r7, 0;
@%p16 bra BB3_15;

shl.b32 %r194, %r37, 2;
add.s32 %r196, %r194, %r184;
add.s32 %r407, %r38, -1;
st.shared.u32 [%r196+-4], %r407;

BB3_15:
setp.lt.s32	%p17, %r154, 1;
@%p17 bra BB3_23;

mul.lo.s32 %r40, %r154, %r415;
mov.u32 %r416, 0;

BB3_17:
setp.lt.s32	%p18, %r11, 1;
@%p18 bra BB3_22;

mad.lo.s32 %r200, %r155, %r416, %r192;
mul.lo.s32 %r201, %r153, %r200;
shl.b32 %r202, %r201, 2;
add.s32 %r419, %r32, %r202;
add.s32 %r203, %r40, %r416;
mad.lo.s32 %r417, %r152, %r203, %r34;
mov.u32 %r420, 0;
mov.u32 %r418, %r34;

BB3_19:
setp.ge.u32	%p19, %r418, %r152;
@%p19 bra BB3_21;

mul.wide.s32 %rd30, %r417, 4;
add.s64 %rd31, %rd2, %rd30;
ld.global.f32 %f26, [%rd31];
atom.shared.add.f32 %f27, [%r419], %f26;

BB3_21:
add.s32 %r419, %r419, %r33;
add.s32 %r418, %r418, %r5;
add.s32 %r417, %r417, %r5;
add.s32 %r420, %r420, 1;
setp.lt.s32	%p20, %r420, %r11;
@%p20 bra BB3_19;

BB3_22:
add.s32 %r416, %r416, 1;
setp.lt.s32	%p21, %r416, %r154;
@%p21 bra BB3_17;

BB3_23:
setp.lt.s32	%p22, %r11, 1;
@%p22 bra BB3_28;

mul.lo.s32 %r206, %r153, %r192;
shl.b32 %r207, %r206, 2;
add.s32 %r423, %r35, %r207;
mad.lo.s32 %r208, %r154, %r415, %r38;
add.s32 %r209, %r208, -1;
mad.lo.s32 %r421, %r152, %r209, %r34;
mov.u32 %r424, 0;
mov.u32 %r422, %r34;

BB3_25:
setp.ge.u32	%p23, %r422, %r152;
@%p23 bra BB3_27;

mul.wide.s32 %rd32, %r421, 4;
add.s64 %rd33, %rd2, %rd32;
ld.global.f32 %f28, [%rd33];
mul.f32 %f29, %f28, %f28;
atom.shared.add.f32 %f30, [%r423], %f29;

BB3_27:
add.s32 %r423, %r423, %r33;
add.s32 %r422, %r422, %r5;
add.s32 %r421, %r421, %r5;
add.s32 %r424, %r424, 1;
setp.lt.s32	%p24, %r424, %r11;
@%p24 bra BB3_25;

BB3_28:
add.s32 %r415, %r415, %r9;
setp.lt.s32	%p25, %r415, %r4;
@%p25 bra BB3_12;

BB3_29:
bar.sync 0;
or.b32 %r210, %r7, %r451;
add.s32 %r211, %r2, %r155;
shl.b32 %r212, %r211, 2;
mov.u32 %r213, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r64, %r213, %r212;
setp.ne.s32	%p26, %r210, 0;
@%p26 bra BB3_63;

mov.u32 %r435, 0;
setp.lt.s32	%p27, %r155, 1;
@%p27 bra BB3_62;

and.b32 %r221, %r155, 3;
mov.u32 %r426, 0;
setp.eq.s32	%p28, %r221, 0;
@%p28 bra BB3_32;

setp.eq.s32	%p29, %r221, 1;
@%p29 bra BB3_34;
bra.uni BB3_35;

BB3_34:
mov.u32 %r429, %r426;
bra.uni BB3_43;

BB3_32:
mov.u32 %r435, %r426;
bra.uni BB3_47;

BB3_35:
setp.eq.s32	%p30, %r221, 2;
@%p30 bra BB3_36;
bra.uni BB3_37;

BB3_36:
mov.u32 %r425, %r426;
bra.uni BB3_39;

BB3_37:
ld.shared.u32 %r224, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r425, 1;
setp.lt.s32	%p31, %r224, 0;
@%p31 bra BB3_39;

shl.b32 %r227, %r2, 2;
add.s32 %r229, %r213, %r227;
mov.u32 %r230, 0;
st.shared.u32 [%r229], %r230;
mov.u32 %r425, 1;
mov.u32 %r426, %r425;

BB3_39:
shl.b32 %r231, %r425, 2;
add.s32 %r233, %r213, %r231;
ld.shared.u32 %r234, [%r233];
setp.lt.s32	%p32, %r234, 0;
@%p32 bra BB3_40;

add.s32 %r429, %r426, 1;
add.s32 %r235, %r426, %r2;
shl.b32 %r236, %r235, 2;
add.s32 %r238, %r213, %r236;
st.shared.u32 [%r238], %r425;
bra.uni BB3_42;

BB3_40:
mov.u32 %r429, %r426;

BB3_42:
add.s32 %r426, %r425, 1;

BB3_43:
shl.b32 %r239, %r426, 2;
add.s32 %r241, %r213, %r239;
ld.shared.u32 %r242, [%r241];
setp.lt.s32	%p33, %r242, 0;
@%p33 bra BB3_44;

add.s32 %r435, %r429, 1;
add.s32 %r243, %r429, %r2;
shl.b32 %r244, %r243, 2;
add.s32 %r246, %r213, %r244;
st.shared.u32 [%r246], %r426;
bra.uni BB3_46;

BB3_44:
mov.u32 %r435, %r429;

BB3_46:
add.s32 %r426, %r426, 1;

BB3_47:
setp.lt.u32	%p34, %r155, 4;
@%p34 bra BB3_62;

shl.b32 %r247, %r426, 2;
add.s32 %r433, %r213, %r247;

BB3_49:
ld.shared.u32 %r249, [%r433];
setp.lt.s32	%p35, %r249, 0;
@%p35 bra BB3_50;

add.s32 %r436, %r435, 1;
add.s32 %r250, %r435, %r2;
shl.b32 %r251, %r250, 2;
add.s32 %r253, %r213, %r251;
st.shared.u32 [%r253], %r426;
bra.uni BB3_52;

BB3_50:
mov.u32 %r436, %r435;

BB3_52:
ld.shared.u32 %r254, [%r433+4];
setp.lt.s32	%p36, %r254, 0;
@%p36 bra BB3_53;

add.s32 %r437, %r436, 1;
add.s32 %r255, %r436, %r2;
shl.b32 %r256, %r255, 2;
add.s32 %r258, %r213, %r256;
add.s32 %r259, %r426, 1;
st.shared.u32 [%r258], %r259;
bra.uni BB3_55;

BB3_53:
mov.u32 %r437, %r436;

BB3_55:
ld.shared.u32 %r260, [%r433+8];
setp.lt.s32	%p37, %r260, 0;
@%p37 bra BB3_56;

add.s32 %r438, %r437, 1;
add.s32 %r261, %r437, %r2;
shl.b32 %r262, %r261, 2;
add.s32 %r264, %r213, %r262;
add.s32 %r265, %r426, 2;
st.shared.u32 [%r264], %r265;
bra.uni BB3_58;

BB3_56:
mov.u32 %r438, %r437;

BB3_58:
ld.shared.u32 %r266, [%r433+12];
setp.lt.s32	%p38, %r266, 0;
@%p38 bra BB3_59;

add.s32 %r435, %r438, 1;
add.s32 %r267, %r438, %r2;
shl.b32 %r268, %r267, 2;
add.s32 %r270, %r213, %r268;
add.s32 %r271, %r426, 3;
st.shared.u32 [%r270], %r271;
bra.uni BB3_61;

BB3_59:
mov.u32 %r435, %r438;

BB3_61:
add.s32 %r426, %r426, 4;
setp.lt.s32	%p39, %r426, %r155;
add.s32 %r433, %r433, 16;
@%p39 bra BB3_49;

BB3_62:
st.shared.u32 [%r64], %r435;

BB3_63:
bar.sync 0;
cvt.s64.s32	%rd34, %r155;
add.s64 %rd7, %rd34, 1;
cvt.u64.u32	%rd35, %r155;
add.s64 %rd36, %rd7, %rd35;
cvt.u32.u64	%r272, %rd36;
add.s32 %r273, %r272, %r2;
shl.b32 %r274, %r273, 2;
add.s32 %r276, %r213, %r274;
ld.shared.u32 %r93, [%r276];
ld.shared.u32 %r94, [%r64];
setp.ge.s32	%p40, %r451, %r94;
@%p40 bra BB3_83;

mad.lo.s32 %r277, %r154, %r153, 4;
mad.lo.s32 %r278, %r155, %r277, %r7;
shl.b32 %r279, %r278, 2;
add.s32 %r281, %r279, %r213;
add.s32 %r95, %r281, 16;
shl.b32 %r96, %r5, 2;
mad.lo.s32 %r282, %r155, 4, %r7;
shl.b32 %r283, %r282, 2;
add.s32 %r284, %r283, %r213;
add.s32 %r97, %r284, 16;
mad.lo.s32 %r98, %r12, %r153, %r7;
mov.u32 %r441, %r451;

BB3_65:
add.s32 %r285, %r441, %r2;
shl.b32 %r286, %r285, 2;
add.s32 %r288, %r213, %r286;
ld.shared.u32 %r100, [%r288];
cvt.s64.s32	%rd8, %r100;
shl.b32 %r289, %r100, 2;
add.s32 %r290, %r213, %r289;
ld.shared.u32 %r101, [%r290];
add.s32 %r291, %r100, 1;
mul.lo.s32 %r292, %r291, %r100;
shr.u32 %r293, %r292, 31;
add.s32 %r294, %r292, %r293;
shr.s32 %r102, %r294, 1;
mov.f32 %f78, 0f00000000;
setp.lt.s32	%p41, %r11, 1;
@%p41 bra BB3_70;

mul.lo.s32 %r296, %r153, %r100;
shl.b32 %r297, %r296, 2;
add.s32 %r444, %r95, %r297;
mad.lo.s32 %r298, %r155, %r101, %r100;
mul.lo.s32 %r299, %r153, %r298;
shl.b32 %r300, %r299, 2;
add.s32 %r443, %r97, %r300;
mov.f32 %f32, 0f00000000;
mov.u32 %r445, 0;
mov.u32 %r442, %r98;
mov.f32 %f78, %f32;

BB3_67:
setp.ge.u32	%p42, %r442, %r152;
mov.f32 %f76, %f32;
mov.f32 %f77, %f32;
@%p42 bra BB3_69;

ld.shared.f32 %f76, [%r443];
ld.shared.f32 %f77, [%r444];

BB3_69:
mul.f32 %f35, %f76, %f76;
sub.f32 %f36, %f35, %f77;
fma.rn.f32 %f78, %f36, 0f3F000000, %f78;
add.s32 %r444, %r444, %r96;
add.s32 %r443, %r443, %r96;
add.s32 %r442, %r442, %r5;
add.s32 %r445, %r445, 1;
setp.lt.s32	%p43, %r445, %r11;
@%p43 bra BB3_67;

BB3_70:
bar.warp.sync -1;
mov.b32 %r301, %f78;
mov.u32 %r302, 31;
mov.u32 %r303, 1;
mov.u32 %r304, -1;
shfl.sync.bfly.b32 %r305|%p44, %r301, %r303, %r302, %r304;
mov.b32 %f37, %r305;
add.f32 %f38, %f78, %f37;
mov.b32 %r306, %f38;
mov.u32 %r307, 2;
shfl.sync.bfly.b32 %r308|%p45, %r306, %r307, %r302, %r304;
mov.b32 %f39, %r308;
add.f32 %f40, %f38, %f39;
mov.b32 %r309, %f40;
mov.u32 %r310, 4;
shfl.sync.bfly.b32 %r311|%p46, %r309, %r310, %r302, %r304;
mov.b32 %f41, %r311;
add.f32 %f42, %f40, %f41;
mov.b32 %r312, %f42;
mov.u32 %r313, 8;
shfl.sync.bfly.b32 %r314|%p47, %r312, %r313, %r302, %r304;
mov.b32 %f43, %r314;
add.f32 %f44, %f42, %f43;
mov.b32 %r315, %f44;
mov.u32 %r316, 16;
shfl.sync.bfly.b32 %r317|%p48, %r315, %r316, %r302, %r304;
mov.b32 %f45, %r317;
add.f32 %f8, %f44, %f45;
setp.ne.s32	%p49, %r7, 0;
@%p49 bra BB3_72;

cvt.s64.s32	%rd37, %r102;
add.s64 %rd38, %rd8, %rd4;
add.s64 %rd39, %rd38, %rd37;
shl.b64 %rd40, %rd39, 2;
add.s64 %rd41, %rd3, %rd40;
atom.global.add.f32 %f46, [%rd41], %f8;

BB3_72:
bar.warp.sync -1;
setp.lt.s32	%p50, %r441, 1;
@%p50 bra BB3_82;

cvt.s64.s32	%rd42, %r102;
add.s64 %rd9, %rd42, %rd4;
mul.lo.s32 %r113, %r155, %r101;
mov.u32 %r446, 0;

BB3_74:
add.s32 %r319, %r446, %r2;
shl.b32 %r320, %r319, 2;
add.s32 %r322, %r213, %r320;
ld.shared.u32 %r115, [%r322];
cvt.s64.s32	%rd10, %r115;
mov.f32 %f82, 0f00000000;
@%p41 bra BB3_79;

shl.b32 %r324, %r115, 2;
add.s32 %r326, %r213, %r324;
ld.shared.u32 %r327, [%r326];
cvt.u32.u64	%r328, %rd10;
cvt.u32.u64	%r329, %rd8;
mad.lo.s32 %r330, %r155, %r327, %r329;
mul.lo.s32 %r331, %r153, %r330;
shl.b32 %r332, %r331, 2;
add.s32 %r449, %r97, %r332;
add.s32 %r333, %r113, %r328;
mul.lo.s32 %r334, %r153, %r333;
shl.b32 %r335, %r334, 2;
add.s32 %r448, %r97, %r335;
mov.f32 %f48, 0f00000000;
mov.u32 %r450, 0;
mov.u32 %r447, %r98;
mov.f32 %f82, %f48;

BB3_76:
setp.ge.u32	%p52, %r447, %r152;
mov.f32 %f80, %f48;
mov.f32 %f81, %f48;
@%p52 bra BB3_78;

ld.shared.f32 %f80, [%r448];
ld.shared.f32 %f81, [%r449];

BB3_78:
fma.rn.f32 %f82, %f80, %f81, %f82;
add.s32 %r449, %r449, %r96;
add.s32 %r448, %r448, %r96;
add.s32 %r447, %r447, %r5;
add.s32 %r450, %r450, 1;
setp.lt.s32	%p53, %r450, %r11;
@%p53 bra BB3_76;

BB3_79:
bar.warp.sync -1;
mov.b32 %r336, %f82;
shfl.sync.bfly.b32 %r340|%p54, %r336, %r303, %r302, %r304;
mov.b32 %f51, %r340;
add.f32 %f52, %f82, %f51;
mov.b32 %r341, %f52;
shfl.sync.bfly.b32 %r343|%p55, %r341, %r307, %r302, %r304;
mov.b32 %f53, %r343;
add.f32 %f54, %f52, %f53;
mov.b32 %r344, %f54;
shfl.sync.bfly.b32 %r346|%p56, %r344, %r310, %r302, %r304;
mov.b32 %f55, %r346;
add.f32 %f56, %f54, %f55;
mov.b32 %r347, %f56;
shfl.sync.bfly.b32 %r349|%p57, %r347, %r313, %r302, %r304;
mov.b32 %f57, %r349;
add.f32 %f58, %f56, %f57;
mov.b32 %r350, %f58;
shfl.sync.bfly.b32 %r352|%p58, %r350, %r316, %r302, %r304;
mov.b32 %f59, %r352;
add.f32 %f16, %f58, %f59;
@%p49 bra BB3_81;

add.s64 %rd43, %rd9, %rd10;
shl.b64 %rd44, %rd43, 2;
add.s64 %rd45, %rd3, %rd44;
atom.global.add.f32 %f60, [%rd45], %f16;

BB3_81:
bar.warp.sync -1;
add.s32 %r446, %r446, 1;
setp.lt.s32	%p60, %r446, %r441;
@%p60 bra BB3_74;

BB3_82:
add.s32 %r441, %r441, %r9;
setp.lt.s32	%p61, %r441, %r94;
@%p61 bra BB3_65;

BB3_83:
setp.ge.s32	%p62, %r451, %r93;
@%p62 bra BB3_96;

mov.u32 %r408, %ctaid.y;
mad.lo.s32 %r353, %r155, 4, %r7;
shl.b32 %r354, %r353, 2;
add.s32 %r356, %r354, %r213;
add.s32 %r128, %r356, 16;
shl.b32 %r129, %r5, 2;
mad.lo.s32 %r131, %r408, %r153, %r7;

BB3_85:
cvt.u64.u32	%rd46, %r451;
add.s64 %rd47, %rd46, %rd7;
cvt.u32.u64	%r357, %rd47;
add.s32 %r358, %r357, %r2;
shl.b32 %r359, %r358, 2;
add.s32 %r361, %r213, %r359;
ld.shared.u32 %r453, [%r361];
setp.lt.s32	%p63, %r94, 1;
@%p63 bra BB3_95;

add.s32 %r363, %r453, %r1;
shl.b32 %r364, %r363, 2;
add.s32 %r366, %r213, %r364;
ld.shared.u32 %r367, [%r366];
mul.lo.s32 %r134, %r155, %r367;
mov.u32 %r452, 0;

BB3_87:
add.s32 %r368, %r452, %r2;
shl.b32 %r369, %r368, 2;
add.s32 %r371, %r213, %r369;
ld.shared.u32 %r137, [%r371];
min.s32 %r138, %r137, %r453;
mov.f32 %f86, 0f00000000;
setp.lt.s32	%p64, %r11, 1;
@%p64 bra BB3_92;

shl.b32 %r373, %r137, 2;
add.s32 %r375, %r213, %r373;
ld.shared.u32 %r376, [%r375];
mad.lo.s32 %r377, %r155, %r376, %r453;
mul.lo.s32 %r378, %r153, %r377;
shl.b32 %r379, %r378, 2;
add.s32 %r456, %r128, %r379;
add.s32 %r380, %r134, %r137;
mul.lo.s32 %r381, %r153, %r380;
shl.b32 %r382, %r381, 2;
add.s32 %r455, %r128, %r382;
mov.f32 %f62, 0f00000000;
mov.u32 %r457, 0;
mov.u32 %r454, %r131;
mov.f32 %f86, %f62;

BB3_89:
setp.ge.u32	%p65, %r454, %r152;
mov.f32 %f84, %f62;
mov.f32 %f85, %f62;
@%p65 bra BB3_91;

ld.shared.f32 %f84, [%r455];
ld.shared.f32 %f85, [%r456];

BB3_91:
fma.rn.f32 %f86, %f84, %f85, %f86;
add.s32 %r456, %r456, %r129;
add.s32 %r455, %r455, %r129;
add.s32 %r454, %r454, %r5;
add.s32 %r457, %r457, 1;
setp.lt.s32	%p66, %r457, %r11;
@%p66 bra BB3_89;

BB3_92:
bar.warp.sync -1;
mov.b32 %r383, %f86;
mov.u32 %r384, 31;
mov.u32 %r385, 1;
mov.u32 %r386, -1;
shfl.sync.bfly.b32 %r387|%p67, %r383, %r385, %r384, %r386;
mov.b32 %f65, %r387;
add.f32 %f66, %f86, %f65;
mov.b32 %r388, %f66;
mov.u32 %r389, 2;
shfl.sync.bfly.b32 %r390|%p68, %r388, %r389, %r384, %r386;
mov.b32 %f67, %r390;
add.f32 %f68, %f66, %f67;
mov.b32 %r391, %f68;
mov.u32 %r392, 4;
shfl.sync.bfly.b32 %r393|%p69, %r391, %r392, %r384, %r386;
mov.b32 %f69, %r393;
add.f32 %f70, %f68, %f69;
mov.b32 %r394, %f70;
mov.u32 %r395, 8;
shfl.sync.bfly.b32 %r396|%p70, %r394, %r395, %r384, %r386;
mov.b32 %f71, %r396;
add.f32 %f72, %f70, %f71;
mov.b32 %r397, %f72;
mov.u32 %r398, 16;
shfl.sync.bfly.b32 %r399|%p71, %r397, %r398, %r384, %r386;
mov.b32 %f73, %r399;
add.f32 %f24, %f72, %f73;
add.s32 %r400, %r453, 1;
mul.lo.s32 %r401, %r400, %r453;
setp.lt.s32	%p72, %r137, %r453;
add.s32 %r402, %r137, 1;
mul.lo.s32 %r403, %r402, %r137;
selp.b32	%r149, %r401, %r403, %p72;
setp.ne.s32	%p73, %r7, 0;
@%p73 bra BB3_94;

shr.u32 %r404, %r149, 31;
add.s32 %r405, %r149, %r404;
shr.s32 %r406, %r405, 1;
cvt.s64.s32	%rd48, %r406;
cvt.s64.s32	%rd49, %r138;
add.s64 %rd50, %rd49, %rd4;
add.s64 %rd51, %rd50, %rd48;
shl.b64 %rd52, %rd51, 2;
add.s64 %rd53, %rd3, %rd52;
atom.global.add.f32 %f74, [%rd53], %f24;

BB3_94:
bar.warp.sync -1;
add.s32 %r452, %r452, 1;
setp.lt.s32	%p74, %r452, %r94;
mov.u32 %r453, %r138;
@%p74 bra BB3_87;

BB3_95:
add.s32 %r451, %r451, %r9;
setp.lt.s32	%p75, %r451, %r93;
@%p75 bra BB3_85;

BB3_96:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_11
)
{
.reg .pred %p<38>;
.reg .f32 %f<6>;
.reg .b32 %r<176>;
.reg .b64 %rd<89>;


ld.param.u32 %r51, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_0];
ld.param.u32 %r52, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_1];
ld.param.u32 %r53, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_2];
ld.param.u32 %r54, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_3];
ld.param.u32 %r55, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_4];
ld.param.u64 %rd31, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_5];
ld.param.u64 %rd25, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_6];
ld.param.u64 %rd26, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_7];
ld.param.u64 %rd27, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_8];
ld.param.u64 %rd28, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_9];
ld.param.u64 %rd29, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_10];
ld.param.u64 %rd30, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartIfLi32EEEviiiiiPiPKT_PKiPS3_S8_S2_S8__param_11];
cvta.to.global.u64 %rd32, %rd31;
mov.u32 %r56, %ntid.x;
mov.u32 %r57, %tid.y;
mov.u32 %r58, %tid.x;
mad.lo.s32 %r59, %r56, %r57, %r58;
ld.global.u32 %r1, [%rd32];
setp.ge.s32	%p1, %r59, %r54;
@%p1 bra BB4_24;

add.s32 %r60, %r54, -1;
mul.lo.s32 %r63, %r56, %r57;
sub.s32 %r64, %r60, %r63;
mov.u32 %r65, %tid.x;
sub.s32 %r66, %r64, %r65;
shr.u32 %r67, %r66, 10;
add.s32 %r68, %r67, 1;
and.b32 %r2, %r68, 3;
setp.eq.s32	%p2, %r2, 0;
add.s32 %r165, %r63, %r65;
@%p2 bra BB4_13;

setp.eq.s32	%p3, %r2, 1;
mad.lo.s32 %r163, %r56, %r57, %r65;
@%p3 bra BB4_10;

setp.eq.s32	%p4, %r2, 2;
mad.lo.s32 %r162, %r56, %r57, %r65;
@%p4 bra BB4_7;

mad.lo.s32 %r6, %r56, %r57, %r65;
cvta.to.global.u64 %rd33, %rd27;
mul.wide.s32 %rd34, %r6, 4;
add.s64 %rd35, %rd33, %rd34;
mov.u32 %r78, 0;
st.global.u32 [%rd35], %r78;
setp.ge.s32	%p5, %r6, %r53;
@%p5 bra BB4_6;

cvta.to.global.u64 %rd36, %rd29;
add.s64 %rd38, %rd36, %rd34;
mov.u32 %r79, -1;
st.global.u32 [%rd38], %r79;

BB4_6:
add.s32 %r162, %r6, 1024;

BB4_7:
cvta.to.global.u64 %rd39, %rd27;
cvt.s64.s32	%rd1, %r162;
mul.wide.s32 %rd40, %r162, 4;
add.s64 %rd41, %rd39, %rd40;
mov.u32 %r84, 0;
st.global.u32 [%rd41], %r84;
setp.ge.s32	%p6, %r162, %r53;
@%p6 bra BB4_9;

cvta.to.global.u64 %rd42, %rd29;
add.s64 %rd44, %rd42, %rd40;
mov.u32 %r85, -1;
st.global.u32 [%rd44], %r85;

BB4_9:
cvt.u32.u64	%r86, %rd1;
add.s32 %r163, %r86, 1024;

BB4_10:
cvta.to.global.u64 %rd45, %rd27;
cvt.s64.s32	%rd2, %r163;
mul.wide.s32 %rd46, %r163, 4;
add.s64 %rd47, %rd45, %rd46;
mov.u32 %r87, 0;
st.global.u32 [%rd47], %r87;
setp.ge.s32	%p7, %r163, %r53;
@%p7 bra BB4_12;

cvta.to.global.u64 %rd48, %rd29;
add.s64 %rd50, %rd48, %rd46;
mov.u32 %r88, -1;
st.global.u32 [%rd50], %r88;

BB4_12:
cvt.u32.u64	%r89, %rd2;
add.s32 %r165, %r89, 1024;

BB4_13:
setp.lt.u32	%p8, %r68, 4;
@%p8 bra BB4_24;

mul.wide.s32 %rd82, %r165, 4;
cvta.to.global.u64 %rd51, %rd27;
add.s64 %rd83, %rd51, %rd82;

BB4_15:
mov.u32 %r99, 0;
st.global.u32 [%rd83], %r99;
cvta.to.global.u64 %rd52, %rd29;
add.s64 %rd7, %rd52, %rd82;
setp.ge.s32	%p9, %r165, %r53;
@%p9 bra BB4_17;

mov.u32 %r100, -1;
st.global.u32 [%rd7], %r100;

BB4_17:
add.s32 %r14, %r165, 1024;
st.global.u32 [%rd83+4096], %r99;
setp.ge.s32	%p10, %r14, %r53;
@%p10 bra BB4_19;

mov.u32 %r102, -1;
st.global.u32 [%rd7+4096], %r102;

BB4_19:
st.global.u32 [%rd83+8192], %r99;
add.s32 %r15, %r14, 1024;
setp.ge.s32	%p11, %r15, %r53;
@%p11 bra BB4_21;

mov.u32 %r104, -1;
st.global.u32 [%rd7+8192], %r104;

BB4_21:
st.global.u32 [%rd83+12288], %r99;
add.s32 %r16, %r15, 1024;
setp.ge.s32	%p12, %r16, %r53;
@%p12 bra BB4_23;

mov.u32 %r106, -1;
st.global.u32 [%rd7+12288], %r106;

BB4_23:
add.s64 %rd82, %rd82, 16384;
add.s32 %r165, %r16, 1024;
setp.lt.s32	%p13, %r165, %r54;
add.s64 %rd83, %rd83, 16384;
@%p13 bra BB4_15;

BB4_24:
setp.ge.s32	%p14, %r59, %r55;
@%p14 bra BB4_34;

add.s32 %r111, %r55, -1;
mul.lo.s32 %r114, %r56, %r57;
sub.s32 %r115, %r111, %r114;
mov.u32 %r116, %tid.x;
sub.s32 %r117, %r115, %r116;
shr.u32 %r118, %r117, 10;
add.s32 %r18, %r118, 1;
and.b32 %r19, %r18, 3;
setp.eq.s32	%p15, %r19, 0;
add.s32 %r169, %r114, %r116;
@%p15 bra BB4_31;

setp.eq.s32	%p16, %r19, 1;
mad.lo.s32 %r167, %r56, %r57, %r116;
@%p16 bra BB4_30;

setp.eq.s32	%p17, %r19, 2;
mad.lo.s32 %r166, %r56, %r57, %r116;
@%p17 bra BB4_29;

mad.lo.s32 %r128, %r56, %r57, %r116;
cvta.to.global.u64 %rd53, %rd30;
mul.wide.s32 %rd54, %r128, 4;
add.s64 %rd55, %rd53, %rd54;
mov.u32 %r129, 0;
st.global.u32 [%rd55], %r129;
add.s32 %r166, %r128, 1024;

BB4_29:
cvta.to.global.u64 %rd56, %rd30;
mul.wide.s32 %rd57, %r166, 4;
add.s64 %rd58, %rd56, %rd57;
mov.u32 %r130, 0;
st.global.u32 [%rd58], %r130;
add.s32 %r167, %r166, 1024;

BB4_30:
cvta.to.global.u64 %rd59, %rd30;
mul.wide.s32 %rd60, %r167, 4;
add.s64 %rd61, %rd59, %rd60;
mov.u32 %r131, 0;
st.global.u32 [%rd61], %r131;
add.s32 %r169, %r167, 1024;

BB4_31:
setp.lt.u32	%p18, %r18, 4;
@%p18 bra BB4_34;

cvta.to.global.u64 %rd62, %rd30;
mul.wide.s32 %rd63, %r169, 4;
add.s64 %rd84, %rd62, %rd63;

BB4_33:
mov.u32 %r132, 0;
st.global.u32 [%rd84], %r132;
st.global.u32 [%rd84+4096], %r132;
st.global.u32 [%rd84+8192], %r132;
st.global.u32 [%rd84+12288], %r132;
add.s64 %rd84, %rd84, 16384;
add.s32 %r169, %r169, 4096;
setp.lt.s32	%p19, %r169, %r55;
@%p19 bra BB4_33;

BB4_34:
mov.u32 %r133, %ctaid.x;
shl.b32 %r134, %r133, 5;
add.s32 %r170, %r134, %r57;
setp.ge.s32	%p20, %r170, %r1;
@%p20 bra BB4_53;

add.s32 %r137, %r51, 31;
shr.s32 %r138, %r137, 31;
shr.u32 %r139, %r138, 27;
add.s32 %r140, %r137, %r139;
and.b32 %r31, %r140, -32;
cvta.to.global.u64 %rd64, %rd26;
cvta.to.global.u64 %rd78, %rd28;

BB4_36:
shl.b32 %r144, %r170, 1;
mul.wide.s32 %rd65, %r144, 4;
add.s64 %rd66, %rd64, %rd65;
ld.global.u32 %r34, [%rd66];
add.s32 %r35, %r34, -1;
ld.global.u32 %r36, [%rd66+4];
add.s32 %r37, %r36, -1;
setp.lt.s32	%p21, %r37, 0;
setp.ge.s32	%p22, %r37, %r53;
or.pred %p23, %p21, %p22;
setp.lt.s32	%p24, %r35, 0;
or.pred %p25, %p23, %p24;
setp.ge.s32	%p26, %r35, %r52;
or.pred %p27, %p25, %p26;
@%p27 bra BB4_52;

setp.ne.s32	%p28, %r58, 0;
@%p28 bra BB4_39;

cvta.to.global.u64 %rd67, %rd29;
mul.wide.s32 %rd68, %r37, 4;
add.s64 %rd69, %rd67, %rd68;
add.s32 %r161, %r34, -1;
st.global.u32 [%rd69], %r161;

BB4_39:
setp.lt.s32	%p29, %r52, 1;
@%p29 bra BB4_47;

mov.u32 %r171, 0;

BB4_41:
setp.lt.s32	%p30, %r51, 1;
@%p30 bra BB4_46;

mad.lo.s32 %r148, %r52, %r170, %r171;
mov.u32 %r172, %tid.x;
mad.lo.s32 %r149, %r51, %r148, %r172;
cvta.to.global.u64 %rd70, %rd25;
mul.wide.s32 %rd71, %r149, 4;
add.s64 %rd86, %rd70, %rd71;
mad.lo.s32 %r150, %r53, %r171, %r36;
add.s32 %r151, %r150, -1;
mul.lo.s32 %r152, %r51, %r151;
cvta.to.global.u64 %rd72, %rd27;
mul.wide.s32 %rd73, %r172, 4;
add.s64 %rd74, %rd72, %rd73;
mul.wide.s32 %rd75, %r152, 4;
add.s64 %rd85, %rd74, %rd75;
mov.u32 %r173, 0;

BB4_43:
setp.ge.s32	%p31, %r172, %r51;
@%p31 bra BB4_45;

ld.global.f32 %f1, [%rd86];
atom.global.add.f32 %f2, [%rd85], %f1;

BB4_45:
add.s32 %r173, %r173, 32;
add.s64 %rd86, %rd86, 128;
add.s32 %r172, %r172, 32;
add.s64 %rd85, %rd85, 128;
setp.lt.s32	%p32, %r173, %r31;
@%p32 bra BB4_43;

BB4_46:
add.s32 %r171, %r171, 1;
setp.lt.s32	%p33, %r171, %r52;
@%p33 bra BB4_41;

BB4_47:
setp.lt.s32	%p34, %r51, 1;
@%p34 bra BB4_52;

mad.lo.s32 %r154, %r52, %r170, %r34;
add.s32 %r155, %r154, -1;
mad.lo.s32 %r156, %r51, %r155, %r58;
cvta.to.global.u64 %rd76, %rd25;
mul.wide.s32 %rd77, %r156, 4;
add.s64 %rd88, %rd76, %rd77;
mul.lo.s32 %r158, %r51, %r37;
mul.wide.s32 %rd79, %r58, 4;
add.s64 %rd80, %rd78, %rd79;
mul.wide.s32 %rd81, %r158, 4;
add.s64 %rd87, %rd80, %rd81;
mov.u32 %r175, 0;
mov.u32 %r174, %r58;

BB4_49:
setp.ge.s32	%p35, %r174, %r51;
@%p35 bra BB4_51;

ld.global.f32 %f3, [%rd88];
mul.f32 %f4, %f3, %f3;
atom.global.add.f32 %f5, [%rd87], %f4;

BB4_51:
add.s32 %r175, %r175, 32;
add.s64 %rd88, %rd88, 128;
add.s32 %r174, %r174, 32;
add.s64 %rd87, %rd87, 128;
setp.lt.s32	%p36, %r175, %r31;
@%p36 bra BB4_49;

BB4_52:
mov.u32 %r159, %nctaid.x;
shl.b32 %r160, %r159, 5;
add.s32 %r170, %r170, %r160;
setp.lt.s32	%p37, %r170, %r1;
@%p37 bra BB4_36;

BB4_53:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_6
)
{
.reg .f32 %f<4>;
.reg .b32 %r<20>;
.reg .b64 %rd<18>;


ld.param.u32 %r1, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_1];
ld.param.u32 %r2, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_2];
ld.param.u32 %r3, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_3];
ld.param.u64 %rd1, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_4];
ld.param.u64 %rd2, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_5];
ld.param.u64 %rd3, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartIfEEviiiiPT_S3_S3__param_6];
cvta.to.global.u64 %rd4, %rd1;
cvta.to.global.u64 %rd5, %rd3;
cvta.to.global.u64 %rd6, %rd2;
mov.u32 %r4, %tid.x;
mov.u32 %r5, %ctaid.x;
rem.u32 %r6, %r5, %r3;
mad.lo.s32 %r7, %r6, %r1, %r4;
div.u32 %r8, %r5, %r3;
mul.wide.s32 %rd7, %r7, 4;
add.s64 %rd8, %rd6, %rd7;
ld.global.f32 %f1, [%rd8];
add.s32 %r9, %r2, 1;
mul.lo.s32 %r10, %r9, %r1;
mul.lo.s32 %r11, %r10, %r3;
mul.lo.s32 %r12, %r11, %r8;
mul.lo.s32 %r13, %r2, %r1;
mad.lo.s32 %r14, %r13, %r3, %r12;
add.s32 %r15, %r14, %r7;
mul.wide.s32 %rd9, %r15, 4;
add.s64 %rd10, %rd5, %rd9;
st.global.f32 [%rd10], %f1;
add.s64 %rd11, %rd4, %rd7;
ld.global.f32 %f2, [%rd11];
mul.lo.s32 %r16, %r3, %r1;
add.s32 %r17, %r7, %r16;
mul.wide.s32 %rd12, %r17, 4;
add.s64 %rd13, %rd4, %rd12;
ld.global.f32 %f3, [%rd13];
add.s32 %r18, %r12, %r7;
mul.wide.s32 %rd14, %r18, 4;
add.s64 %rd15, %rd5, %rd14;
st.global.f32 [%rd15], %f2;
add.s32 %r19, %r18, %r16;
mul.wide.s32 %rd16, %r19, 4;
add.s64 %rd17, %rd5, %rd16;
st.global.f32 [%rd17], %f3;
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_2,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_10
)
{
.local .align 16 .b8 __local_depot6[48];
.reg .b64 %SP;
.reg .b64 %SPL;
.reg .pred %p<93>;
.reg .f32 %f<148>;
.reg .b32 %r<438>;
.reg .b64 %rd<134>;


mov.u64 %SPL, __local_depot6;
ld.param.u32 %r116, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_0];
ld.param.u32 %r117, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_1];
ld.param.u32 %r118, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_2];
ld.param.u64 %rd47, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_3];
ld.param.u64 %rd48, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_6];
ld.param.u64 %rd49, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_7];
ld.param.u64 %rd46, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_8];
ld.param.u64 %rd50, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_9];
ld.param.u64 %rd51, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuIfLi32EEEviiiPiPT_S4_S2_PKS3_PKiS4_S4__param_10];
cvta.to.global.u64 %rd1, %rd48;
cvta.to.global.u64 %rd2, %rd49;
cvta.to.global.u64 %rd3, %rd51;
cvta.to.global.u64 %rd4, %rd50;
add.u64 %rd5, %SPL, 0;
add.s32 %r119, %r118, 1;
mul.lo.s32 %r120, %r119, %r118;
shr.u32 %r121, %r120, 31;
add.s32 %r122, %r120, %r121;
shr.s32 %r123, %r122, 1;
mov.u32 %r1, %ctaid.x;
mul.lo.s32 %r2, %r123, %r1;
add.s32 %r124, %r117, 1;
mul.lo.s32 %r125, %r124, %r116;
mul.lo.s32 %r3, %r125, %r118;
mul.lo.s32 %r4, %r3, %r1;
mul.lo.s32 %r126, %r117, %r116;
mad.lo.s32 %r5, %r126, %r118, %r4;
cvta.to.global.u64 %rd53, %rd47;
mul.wide.s32 %rd54, %r1, 4;
add.s64 %rd55, %rd53, %rd54;
ld.global.u32 %r6, [%rd55];
ld.global.u32 %r7, [%rd55+4];
mov.u32 %r127, %ntid.x;
mov.u32 %r424, %tid.y;
mov.u32 %r9, %tid.x;
mad.lo.s32 %r10, %r127, %r424, %r9;
setp.lt.s32	%p1, %r118, 1;
@%p1 bra BB6_23;

add.s32 %r132, %r118, -1;
shr.u32 %r133, %r132, 10;
add.s32 %r11, %r133, 1;
and.b32 %r131, %r11, 3;
mov.u32 %r396, 0;
setp.eq.s32	%p2, %r131, 0;
@%p2 bra BB6_12;

setp.eq.s32	%p3, %r131, 1;
@%p3 bra BB6_9;

setp.eq.s32	%p4, %r131, 2;
@%p4 bra BB6_6;

mov.u32 %r396, 1024;
setp.ge.s32	%p5, %r10, %r118;
@%p5 bra BB6_6;

mul.wide.s32 %rd56, %r10, 4;
add.s64 %rd57, %rd1, %rd56;
ld.global.u32 %r136, [%rd57];
shl.b32 %r137, %r10, 2;
mov.u32 %r138, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r139, %r138, %r137;
st.shared.u32 [%r139], %r136;

BB6_6:
add.s32 %r13, %r10, %r396;
setp.ge.s32	%p6, %r13, %r118;
@%p6 bra BB6_8;

mul.wide.s32 %rd58, %r13, 4;
add.s64 %rd59, %rd1, %rd58;
ld.global.u32 %r140, [%rd59];
shl.b32 %r141, %r13, 2;
mov.u32 %r142, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r143, %r142, %r141;
st.shared.u32 [%r143], %r140;

BB6_8:
add.s32 %r396, %r396, 1024;

BB6_9:
add.s32 %r16, %r10, %r396;
setp.ge.s32	%p7, %r16, %r118;
@%p7 bra BB6_11;

mul.wide.s32 %rd60, %r16, 4;
add.s64 %rd61, %rd1, %rd60;
ld.global.u32 %r144, [%rd61];
shl.b32 %r145, %r16, 2;
mov.u32 %r146, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r147, %r146, %r145;
st.shared.u32 [%r147], %r144;

BB6_11:
add.s32 %r396, %r396, 1024;

BB6_12:
setp.lt.u32	%p8, %r11, 4;
@%p8 bra BB6_23;

add.s32 %r399, %r10, %r396;
shl.b32 %r148, %r399, 2;
mov.u32 %r149, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r400, %r149, %r148;
mul.wide.s32 %rd62, %r399, 4;
add.s64 %rd124, %rd1, %rd62;

BB6_14:
setp.ge.s32	%p9, %r399, %r118;
@%p9 bra BB6_16;

ld.global.u32 %r150, [%rd124];
st.shared.u32 [%r400], %r150;

BB6_16:
add.s32 %r151, %r399, 1024;
setp.ge.s32	%p10, %r151, %r118;
@%p10 bra BB6_18;

ld.global.u32 %r152, [%rd124+4096];
st.shared.u32 [%r400+4096], %r152;

BB6_18:
add.s32 %r153, %r399, 2048;
setp.ge.s32	%p11, %r153, %r118;
@%p11 bra BB6_20;

ld.global.u32 %r154, [%rd124+8192];
st.shared.u32 [%r400+8192], %r154;

BB6_20:
add.s32 %r155, %r399, 3072;
setp.ge.s32	%p12, %r155, %r118;
@%p12 bra BB6_22;

ld.global.u32 %r156, [%rd124+12288];
st.shared.u32 [%r400+12288], %r156;

BB6_22:
add.s32 %r396, %r396, 4096;
add.s64 %rd124, %rd124, 16384;
add.s32 %r399, %r399, 4096;
setp.lt.s32	%p13, %r396, %r118;
add.s32 %r400, %r400, 16384;
@%p13 bra BB6_14;

BB6_23:
bar.sync 0;
add.s32 %r402, %r424, %r6;
setp.ge.s32	%p14, %r402, %r7;
@%p14 bra BB6_42;

cvta.to.global.u64 %rd9, %rd46;
add.s32 %r157, %r116, 31;
shr.s32 %r158, %r157, 31;
shr.u32 %r159, %r158, 27;
add.s32 %r160, %r157, %r159;
and.b32 %r28, %r160, -32;
mad.lo.s32 %r161, %r3, %r1, %r9;
mul.wide.s32 %rd63, %r161, 4;
add.s64 %rd10, %rd3, %rd63;
add.s32 %r162, %r5, %r9;
mul.wide.s32 %rd64, %r162, 4;
add.s64 %rd11, %rd3, %rd64;

BB6_25:
shl.b32 %r163, %r402, 1;
mul.wide.s32 %rd65, %r163, 4;
add.s64 %rd66, %rd9, %rd65;
add.s32 %r164, %r163, 1;
mul.wide.s32 %rd67, %r164, 4;
add.s64 %rd68, %rd9, %rd67;
ld.global.u32 %r30, [%rd68];
add.s32 %r165, %r30, -1;
setp.lt.s32	%p15, %r165, 0;
setp.ge.s32	%p16, %r165, %r118;
or.pred %p17, %p15, %p16;
ld.global.u32 %r31, [%rd66];
add.s32 %r166, %r31, -1;
setp.lt.s32	%p18, %r166, 0;
or.pred %p19, %p17, %p18;
setp.ge.s32	%p20, %r166, %r117;
or.pred %p21, %p19, %p20;
@%p21 bra BB6_41;

setp.ne.s32	%p22, %r9, 0;
@%p22 bra BB6_28;

shl.b32 %r167, %r30, 2;
mov.u32 %r168, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r169, %r167, %r168;
add.s32 %r395, %r31, -1;
st.shared.u32 [%r169+-4], %r395;

BB6_28:
setp.lt.s32	%p23, %r117, 1;
@%p23 bra BB6_36;

mul.lo.s32 %r32, %r117, %r402;
mov.u32 %r403, 0;

BB6_30:
setp.lt.s32	%p24, %r116, 1;
@%p24 bra BB6_35;

add.s32 %r173, %r32, %r403;
mad.lo.s32 %r174, %r116, %r173, %r9;
mul.wide.s32 %rd69, %r174, 4;
add.s64 %rd126, %rd2, %rd69;
mad.lo.s32 %r175, %r118, %r403, %r165;
mul.lo.s32 %r176, %r116, %r175;
mul.wide.s32 %rd70, %r176, 4;
add.s64 %rd125, %rd10, %rd70;
mov.u32 %r405, 0;
mov.u32 %r404, %r9;

BB6_32:
setp.ge.s32	%p25, %r404, %r116;
@%p25 bra BB6_34;

ld.global.f32 %f51, [%rd126];
atom.global.add.f32 %f52, [%rd125], %f51;

BB6_34:
add.s32 %r405, %r405, 32;
add.s64 %rd126, %rd126, 128;
add.s32 %r404, %r404, 32;
add.s64 %rd125, %rd125, 128;
setp.lt.s32	%p26, %r405, %r28;
@%p26 bra BB6_32;

BB6_35:
add.s32 %r403, %r403, 1;
setp.lt.s32	%p27, %r403, %r117;
@%p27 bra BB6_30;

BB6_36:
setp.lt.s32	%p28, %r116, 1;
@%p28 bra BB6_41;

mad.lo.s32 %r178, %r117, %r402, %r31;
add.s32 %r179, %r178, -1;
mad.lo.s32 %r180, %r116, %r179, %r9;
mul.wide.s32 %rd71, %r180, 4;
add.s64 %rd128, %rd2, %rd71;
mul.lo.s32 %r182, %r116, %r165;
mul.wide.s32 %rd72, %r182, 4;
add.s64 %rd127, %rd11, %rd72;
mov.u32 %r407, 0;
mov.u32 %r406, %r9;

BB6_38:
setp.ge.s32	%p29, %r406, %r116;
@%p29 bra BB6_40;

ld.global.f32 %f53, [%rd128];
mul.f32 %f54, %f53, %f53;
atom.global.add.f32 %f55, [%rd127], %f54;

BB6_40:
add.s32 %r407, %r407, 32;
add.s64 %rd128, %rd128, 128;
add.s32 %r406, %r406, 32;
add.s64 %rd127, %rd127, 128;
setp.lt.s32	%p30, %r407, %r28;
@%p30 bra BB6_38;

BB6_41:
add.s32 %r402, %r402, 32;
setp.lt.s32	%p31, %r402, %r7;
@%p31 bra BB6_25;

BB6_42:
bar.sync 0;
or.b32 %r183, %r9, %r424;
add.s32 %r184, %r118, %r118;
shl.b32 %r185, %r184, 2;
mov.u32 %r186, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r45, %r186, %r185;
setp.ne.s32	%p32, %r183, 0;
@%p32 bra BB6_76;

mov.u32 %r418, 0;
@%p1 bra BB6_75;

and.b32 %r194, %r118, 3;
mov.u32 %r409, 0;
setp.eq.s32	%p34, %r194, 0;
@%p34 bra BB6_45;

setp.eq.s32	%p35, %r194, 1;
@%p35 bra BB6_47;
bra.uni BB6_48;

BB6_47:
mov.u32 %r412, %r409;
bra.uni BB6_56;

BB6_45:
mov.u32 %r418, %r409;
bra.uni BB6_60;

BB6_48:
setp.eq.s32	%p36, %r194, 2;
@%p36 bra BB6_49;
bra.uni BB6_50;

BB6_49:
mov.u32 %r408, %r409;
bra.uni BB6_52;

BB6_50:
ld.shared.u32 %r197, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r408, 1;
setp.lt.s32	%p37, %r197, 0;
@%p37 bra BB6_52;

shl.b32 %r200, %r118, 2;
add.s32 %r202, %r186, %r200;
mov.u32 %r203, 0;
st.shared.u32 [%r202], %r203;
mov.u32 %r408, 1;
mov.u32 %r409, %r408;

BB6_52:
shl.b32 %r204, %r408, 2;
add.s32 %r206, %r186, %r204;
ld.shared.u32 %r207, [%r206];
setp.lt.s32	%p38, %r207, 0;
@%p38 bra BB6_53;

add.s32 %r412, %r409, 1;
add.s32 %r208, %r409, %r118;
shl.b32 %r209, %r208, 2;
add.s32 %r211, %r186, %r209;
st.shared.u32 [%r211], %r408;
bra.uni BB6_55;

BB6_53:
mov.u32 %r412, %r409;

BB6_55:
add.s32 %r409, %r408, 1;

BB6_56:
shl.b32 %r212, %r409, 2;
add.s32 %r214, %r186, %r212;
ld.shared.u32 %r215, [%r214];
setp.lt.s32	%p39, %r215, 0;
@%p39 bra BB6_57;

add.s32 %r418, %r412, 1;
add.s32 %r216, %r412, %r118;
shl.b32 %r217, %r216, 2;
add.s32 %r219, %r186, %r217;
st.shared.u32 [%r219], %r409;
bra.uni BB6_59;

BB6_57:
mov.u32 %r418, %r412;

BB6_59:
add.s32 %r409, %r409, 1;

BB6_60:
setp.lt.u32	%p40, %r118, 4;
@%p40 bra BB6_75;

shl.b32 %r220, %r409, 2;
add.s32 %r416, %r186, %r220;

BB6_62:
ld.shared.u32 %r222, [%r416];
setp.lt.s32	%p41, %r222, 0;
@%p41 bra BB6_63;

add.s32 %r419, %r418, 1;
add.s32 %r223, %r418, %r118;
shl.b32 %r224, %r223, 2;
add.s32 %r226, %r186, %r224;
st.shared.u32 [%r226], %r409;
bra.uni BB6_65;

BB6_63:
mov.u32 %r419, %r418;

BB6_65:
ld.shared.u32 %r227, [%r416+4];
setp.lt.s32	%p42, %r227, 0;
@%p42 bra BB6_66;

add.s32 %r420, %r419, 1;
add.s32 %r228, %r419, %r118;
shl.b32 %r229, %r228, 2;
add.s32 %r231, %r186, %r229;
add.s32 %r232, %r409, 1;
st.shared.u32 [%r231], %r232;
bra.uni BB6_68;

BB6_66:
mov.u32 %r420, %r419;

BB6_68:
ld.shared.u32 %r233, [%r416+8];
setp.lt.s32	%p43, %r233, 0;
@%p43 bra BB6_69;

add.s32 %r421, %r420, 1;
add.s32 %r234, %r420, %r118;
shl.b32 %r235, %r234, 2;
add.s32 %r237, %r186, %r235;
add.s32 %r238, %r409, 2;
st.shared.u32 [%r237], %r238;
bra.uni BB6_71;

BB6_69:
mov.u32 %r421, %r420;

BB6_71:
ld.shared.u32 %r239, [%r416+12];
setp.lt.s32	%p44, %r239, 0;
@%p44 bra BB6_72;

add.s32 %r418, %r421, 1;
add.s32 %r240, %r421, %r118;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r186, %r241;
add.s32 %r244, %r409, 3;
st.shared.u32 [%r243], %r244;
bra.uni BB6_74;

BB6_72:
mov.u32 %r418, %r421;

BB6_74:
add.s32 %r409, %r409, 4;
setp.lt.s32	%p45, %r409, %r118;
add.s32 %r416, %r416, 16;
@%p45 bra BB6_62;

BB6_75:
st.shared.u32 [%r45], %r418;

BB6_76:
bar.sync 0;
add.s32 %r245, %r116, 31;
shr.s32 %r246, %r245, 31;
shr.u32 %r247, %r246, 27;
add.s32 %r248, %r245, %r247;
and.b32 %r73, %r248, -32;
ld.shared.u32 %r74, [%r45];
setp.ge.s32	%p46, %r424, %r74;
@%p46 bra BB6_141;

setp.gt.s32	%p47, %r73, 32;
add.s32 %r249, %r73, -1;
shr.u32 %r250, %r249, 5;
add.s32 %r251, %r250, 1;
selp.b32	%r75, %r251, 1, %p47;
and.b32 %r77, %r75, 3;
mul.wide.s32 %rd73, %r4, 4;
add.s64 %rd24, %rd3, %rd73;
mul.lo.s32 %r253, %r1, %r118;
mul.lo.s32 %r254, %r253, %r116;
mul.lo.s32 %r256, %r254, %r124;
mul.wide.s32 %rd74, %r256, 4;
add.s64 %rd27, %rd3, %rd74;
mul.lo.s32 %r257, %r118, %r117;
mad.lo.s32 %r258, %r257, %r116, %r256;
mul.wide.s32 %rd75, %r258, 4;
add.s64 %rd28, %rd3, %rd75;

BB6_78:
add.s32 %r259, %r424, %r118;
shl.b32 %r260, %r259, 2;
add.s32 %r262, %r186, %r260;
ld.shared.u32 %r79, [%r262];
shl.b32 %r263, %r79, 2;
add.s32 %r264, %r186, %r263;
ld.shared.u32 %r80, [%r264];
add.s32 %r265, %r79, 2;
add.s32 %r266, %r79, 1;
mul.lo.s32 %r267, %r265, %r266;
shr.u32 %r268, %r267, 31;
add.s32 %r269, %r267, %r268;
shr.s32 %r270, %r269, 1;
sub.s32 %r81, %r270, %r266;
mov.f32 %f147, 0f00000000;
st.local.v4.f32 [%rd5], {%f147, %f147, %f147, %f147};
st.local.v4.f32 [%rd5+16], {%f147, %f147, %f147, %f147};
st.local.v4.f32 [%rd5+32], {%f147, %f147, %f147, %f147};
setp.lt.s32	%p48, %r116, 1;
@%p48 bra BB6_101;

mad.lo.s32 %r82, %r79, %r116, %r9;
add.s32 %r274, %r79, %r118;
mad.lo.s32 %r83, %r274, %r116, %r9;
mov.u32 %r425, 0;
setp.eq.s32	%p49, %r77, 0;
@%p49 bra BB6_90;

setp.eq.s32	%p50, %r77, 1;
@%p50 bra BB6_87;

setp.eq.s32	%p51, %r77, 2;
@%p51 bra BB6_84;

mov.u32 %r425, 32;
setp.ge.s32	%p52, %r9, %r116;
@%p52 bra BB6_84;

add.s32 %r277, %r82, %r4;
mul.wide.s32 %rd78, %r277, 4;
add.s64 %rd79, %rd3, %rd78;
ld.global.f32 %f57, [%rd79];
st.local.f32 [%rd5], %f57;
add.s32 %r278, %r83, %r4;
mul.wide.s32 %rd80, %r278, 4;
add.s64 %rd81, %rd3, %rd80;
ld.global.f32 %f58, [%rd81];
st.local.f32 [%rd5+24], %f58;

BB6_84:
add.s32 %r279, %r425, %r9;
setp.ge.s32	%p53, %r279, %r116;
@%p53 bra BB6_86;

add.s32 %r280, %r82, %r425;
add.s32 %r281, %r280, %r4;
mul.wide.s32 %rd82, %r281, 4;
add.s64 %rd83, %rd3, %rd82;
ld.global.f32 %f59, [%rd83];
shr.u32 %r282, %r425, 5;
mul.wide.u32 %rd84, %r282, 4;
add.s64 %rd85, %rd5, %rd84;
st.local.f32 [%rd85], %f59;
add.s32 %r283, %r83, %r425;
add.s32 %r284, %r283, %r4;
mul.wide.s32 %rd86, %r284, 4;
add.s64 %rd87, %rd3, %rd86;
ld.global.f32 %f60, [%rd87];
st.local.f32 [%rd85+24], %f60;

BB6_86:
add.s32 %r425, %r425, 32;

BB6_87:
add.s32 %r285, %r425, %r9;
setp.ge.s32	%p54, %r285, %r116;
@%p54 bra BB6_89;

add.s32 %r286, %r82, %r425;
add.s32 %r287, %r286, %r4;
mul.wide.s32 %rd88, %r287, 4;
add.s64 %rd89, %rd3, %rd88;
ld.global.f32 %f61, [%rd89];
shr.s32 %r288, %r425, 31;
shr.u32 %r289, %r288, 27;
add.s32 %r290, %r425, %r289;
shr.s32 %r291, %r290, 5;
mul.wide.s32 %rd90, %r291, 4;
add.s64 %rd91, %rd5, %rd90;
st.local.f32 [%rd91], %f61;
add.s32 %r292, %r83, %r425;
add.s32 %r293, %r292, %r4;
mul.wide.s32 %rd92, %r293, 4;
add.s64 %rd93, %rd3, %rd92;
ld.global.f32 %f62, [%rd93];
st.local.f32 [%rd91+24], %f62;

BB6_89:
add.s32 %r425, %r425, 32;

BB6_90:
setp.lt.u32	%p55, %r75, 4;
@%p55 bra BB6_101;

add.s32 %r428, %r9, %r425;
mad.lo.s32 %r294, %r116, %r79, %r428;
mul.wide.s32 %rd94, %r294, 4;
add.s64 %rd129, %rd27, %rd94;
add.s32 %r295, %r118, %r79;
mad.lo.s32 %r296, %r116, %r295, %r428;
mul.wide.s32 %rd95, %r296, 4;
add.s64 %rd130, %rd27, %rd95;

BB6_92:
setp.ge.s32	%p56, %r428, %r116;
@%p56 bra BB6_94;

ld.global.f32 %f63, [%rd129];
shr.s32 %r297, %r425, 31;
shr.u32 %r298, %r297, 27;
add.s32 %r299, %r425, %r298;
shr.s32 %r300, %r299, 5;
mul.wide.s32 %rd96, %r300, 4;
add.s64 %rd97, %rd5, %rd96;
st.local.f32 [%rd97], %f63;
ld.global.f32 %f64, [%rd130];
st.local.f32 [%rd97+24], %f64;

BB6_94:
add.s32 %r301, %r428, 32;
setp.ge.s32	%p57, %r301, %r116;
@%p57 bra BB6_96;

ld.global.f32 %f65, [%rd129+128];
add.s32 %r302, %r425, 32;
shr.s32 %r303, %r302, 31;
shr.u32 %r304, %r303, 27;
add.s32 %r305, %r302, %r304;
shr.s32 %r306, %r305, 5;
mul.wide.s32 %rd98, %r306, 4;
add.s64 %rd99, %rd5, %rd98;
st.local.f32 [%rd99], %f65;
ld.global.f32 %f66, [%rd130+128];
st.local.f32 [%rd99+24], %f66;

BB6_96:
add.s32 %r307, %r428, 64;
setp.ge.s32	%p58, %r307, %r116;
@%p58 bra BB6_98;

ld.global.f32 %f67, [%rd129+256];
add.s32 %r308, %r425, 64;
shr.s32 %r309, %r308, 31;
shr.u32 %r310, %r309, 27;
add.s32 %r311, %r308, %r310;
shr.s32 %r312, %r311, 5;
mul.wide.s32 %rd100, %r312, 4;
add.s64 %rd101, %rd5, %rd100;
st.local.f32 [%rd101], %f67;
ld.global.f32 %f68, [%rd130+256];
st.local.f32 [%rd101+24], %f68;

BB6_98:
add.s32 %r313, %r428, 96;
setp.ge.s32	%p59, %r313, %r116;
@%p59 bra BB6_100;

ld.global.f32 %f69, [%rd129+384];
add.s32 %r314, %r425, 96;
shr.s32 %r315, %r314, 31;
shr.u32 %r316, %r315, 27;
add.s32 %r317, %r314, %r316;
shr.s32 %r318, %r317, 5;
mul.wide.s32 %rd102, %r318, 4;
add.s64 %rd103, %rd5, %rd102;
st.local.f32 [%rd103], %f69;
ld.global.f32 %f70, [%rd130+384];
st.local.f32 [%rd103+24], %f70;

BB6_100:
add.s32 %r425, %r425, 128;
add.s32 %r428, %r428, 128;
setp.lt.s32	%p60, %r425, %r73;
add.s64 %rd129, %rd129, 512;
add.s64 %rd130, %rd130, 512;
@%p60 bra BB6_92;

BB6_101:
setp.lt.s32	%p61, %r424, 1;
@%p61 bra BB6_113;

mul.lo.s32 %r94, %r118, %r80;
mov.u32 %r430, 0;

BB6_103:
add.s32 %r320, %r430, %r118;
shl.b32 %r321, %r320, 2;
add.s32 %r323, %r186, %r321;
ld.shared.u32 %r96, [%r323];
mov.f32 %f128, 0f00000000;
@%p48 bra BB6_110;

shl.b32 %r325, %r96, 2;
add.s32 %r327, %r186, %r325;
ld.shared.u32 %r97, [%r327];
add.s32 %r328, %r94, %r96;
mad.lo.s32 %r329, %r116, %r328, %r9;
mul.wide.s32 %rd104, %r329, 4;
add.s64 %rd131, %rd24, %rd104;
mov.f32 %f72, 0f00000000;
mov.u32 %r432, 0;
mov.u32 %r431, %r9;
mov.f32 %f128, %f72;

BB6_105:
setp.ge.s32	%p63, %r431, %r116;
mov.f32 %f126, %f72;
mov.f32 %f127, %f72;
@%p63 bra BB6_109;

setp.eq.s32	%p64, %r97, 0;
ld.global.f32 %f126, [%rd131];
shr.s32 %r330, %r432, 31;
shr.u32 %r331, %r330, 27;
add.s32 %r332, %r432, %r331;
shr.s32 %r333, %r332, 5;
mul.wide.s32 %rd105, %r333, 4;
add.s64 %rd38, %rd5, %rd105;
@%p64 bra BB6_108;
bra.uni BB6_107;

BB6_108:
ld.local.f32 %f127, [%rd38];
bra.uni BB6_109;

BB6_107:
ld.local.f32 %f127, [%rd38+24];

BB6_109:
fma.rn.f32 %f128, %f126, %f127, %f128;
add.s32 %r431, %r431, 32;
add.s64 %rd131, %rd131, 128;
add.s32 %r432, %r432, 32;
setp.lt.s32	%p65, %r432, %r73;
@%p65 bra BB6_105;

BB6_110:
bar.warp.sync -1;
mov.b32 %r334, %f128;
mov.u32 %r335, 31;
mov.u32 %r336, 1;
mov.u32 %r337, -1;
shfl.sync.bfly.b32 %r338|%p66, %r334, %r336, %r335, %r337;
mov.b32 %f75, %r338;
add.f32 %f76, %f128, %f75;
mov.b32 %r339, %f76;
mov.u32 %r340, 2;
shfl.sync.bfly.b32 %r341|%p67, %r339, %r340, %r335, %r337;
mov.b32 %f77, %r341;
add.f32 %f78, %f76, %f77;
mov.b32 %r342, %f78;
mov.u32 %r343, 4;
shfl.sync.bfly.b32 %r344|%p68, %r342, %r343, %r335, %r337;
mov.b32 %f79, %r344;
add.f32 %f80, %f78, %f79;
mov.b32 %r345, %f80;
mov.u32 %r346, 8;
shfl.sync.bfly.b32 %r347|%p69, %r345, %r346, %r335, %r337;
mov.b32 %f81, %r347;
add.f32 %f82, %f80, %f81;
mov.b32 %r348, %f82;
mov.u32 %r349, 16;
shfl.sync.bfly.b32 %r350|%p70, %r348, %r349, %r335, %r337;
mov.b32 %f83, %r350;
add.f32 %f9, %f82, %f83;
setp.ne.s32	%p71, %r9, 0;
@%p71 bra BB6_112;

add.s32 %r351, %r96, %r81;
add.s32 %r352, %r351, %r2;
mul.wide.s32 %rd106, %r352, 4;
add.s64 %rd107, %rd4, %rd106;
st.global.f32 [%rd107], %f9;

BB6_112:
bar.warp.sync -1;
add.s32 %r430, %r430, 1;
setp.lt.s32	%p72, %r430, %r424;
@%p72 bra BB6_103;

BB6_113:
@%p48 bra BB6_138;

mad.lo.s32 %r356, %r80, %r118, %r79;
mad.lo.s32 %r103, %r356, %r116, %r9;
mad.lo.s32 %r104, %r79, %r116, %r9;
mov.f32 %f147, 0f00000000;
mov.u32 %r435, 0;
setp.eq.s32	%p74, %r77, 0;
@%p74 bra BB6_127;

setp.eq.s32	%p75, %r77, 1;
@%p75 bra BB6_116;
bra.uni BB6_117;

BB6_116:
mov.f32 %f134, %f147;
bra.uni BB6_124;

BB6_117:
setp.eq.s32	%p76, %r77, 2;
mov.f32 %f131, %f147;
@%p76 bra BB6_121;

mov.f32 %f129, 0f00000000;
setp.ge.s32	%p77, %r9, %r116;
mov.f32 %f130, %f129;
@%p77 bra BB6_120;

add.s32 %r357, %r103, %r4;
mul.wide.s32 %rd108, %r357, 4;
add.s64 %rd109, %rd3, %rd108;
ld.global.f32 %f129, [%rd109];
add.s32 %r358, %r104, %r5;
mul.wide.s32 %rd110, %r358, 4;
add.s64 %rd111, %rd3, %rd110;
ld.global.f32 %f130, [%rd111];

BB6_120:
mul.f32 %f90, %f129, %f129;
sub.f32 %f91, %f90, %f130;
fma.rn.f32 %f131, %f91, 0f3F000000, 0f00000000;
mov.u32 %r435, 32;

BB6_121:
add.s32 %r360, %r435, %r9;
setp.ge.s32	%p78, %r360, %r116;
mov.f32 %f132, %f147;
mov.f32 %f133, %f147;
@%p78 bra BB6_123;

add.s32 %r361, %r103, %r435;
add.s32 %r362, %r361, %r4;
mul.wide.s32 %rd112, %r362, 4;
add.s64 %rd113, %rd3, %rd112;
ld.global.f32 %f132, [%rd113];
add.s32 %r363, %r104, %r435;
add.s32 %r364, %r363, %r5;
mul.wide.s32 %rd114, %r364, 4;
add.s64 %rd115, %rd3, %rd114;
ld.global.f32 %f133, [%rd115];

BB6_123:
mul.f32 %f94, %f132, %f132;
sub.f32 %f95, %f94, %f133;
fma.rn.f32 %f134, %f95, 0f3F000000, %f131;
add.s32 %r435, %r435, 32;

BB6_124:
add.s32 %r365, %r435, %r9;
setp.ge.s32	%p79, %r365, %r116;
mov.f32 %f136, %f147;
@%p79 bra BB6_126;

add.s32 %r366, %r103, %r435;
add.s32 %r367, %r366, %r4;
mul.wide.s32 %rd116, %r367, 4;
add.s64 %rd117, %rd3, %rd116;
ld.global.f32 %f147, [%rd117];
add.s32 %r368, %r104, %r435;
add.s32 %r369, %r368, %r5;
mul.wide.s32 %rd118, %r369, 4;
add.s64 %rd119, %rd3, %rd118;
ld.global.f32 %f136, [%rd119];

BB6_126:
mul.f32 %f98, %f147, %f147;
sub.f32 %f99, %f98, %f136;
fma.rn.f32 %f147, %f99, 0f3F000000, %f134;
add.s32 %r435, %r435, 32;

BB6_127:
setp.lt.u32	%p80, %r75, 4;
@%p80 bra BB6_138;

add.s32 %r436, %r9, %r435;
mad.lo.s32 %r370, %r118, %r80, %r79;
mad.lo.s32 %r371, %r116, %r370, %r436;
mul.wide.s32 %rd120, %r371, 4;
add.s64 %rd132, %rd27, %rd120;
mad.lo.s32 %r372, %r116, %r79, %r436;
mul.wide.s32 %rd121, %r372, 4;
add.s64 %rd133, %rd28, %rd121;

BB6_129:
mov.f32 %f145, 0f00000000;
setp.ge.s32	%p81, %r436, %r116;
mov.f32 %f139, %f145;
mov.f32 %f140, %f145;
@%p81 bra BB6_131;

ld.global.f32 %f139, [%rd132];
ld.global.f32 %f140, [%rd133];

BB6_131:
mul.f32 %f104, %f139, %f139;
sub.f32 %f105, %f104, %f140;
fma.rn.f32 %f33, %f105, 0f3F000000, %f147;
add.s32 %r373, %r436, 32;
setp.ge.s32	%p82, %r373, %r116;
mov.f32 %f141, %f145;
mov.f32 %f142, %f145;
@%p82 bra BB6_133;

ld.global.f32 %f141, [%rd132+128];
ld.global.f32 %f142, [%rd133+128];

BB6_133:
mul.f32 %f108, %f141, %f141;
sub.f32 %f109, %f108, %f142;
fma.rn.f32 %f38, %f109, 0f3F000000, %f33;
add.s32 %r374, %r436, 64;
setp.ge.s32	%p83, %r374, %r116;
mov.f32 %f143, %f145;
mov.f32 %f144, %f145;
@%p83 bra BB6_135;

ld.global.f32 %f143, [%rd132+256];
ld.global.f32 %f144, [%rd133+256];

BB6_135:
mul.f32 %f112, %f143, %f143;
sub.f32 %f113, %f112, %f144;
fma.rn.f32 %f43, %f113, 0f3F000000, %f38;
add.s32 %r375, %r436, 96;
setp.ge.s32	%p84, %r375, %r116;
mov.f32 %f146, %f145;
@%p84 bra BB6_137;

ld.global.f32 %f145, [%rd132+384];
ld.global.f32 %f146, [%rd133+384];

BB6_137:
mul.f32 %f114, %f145, %f145;
sub.f32 %f115, %f114, %f146;
fma.rn.f32 %f147, %f115, 0f3F000000, %f43;
add.s32 %r436, %r436, 128;
add.s32 %r435, %r435, 128;
setp.lt.s32	%p85, %r435, %r73;
add.s64 %rd132, %rd132, 512;
add.s64 %rd133, %rd133, 512;
@%p85 bra BB6_129;

BB6_138:
bar.warp.sync -1;
mov.b32 %r376, %f147;
mov.u32 %r377, 31;
mov.u32 %r378, 1;
mov.u32 %r379, -1;
shfl.sync.bfly.b32 %r380|%p86, %r376, %r378, %r377, %r379;
mov.b32 %f116, %r380;
add.f32 %f117, %f147, %f116;
mov.b32 %r381, %f117;
mov.u32 %r382, 2;
shfl.sync.bfly.b32 %r383|%p87, %r381, %r382, %r377, %r379;
mov.b32 %f118, %r383;
add.f32 %f119, %f117, %f118;
mov.b32 %r384, %f119;
mov.u32 %r385, 4;
shfl.sync.bfly.b32 %r386|%p88, %r384, %r385, %r377, %r379;
mov.b32 %f120, %r386;
add.f32 %f121, %f119, %f120;
mov.b32 %r387, %f121;
mov.u32 %r388, 8;
shfl.sync.bfly.b32 %r389|%p89, %r387, %r388, %r377, %r379;
mov.b32 %f122, %r389;
add.f32 %f123, %f121, %f122;
mov.b32 %r390, %f123;
mov.u32 %r391, 16;
shfl.sync.bfly.b32 %r392|%p90, %r390, %r391, %r377, %r379;
mov.b32 %f124, %r392;
add.f32 %f50, %f123, %f124;
setp.ne.s32	%p91, %r9, 0;
@%p91 bra BB6_140;

add.s32 %r393, %r81, %r79;
add.s32 %r394, %r393, %r2;
mul.wide.s32 %rd122, %r394, 4;
add.s64 %rd123, %rd4, %rd122;
st.global.f32 [%rd123], %f50;

BB6_140:
bar.warp.sync -1;
add.s32 %r424, %r424, 32;
setp.lt.s32	%p92, %r424, %r74;
@%p92 bra BB6_78;

BB6_141:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT_(
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_2,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_5
)
{
.reg .pred %p<22>;
.reg .b32 %r<64>;
.reg .b64 %rd<40>;


ld.param.u64 %rd10, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_0];
ld.param.u32 %r35, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_1];
ld.param.u32 %r36, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_2];
ld.param.u64 %rd11, [_ZN8nvinfer119sparse_fipnn_shared20ComputeBatchBoundaryI6__halfEEvPKiiiPiiPT__param_3];
cvta.to.global.u64 %rd1, %rd10;
cvta.to.global.u64 %rd2, %rd11;
mov.u32 %r37, %ntid.x;
mov.u32 %r38, %ctaid.x;
mov.u32 %r39, %tid.x;
mad.lo.s32 %r1, %r37, %r38, %r39;
add.s32 %r40, %r36, 1;
setp.ge.s32	%p1, %r1, %r40;
@%p1 bra BB7_2;

mul.wide.s32 %rd12, %r1, 4;
add.s64 %rd13, %rd2, %rd12;
mov.u32 %r41, 0;
st.global.u32 [%rd13], %r41;

BB7_2:
bar.sync 0;
setp.ge.s32	%p2, %r1, %r35;
@%p2 bra BB7_34;

setp.gt.s32	%p3, %r1, 0;
@%p3 bra BB7_24;
bra.uni BB7_4;

BB7_24:
mul.wide.s32 %rd29, %r1, 4;
add.s64 %rd30, %rd1, %rd29;
ld.global.u32 %r23, [%rd30];
ld.global.u32 %r24, [%rd30+-4];
setp.le.s32	%p16, %r23, %r24;
@%p16 bra BB7_34;

sub.s32 %r25, %r23, %r24;
and.b32 %r26, %r25, 3;
setp.eq.s32	%p17, %r26, 0;
@%p17 bra BB7_31;

setp.eq.s32	%p18, %r26, 1;
@%p18 bra BB7_30;

setp.eq.s32	%p19, %r26, 2;
@%p19 bra BB7_29;

mul.wide.s32 %rd31, %r23, 4;
add.s64 %rd32, %rd2, %rd31;
st.global.u32 [%rd32], %r1;
add.s32 %r23, %r23, -1;

BB7_29:
mul.wide.s32 %rd33, %r23, 4;
add.s64 %rd34, %rd2, %rd33;
st.global.u32 [%rd34], %r1;
add.s32 %r23, %r23, -1;

BB7_30:
mul.wide.s32 %rd35, %r23, 4;
add.s64 %rd36, %rd2, %rd35;
st.global.u32 [%rd36], %r1;
add.s32 %r23, %r23, -1;

BB7_31:
setp.lt.u32	%p20, %r25, 4;
@%p20 bra BB7_34;

mul.wide.s32 %rd37, %r23, 4;
add.s64 %rd39, %rd2, %rd37;

BB7_33:
st.global.u32 [%rd39], %r1;
st.global.u32 [%rd39+-4], %r1;
st.global.u32 [%rd39+-8], %r1;
st.global.u32 [%rd39+-12], %r1;
add.s64 %rd39, %rd39, -16;
add.s32 %r23, %r23, -4;
setp.gt.s32	%p21, %r23, %r24;
@%p21 bra BB7_33;
bra.uni BB7_34;

BB7_4:
ld.global.u32 %r2, [%rd1];
setp.lt.s32	%p4, %r2, 0;
@%p4 bra BB7_14;

add.s32 %r3, %r2, 1;
and.b32 %r45, %r3, 3;
mov.u32 %r54, 0;
setp.eq.s32	%p5, %r45, 0;
@%p5 bra BB7_11;

setp.eq.s32	%p6, %r45, 1;
mov.u32 %r53, %r54;
@%p6 bra BB7_10;

setp.eq.s32	%p7, %r45, 2;
mov.u32 %r52, %r54;
@%p7 bra BB7_9;

mov.u32 %r47, 0;
st.global.u32 [%rd2], %r47;
mov.u32 %r52, 1;

BB7_9:
mul.wide.u32 %rd14, %r52, 4;
add.s64 %rd15, %rd2, %rd14;
st.global.u32 [%rd15], %r54;
add.s32 %r53, %r52, 1;

BB7_10:
mul.wide.s32 %rd16, %r53, 4;
add.s64 %rd17, %rd2, %rd16;
st.global.u32 [%rd17], %r54;
add.s32 %r54, %r53, 1;

BB7_11:
setp.lt.u32	%p8, %r3, 4;
@%p8 bra BB7_14;

add.s32 %r55, %r54, -1;
mul.wide.s32 %rd18, %r54, 4;
add.s64 %rd38, %rd2, %rd18;

BB7_13:
mov.u64 %rd19, 0;
st.global.u32 [%rd38+4], %rd19;
st.global.u32 [%rd38], %rd19;
st.global.u32 [%rd38+12], %rd19;
st.global.u32 [%rd38+8], %rd19;
add.s64 %rd38, %rd38, 16;
add.s32 %r55, %r55, 4;
setp.lt.s32	%p9, %r55, %r2;
@%p9 bra BB7_13;

BB7_14:
add.s32 %r50, %r35, -1;
mul.wide.s32 %rd20, %r50, 4;
add.s64 %rd21, %rd1, %rd20;
add.s32 %r56, %r36, -1;
ld.global.u32 %r13, [%rd21];
mul.wide.s32 %rd22, %r36, 4;
add.s64 %rd6, %rd2, %rd22;
setp.le.s32	%p10, %r56, %r13;
@%p10 bra BB7_23;

sub.s32 %r14, %r56, %r13;
and.b32 %r51, %r14, 3;
setp.eq.s32	%p11, %r51, 0;
@%p11 bra BB7_21;

setp.eq.s32	%p12, %r51, 1;
@%p12 bra BB7_20;

setp.eq.s32	%p13, %r51, 2;
@%p13 bra BB7_19;

st.global.u32 [%rd6+-4], %r35;
add.s32 %r56, %r36, -2;

BB7_19:
mul.wide.s32 %rd23, %r56, 4;
add.s64 %rd24, %rd2, %rd23;
st.global.u32 [%rd24], %r35;
add.s32 %r56, %r56, -1;

BB7_20:
mul.wide.s32 %rd25, %r56, 4;
add.s64 %rd26, %rd2, %rd25;
st.global.u32 [%rd26], %r35;
add.s32 %r56, %r56, -1;

BB7_21:
setp.lt.u32	%p14, %r14, 4;
@%p14 bra BB7_23;

BB7_22:
mul.wide.s32 %rd27, %r56, 4;
add.s64 %rd28, %rd2, %rd27;
st.global.u32 [%rd28], %r35;
st.global.u32 [%rd28+-4], %r35;
st.global.u32 [%rd28+-8], %r35;
st.global.u32 [%rd28+-12], %r35;
add.s32 %r56, %r56, -4;
setp.gt.s32	%p15, %r56, %r13;
@%p15 bra BB7_22;

BB7_23:
st.global.u32 [%rd6], %r35;

BB7_34:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_4,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_11
)
{
.reg .pred %p<66>;
.reg .b16 %rs<138>;
.reg .b32 %r<518>;
.reg .f64 %fd<2>;
.reg .b64 %rd<113>;


ld.param.u32 %r144, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_0];
ld.param.u32 %r145, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_1];
ld.param.u32 %r146, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_2];
ld.param.u32 %r147, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_3];
ld.param.u32 %r148, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_4];
ld.param.u32 %r149, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_5];
ld.param.u64 %rd21, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_6];
ld.param.u64 %rd22, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_7];
ld.param.u64 %rd17, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_8];
ld.param.u64 %rd18, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_9];
ld.param.u64 %rd19, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_10];
ld.param.u64 %rd20, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_11];
cvta.to.global.u64 %rd1, %rd22;
add.s32 %r1, %r147, 1;
mul.lo.s32 %r150, %r1, %r147;
shr.u32 %r151, %r150, 31;
add.s32 %r152, %r150, %r151;
shr.s32 %r2, %r152, 1;
cvt.s64.s32	%rd23, %r147;
add.s64 %rd2, %rd23, 1;
add.s64 %rd3, %rd2, %rd2;
cvt.u32.u64	%r153, %rd3;
shl.b32 %r154, %r153, 2;
mov.u32 %r155, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r3, %r155, %r154;
mul.lo.s32 %r156, %r146, %r145;
mul.lo.s32 %r157, %r156, %r147;
cvt.s64.s32	%rd4, %r157;
mul.lo.s32 %r158, %r147, %r145;
cvt.s64.s32	%rd24, %r158;
add.s64 %rd5, %rd4, %rd24;
mov.u32 %r4, %ntid.x;
mov.u32 %r507, %tid.y;
mov.u32 %r6, %tid.x;
mad.lo.s32 %r513, %r4, %r507, %r6;
mov.u32 %r8, %ntid.y;
mul.lo.s32 %r9, %r4, %r8;
mov.u32 %r10, %ctaid.y;
div.u32 %r11, %r145, %r4;
cvta.to.global.u64 %rd25, %rd21;
ld.global.u32 %r12, [%rd25];
setp.ge.s32	%p1, %r513, %r148;
@%p1 bra BB8_5;

mov.u32 %r468, %r513;

BB8_2:
mov.u32 %r159, 0;

	cvt.rn.f16.s32 %rs18, %r159;

	shl.b32 %r160, %r468, 1;
add.s32 %r161, %r3, %r160;
st.shared.u16 [%r161], %rs18;
setp.ge.s32	%p2, %r468, %r147;
@%p2 bra BB8_4;

shl.b32 %r162, %r468, 2;
add.s32 %r164, %r155, %r162;
mov.u32 %r165, -1;
st.shared.u32 [%r164], %r165;

BB8_4:
add.s32 %r468, %r468, %r9;
setp.lt.s32	%p3, %r468, %r148;
@%p3 bra BB8_2;

BB8_5:
setp.ge.s32	%p4, %r513, %r2;
@%p4 bra BB8_8;

mov.u32 %r469, %r513;

BB8_7:
mov.u32 %r166, 0;

	cvt.rn.f16.s32 %rs19, %r166;

	cvt.u64.u32	%rd26, %r469;
add.s64 %rd27, %rd26, %rd5;
cvt.u32.u64	%r167, %rd27;
shl.b32 %r168, %r167, 1;
add.s32 %r169, %r3, %r168;
st.shared.u16 [%r169], %rs19;
add.s32 %r469, %r469, %r9;
setp.lt.s32	%p5, %r469, %r2;
@%p5 bra BB8_7;

BB8_8:
bar.sync 0;
setp.ge.s32	%p6, %r507, %r12;
@%p6 bra BB8_27;

cvta.to.global.u64 %rd6, %rd17;
shl.b64 %rd28, %rd3, 2;
{
.reg .u64 %temp; 
cvt.u64.u32 %temp, %r155;
cvta.shared.u64 %rd29, %temp;
}
add.s64 %rd7, %rd29, %rd28;
cvt.s64.s32	%rd8, %r6;
add.s64 %rd9, %rd8, %rd4;
mad.lo.s32 %r17, %r10, %r145, %r6;
mov.u32 %r470, %r507;

BB8_10:
shl.b32 %r171, %r470, 1;
mul.wide.s32 %rd30, %r171, 4;
add.s64 %rd31, %rd6, %rd30;
ld.global.u32 %r172, [%rd31+4];
add.s32 %r19, %r172, -1;
setp.lt.s32	%p7, %r19, 0;
setp.ge.s32	%p8, %r19, %r147;
or.pred %p9, %p7, %p8;
ld.global.u32 %r20, [%rd31];
add.s32 %r173, %r20, -1;
setp.lt.s32	%p10, %r173, 0;
or.pred %p11, %p9, %p10;
setp.ge.s32	%p12, %r173, %r146;
or.pred %p13, %p11, %p12;
@%p13 bra BB8_26;

setp.ne.s32	%p14, %r6, 0;
@%p14 bra BB8_13;

shl.b32 %r174, %r19, 2;
add.s32 %r176, %r155, %r174;
add.s32 %r459, %r20, -1;
st.shared.u32 [%r176], %r459;

BB8_13:
setp.lt.s32	%p15, %r146, 1;
@%p15 bra BB8_21;

mul.lo.s32 %r21, %r146, %r470;
mov.u32 %r471, 0;

BB8_15:
setp.lt.s32	%p16, %r11, 1;
@%p16 bra BB8_20;

mad.lo.s32 %r181, %r471, %r147, %r19;
mul.lo.s32 %r182, %r181, %r145;
cvt.s64.s32	%rd32, %r182;
add.s64 %rd10, %rd32, %rd8;
add.s32 %r183, %r21, %r471;
mad.lo.s32 %r472, %r144, %r183, %r17;
mov.u32 %r474, 0;
mov.u32 %r473, %r17;
mov.u32 %r475, %r474;

BB8_17:
setp.ge.u32	%p17, %r473, %r144;
@%p17 bra BB8_19;

mul.wide.s32 %rd34, %r472, 2;
add.s64 %rd35, %rd1, %rd34;
ld.global.u16 %rs21, [%rd35];
cvt.u64.u32	%rd36, %r474;
add.s64 %rd37, %rd10, %rd36;
shl.b64 %rd38, %rd37, 1;
add.s64 %rd33, %rd7, %rd38;

	{ atom.add.noftz.f16 %rs20,[%rd33],%rs21; }



BB8_19:
add.s32 %r475, %r475, 1;
add.s32 %r474, %r474, %r4;
add.s32 %r473, %r473, %r4;
add.s32 %r472, %r472, %r4;
setp.lt.s32	%p18, %r475, %r11;
@%p18 bra BB8_17;

BB8_20:
add.s32 %r471, %r471, 1;
setp.lt.s32	%p19, %r471, %r146;
@%p19 bra BB8_15;

BB8_21:
setp.lt.s32	%p20, %r11, 1;
@%p20 bra BB8_26;

mul.lo.s32 %r186, %r19, %r145;
cvt.s64.s32	%rd39, %r186;
add.s64 %rd11, %rd9, %rd39;
mad.lo.s32 %r187, %r146, %r470, %r20;
add.s32 %r188, %r187, -1;
mad.lo.s32 %r476, %r144, %r188, %r17;
mov.u32 %r478, 0;
mov.u32 %r477, %r17;
mov.u32 %r479, %r478;

BB8_23:
setp.ge.u32	%p21, %r477, %r144;
@%p21 bra BB8_25;

mul.wide.s32 %rd41, %r476, 2;
add.s64 %rd42, %rd1, %rd41;
ld.global.u16 %rs23, [%rd42];

	{mul.f16 %rs22,%rs23,%rs23;
}

	cvt.u64.u32	%rd43, %r478;
add.s64 %rd44, %rd11, %rd43;
shl.b64 %rd45, %rd44, 1;
add.s64 %rd40, %rd7, %rd45;

	{ atom.add.noftz.f16 %rs25,[%rd40],%rs22; }



BB8_25:
add.s32 %r479, %r479, 1;
add.s32 %r478, %r478, %r4;
add.s32 %r477, %r477, %r4;
add.s32 %r476, %r476, %r4;
setp.lt.s32	%p22, %r479, %r11;
@%p22 bra BB8_23;

BB8_26:
add.s32 %r470, %r470, %r8;
setp.lt.s32	%p23, %r470, %r12;
@%p23 bra BB8_10;

BB8_27:
bar.sync 0;
or.b32 %r189, %r6, %r507;
cvt.u64.u32	%rd46, %r147;
add.s64 %rd47, %rd2, %rd46;
cvt.u32.u64	%r190, %rd47;
shl.b32 %r191, %r190, 2;
add.s32 %r43, %r155, %r191;
setp.ne.s32	%p24, %r189, 0;
@%p24 bra BB8_61;

mov.u32 %r490, 0;
setp.lt.s32	%p25, %r147, 1;
@%p25 bra BB8_60;

and.b32 %r200, %r147, 3;
mov.u32 %r481, 0;
setp.eq.s32	%p26, %r200, 0;
@%p26 bra BB8_30;

setp.eq.s32	%p27, %r200, 1;
@%p27 bra BB8_32;
bra.uni BB8_33;

BB8_32:
mov.u32 %r484, %r481;
bra.uni BB8_41;

BB8_30:
mov.u32 %r490, %r481;
bra.uni BB8_45;

BB8_33:
setp.eq.s32	%p28, %r200, 2;
@%p28 bra BB8_34;
bra.uni BB8_35;

BB8_34:
mov.u32 %r480, %r481;
bra.uni BB8_37;

BB8_35:
ld.shared.u32 %r203, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r480, 1;
setp.lt.s32	%p29, %r203, 0;
@%p29 bra BB8_37;

shl.b32 %r206, %r1, 2;
add.s32 %r208, %r155, %r206;
mov.u32 %r209, 0;
st.shared.u32 [%r208], %r209;
mov.u32 %r480, 1;
mov.u32 %r481, %r480;

BB8_37:
shl.b32 %r210, %r480, 2;
add.s32 %r212, %r155, %r210;
ld.shared.u32 %r213, [%r212];
setp.lt.s32	%p30, %r213, 0;
@%p30 bra BB8_38;

add.s32 %r484, %r481, 1;
cvt.u64.u32	%rd48, %r481;
add.s64 %rd49, %rd48, %rd2;
cvt.u32.u64	%r214, %rd49;
shl.b32 %r215, %r214, 2;
add.s32 %r217, %r155, %r215;
st.shared.u32 [%r217], %r480;
bra.uni BB8_40;

BB8_38:
mov.u32 %r484, %r481;

BB8_40:
add.s32 %r481, %r480, 1;

BB8_41:
shl.b32 %r218, %r481, 2;
add.s32 %r220, %r155, %r218;
ld.shared.u32 %r221, [%r220];
setp.lt.s32	%p31, %r221, 0;
@%p31 bra BB8_42;

add.s32 %r490, %r484, 1;
cvt.u64.u32	%rd50, %r484;
add.s64 %rd51, %rd50, %rd2;
cvt.u32.u64	%r222, %rd51;
shl.b32 %r223, %r222, 2;
add.s32 %r225, %r155, %r223;
st.shared.u32 [%r225], %r481;
bra.uni BB8_44;

BB8_42:
mov.u32 %r490, %r484;

BB8_44:
add.s32 %r481, %r481, 1;

BB8_45:
setp.lt.u32	%p32, %r147, 4;
@%p32 bra BB8_60;

shl.b32 %r226, %r481, 2;
add.s32 %r488, %r155, %r226;

BB8_47:
ld.shared.u32 %r228, [%r488];
setp.lt.s32	%p33, %r228, 0;
@%p33 bra BB8_48;

add.s32 %r491, %r490, 1;
cvt.u64.u32	%rd52, %r490;
add.s64 %rd53, %rd52, %rd2;
cvt.u32.u64	%r229, %rd53;
shl.b32 %r230, %r229, 2;
add.s32 %r232, %r155, %r230;
st.shared.u32 [%r232], %r481;
bra.uni BB8_50;

BB8_48:
mov.u32 %r491, %r490;

BB8_50:
ld.shared.u32 %r233, [%r488+4];
setp.lt.s32	%p34, %r233, 0;
@%p34 bra BB8_51;

add.s32 %r492, %r491, 1;
cvt.u64.u32	%rd54, %r491;
add.s64 %rd55, %rd54, %rd2;
cvt.u32.u64	%r234, %rd55;
shl.b32 %r235, %r234, 2;
add.s32 %r237, %r155, %r235;
add.s32 %r238, %r481, 1;
st.shared.u32 [%r237], %r238;
bra.uni BB8_53;

BB8_51:
mov.u32 %r492, %r491;

BB8_53:
ld.shared.u32 %r239, [%r488+8];
setp.lt.s32	%p35, %r239, 0;
@%p35 bra BB8_54;

add.s32 %r493, %r492, 1;
cvt.u64.u32	%rd56, %r492;
add.s64 %rd57, %rd56, %rd2;
cvt.u32.u64	%r240, %rd57;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r155, %r241;
add.s32 %r244, %r481, 2;
st.shared.u32 [%r243], %r244;
bra.uni BB8_56;

BB8_54:
mov.u32 %r493, %r492;

BB8_56:
ld.shared.u32 %r245, [%r488+12];
setp.lt.s32	%p36, %r245, 0;
@%p36 bra BB8_57;

add.s32 %r490, %r493, 1;
cvt.u64.u32	%rd58, %r493;
add.s64 %rd59, %rd58, %rd2;
cvt.u32.u64	%r246, %rd59;
shl.b32 %r247, %r246, 2;
add.s32 %r249, %r155, %r247;
add.s32 %r250, %r481, 3;
st.shared.u32 [%r249], %r250;
bra.uni BB8_59;

BB8_57:
mov.u32 %r490, %r493;

BB8_59:
add.s32 %r481, %r481, 4;
setp.lt.s32	%p37, %r481, %r147;
add.s32 %r488, %r488, 16;
@%p37 bra BB8_47;

BB8_60:
st.shared.u32 [%r43], %r490;

BB8_61:
bar.sync 0;
ld.shared.u32 %r71, [%r43];
setp.ge.s32	%p38, %r507, %r71;
@%p38 bra BB8_81;

shl.b32 %r251, %r147, 3;
add.s32 %r253, %r251, %r155;
add.s32 %r254, %r253, 8;
mad.lo.s32 %r72, %r6, 2, %r254;
shl.b32 %r73, %r145, 1;
mul.lo.s32 %r74, %r147, %r146;
shl.b32 %r75, %r4, 1;
mad.lo.s32 %r76, %r10, %r145, %r6;

	{mov.u32 %r270, WARP_SZ;
}

	shl.b32 %r309, %r270, 8;
add.s32 %r310, %r309, -8192;
or.b32 %r274, %r310, 31;
mov.u32 %r496, %r507;

BB8_63:
cvt.u64.u32	%rd60, %r496;
add.s64 %rd61, %rd60, %rd2;
cvt.u32.u64	%r256, %rd61;
shl.b32 %r257, %r256, 2;
add.s32 %r259, %r155, %r257;
ld.shared.u32 %r78, [%r259];
shl.b32 %r260, %r78, 2;
add.s32 %r261, %r155, %r260;
ld.shared.u32 %r79, [%r261];
add.s32 %r262, %r78, 1;
mul.lo.s32 %r263, %r262, %r78;
shr.u32 %r264, %r263, 31;
add.s32 %r265, %r263, %r264;
shr.s32 %r80, %r265, 1;
mov.u32 %r501, 0;

	cvt.rn.f16.s32 %rs27, %r501;

	setp.lt.s32	%p39, %r11, 1;
mov.u16 %rs133, %rs27;
@%p39 bra BB8_68;

add.s32 %r267, %r74, %r78;
mad.lo.s32 %r499, %r73, %r267, %r72;
mad.lo.s32 %r268, %r147, %r79, %r78;
mad.lo.s32 %r498, %r73, %r268, %r72;
mov.u32 %r500, 0;
mov.u32 %r497, %r76;
mov.u16 %rs133, %rs27;

BB8_65:
setp.ge.u32	%p40, %r497, %r144;
mov.u16 %rs131, %rs27;
mov.u16 %rs132, %rs27;
@%p40 bra BB8_67;

ld.shared.u16 %rs132, [%r498];
ld.shared.u16 %rs131, [%r499];

BB8_67:
mov.f64 %fd1, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs28, %fd1;}


	
	{mul.f16 %rs29,%rs132,%rs132;
}

	
	{sub.f16 %rs32,%rs29,%rs131;
}

	
	{mul.f16 %rs35,%rs28,%rs32;
}

	
	{add.f16 %rs133,%rs133,%rs35;
}

	add.s32 %r499, %r499, %r75;
add.s32 %r498, %r498, %r75;
add.s32 %r497, %r497, %r4;
add.s32 %r500, %r500, 1;
setp.lt.s32	%p41, %r500, %r11;
@%p41 bra BB8_65;

BB8_68:
bar.warp.sync -1;

	{ mov.b32 %r269, {%rs133,%rs133};}


	mov.u32 %r297, 8;
mov.u32 %r273, 1;
mov.u32 %r307, -1;

	{shfl.sync.bfly.b32 %r271,%r269,%r273,%r274,%r307;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r271;
mov.b16 %rs43, low;}

	
	{add.f16 %rs44,%rs133,%rs43;
}

	
	{ mov.b32 %r277, {%rs44,%rs44};}


	mov.u32 %r281, 2;

	{shfl.sync.bfly.b32 %r279,%r277,%r281,%r274,%r307;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r279;
mov.b16 %rs49, low;}

	
	{add.f16 %rs50,%rs44,%rs49;
}

	
	{ mov.b32 %r285, {%rs50,%rs50};}


	mov.u32 %r289, 4;

	{shfl.sync.bfly.b32 %r287,%r285,%r289,%r274,%r307;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r287;
mov.b16 %rs55, low;}

	
	{add.f16 %rs56,%rs50,%rs55;
}

	
	{ mov.b32 %r293, {%rs56,%rs56};}


	
	{shfl.sync.bfly.b32 %r295,%r293,%r297,%r274,%r307;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r295;
mov.b16 %rs61, low;}

	
	{add.f16 %rs62,%rs56,%rs61;
}

	
	{ mov.b32 %r301, {%rs62,%rs62};}


	mov.u32 %r305, 16;

	{shfl.sync.bfly.b32 %r303,%r301,%r305,%r274,%r307;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r303;
mov.b16 %rs67, low;}

	
	{add.f16 %rs68,%rs62,%rs67;
}

	setp.ne.s32	%p42, %r6, 0;
@%p42 bra BB8_70;

add.s32 %r319, %r80, %r78;
cvt.u64.u32	%rd62, %r319;
add.s64 %rd63, %rd62, %rd5;
cvt.u32.u64	%r320, %rd63;
shl.b32 %r321, %r320, 1;
add.s32 %r322, %r3, %r321;
st.shared.u16 [%r322], %rs68;

BB8_70:
bar.warp.sync -1;
setp.lt.s32	%p43, %r496, 1;
@%p43 bra BB8_80;

mul.lo.s32 %r91, %r147, %r79;

BB8_72:
cvt.u64.u32	%rd64, %r501;
add.s64 %rd65, %rd64, %rd2;
cvt.u32.u64	%r324, %rd65;
shl.b32 %r325, %r324, 2;
add.s32 %r327, %r155, %r325;
ld.shared.u32 %r93, [%r327];
mov.u16 %rs137, %rs27;
@%p39 bra BB8_77;

shl.b32 %r329, %r93, 2;
add.s32 %r331, %r155, %r329;
ld.shared.u32 %r332, [%r331];
mad.lo.s32 %r333, %r147, %r332, %r78;
mad.lo.s32 %r504, %r73, %r333, %r72;
add.s32 %r334, %r91, %r93;
mad.lo.s32 %r503, %r73, %r334, %r72;
mov.u32 %r505, 0;
mov.u32 %r502, %r76;
mov.u16 %rs137, %rs27;

BB8_74:
setp.ge.u32	%p45, %r502, %r144;
mov.u16 %rs135, %rs27;
mov.u16 %rs136, %rs27;
@%p45 bra BB8_76;

ld.shared.u16 %rs135, [%r503];
ld.shared.u16 %rs136, [%r504];

BB8_76:

	{mul.f16 %rs71,%rs135,%rs136;
}

	
	{add.f16 %rs137,%rs137,%rs71;
}

	add.s32 %r504, %r504, %r75;
add.s32 %r503, %r503, %r75;
add.s32 %r502, %r502, %r4;
add.s32 %r505, %r505, 1;
setp.lt.s32	%p46, %r505, %r11;
@%p46 bra BB8_74;

BB8_77:
bar.warp.sync -1;
mov.u32 %r467, 8;
mov.u32 %r466, 4;
mov.u32 %r465, 2;
mov.u32 %r464, -1;
mov.u32 %r463, 1;

	{ mov.b32 %r335, {%rs137,%rs137};}


	
	{shfl.sync.bfly.b32 %r337,%r335,%r463,%r274,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r337;
mov.b16 %rs79, low;}

	
	{add.f16 %rs80,%rs137,%rs79;
}

	
	{ mov.b32 %r343, {%rs80,%rs80};}


	
	{shfl.sync.bfly.b32 %r345,%r343,%r465,%r274,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r345;
mov.b16 %rs85, low;}

	
	{add.f16 %rs86,%rs80,%rs85;
}

	
	{ mov.b32 %r351, {%rs86,%rs86};}


	
	{shfl.sync.bfly.b32 %r353,%r351,%r466,%r274,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r353;
mov.b16 %rs91, low;}

	
	{add.f16 %rs92,%rs86,%rs91;
}

	
	{ mov.b32 %r359, {%rs92,%rs92};}


	
	{shfl.sync.bfly.b32 %r361,%r359,%r467,%r274,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r361;
mov.b16 %rs97, low;}

	
	{add.f16 %rs98,%rs92,%rs97;
}

	
	{ mov.b32 %r367, {%rs98,%rs98};}


	
	{shfl.sync.bfly.b32 %r369,%r367,%r305,%r274,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r369;
mov.b16 %rs103, low;}

	
	{add.f16 %rs104,%rs98,%rs103;
}

	@%p42 bra BB8_79;

add.s32 %r385, %r93, %r80;
cvt.u64.u32	%rd66, %r385;
add.s64 %rd67, %rd66, %rd5;
cvt.u32.u64	%r386, %rd67;
shl.b32 %r387, %r386, 1;
add.s32 %r388, %r3, %r387;
st.shared.u16 [%r388], %rs104;

BB8_79:
bar.warp.sync -1;
add.s32 %r501, %r501, 1;
setp.lt.s32	%p48, %r501, %r496;
@%p48 bra BB8_72;

BB8_80:
add.s32 %r496, %r496, %r8;
setp.lt.s32	%p49, %r496, %r71;
@%p49 bra BB8_63;

BB8_81:
cvta.to.global.u64 %rd12, %rd19;
bar.sync 0;
setp.ge.s32	%p50, %r513, %r1;
@%p50 bra BB8_84;

mov.u32 %r506, %r513;

BB8_83:
shl.b32 %r389, %r506, 2;
add.s32 %r391, %r155, %r389;
ld.shared.u32 %r392, [%r391];
mul.wide.s32 %rd68, %r506, 4;
add.s64 %rd69, %rd12, %rd68;
st.global.u32 [%rd69], %r392;
add.s32 %r393, %r506, %r147;
shl.b32 %r394, %r393, 2;
add.s32 %r395, %r155, %r394;
ld.shared.u32 %r396, [%r395+4];
mul.wide.s32 %rd70, %r393, 4;
add.s64 %rd71, %rd12, %rd70;
st.global.u32 [%rd71+4], %r396;
add.s32 %r506, %r506, %r9;
setp.lt.s32	%p51, %r506, %r1;
@%p51 bra BB8_83;

BB8_84:
ld.param.u32 %r460, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_2];
add.s32 %r397, %r460, 1;
mul.lo.s32 %r108, %r397, %r147;
setp.ge.s32	%p52, %r507, %r108;
@%p52 bra BB8_92;

ld.param.u32 %r462, [_ZN8nvinfer119sparse_fipnn_shared23ComputeCommonPartOutputI6__halfEEviiiiiiPiPKT_PKiPS4_S3_S9__param_1];
mov.u32 %r461, %ctaid.y;
shl.b32 %r398, %r147, 3;
add.s32 %r400, %r398, %r155;
add.s32 %r401, %r400, 8;
mad.lo.s32 %r109, %r6, 2, %r401;
shl.b32 %r110, %r462, 1;
shl.b32 %r111, %r4, 1;
mad.lo.s32 %r112, %r461, %r462, %r6;
cvta.to.global.u64 %rd13, %rd18;

BB8_86:
setp.lt.s32	%p53, %r11, 1;
@%p53 bra BB8_91;

mad.lo.s32 %r510, %r110, %r507, %r109;
mad.lo.s32 %r509, %r144, %r507, %r112;
mov.u32 %r511, 0;
mov.u32 %r508, %r112;

BB8_88:
setp.ge.s32	%p54, %r508, %r144;
@%p54 bra BB8_90;

mul.wide.s32 %rd72, %r509, 2;
add.s64 %rd73, %rd13, %rd72;
ld.shared.u16 %rs107, [%r510];
st.global.u16 [%rd73], %rs107;

BB8_90:
add.s32 %r510, %r510, %r111;
add.s32 %r509, %r509, %r4;
add.s32 %r508, %r508, %r4;
add.s32 %r511, %r511, 1;
setp.lt.s32	%p55, %r511, %r11;
@%p55 bra BB8_88;

BB8_91:
add.s32 %r507, %r507, %r8;
setp.lt.s32	%p56, %r507, %r108;
@%p56 bra BB8_86;

BB8_92:
add.s32 %r403, %r149, 3;
shr.s32 %r404, %r403, 31;
shr.u32 %r405, %r404, 30;
add.s32 %r406, %r403, %r405;
shr.s32 %r125, %r406, 2;
setp.ge.s32	%p57, %r513, %r125;
@%p57 bra BB8_107;

shl.b32 %r408, %r9, 2;
neg.s32 %r126, %r408;
shl.b32 %r409, %r513, 2;
sub.s32 %r127, %r149, %r409;
mov.u32 %r512, 0;

BB8_94:
mad.lo.s32 %r130, %r126, %r512, %r127;
shl.b32 %r131, %r513, 2;
add.s32 %r132, %r131, 3;
setp.lt.s32	%p58, %r132, %r149;
@%p58 bra BB8_105;
bra.uni BB8_95;

BB8_105:
mul.wide.s32 %rd103, %r131, 2;
rem.s32 %r441, %r131, %r2;
cvt.u64.u32	%rd104, %r441;
add.s64 %rd105, %rd104, %rd5;
cvt.u32.u64	%r442, %rd105;
shl.b32 %r443, %r442, 1;
add.s32 %r444, %r3, %r443;
ld.shared.u16 %rs123, [%r444];
add.s64 %rd99, %rd20, %rd103;

	{ atom.add.noftz.f16 %rs122,[%rd99],%rs123; }


	add.s64 %rd100, %rd99, 2;
add.s32 %r445, %r131, 1;
rem.s32 %r446, %r445, %r2;
cvt.u64.u32	%rd106, %r446;
add.s64 %rd107, %rd106, %rd5;
cvt.u32.u64	%r447, %rd107;
shl.b32 %r448, %r447, 1;
add.s32 %r449, %r3, %r448;
ld.shared.u16 %rs125, [%r449];

	{ atom.add.noftz.f16 %rs124,[%rd100],%rs125; }


	add.s64 %rd101, %rd99, 4;
add.s32 %r450, %r131, 2;
rem.s32 %r451, %r450, %r2;
cvt.u64.u32	%rd108, %r451;
add.s64 %rd109, %rd108, %rd5;
cvt.u32.u64	%r452, %rd109;
shl.b32 %r453, %r452, 1;
add.s32 %r454, %r3, %r453;
ld.shared.u16 %rs127, [%r454];

	{ atom.add.noftz.f16 %rs126,[%rd101],%rs127; }


	add.s64 %rd102, %rd99, 6;
rem.s32 %r455, %r132, %r2;
cvt.u64.u32	%rd110, %r455;
add.s64 %rd111, %rd110, %rd5;
cvt.u32.u64	%r456, %rd111;
shl.b32 %r457, %r456, 1;
add.s32 %r458, %r3, %r457;
ld.shared.u16 %rs129, [%r458];

	{ atom.add.noftz.f16 %rs128,[%rd102],%rs129; }


	bra.uni BB8_106;

BB8_95:
setp.ge.s32	%p59, %r131, %r149;
@%p59 bra BB8_106;

and.b32 %r133, %r130, 3;
setp.eq.s32	%p60, %r133, 0;
@%p60 bra BB8_102;

setp.eq.s32	%p61, %r133, 1;
@%p61 bra BB8_101;

setp.eq.s32	%p62, %r133, 2;
@%p62 bra BB8_100;

mul.wide.s32 %rd75, %r131, 2;
add.s64 %rd74, %rd20, %rd75;
rem.s32 %r410, %r131, %r2;
cvt.u64.u32	%rd76, %r410;
add.s64 %rd77, %rd76, %rd5;
cvt.u32.u64	%r411, %rd77;
shl.b32 %r412, %r411, 1;
add.s32 %r413, %r3, %r412;
ld.shared.u16 %rs109, [%r413];

	{ atom.add.noftz.f16 %rs108,[%rd74],%rs109; }


	add.s32 %r131, %r131, 1;

BB8_100:
mul.wide.s32 %rd79, %r131, 2;
add.s64 %rd78, %rd20, %rd79;
rem.s32 %r414, %r131, %r2;
cvt.u64.u32	%rd80, %r414;
add.s64 %rd81, %rd80, %rd5;
cvt.u32.u64	%r415, %rd81;
shl.b32 %r416, %r415, 1;
add.s32 %r417, %r3, %r416;
ld.shared.u16 %rs111, [%r417];

	{ atom.add.noftz.f16 %rs110,[%rd78],%rs111; }


	add.s32 %r131, %r131, 1;

BB8_101:
mul.wide.s32 %rd83, %r131, 2;
add.s64 %rd82, %rd20, %rd83;
rem.s32 %r418, %r131, %r2;
cvt.u64.u32	%rd84, %r418;
add.s64 %rd85, %rd84, %rd5;
cvt.u32.u64	%r419, %rd85;
shl.b32 %r420, %r419, 1;
add.s32 %r421, %r3, %r420;
ld.shared.u16 %rs113, [%r421];

	{ atom.add.noftz.f16 %rs112,[%rd82],%rs113; }


	add.s32 %r131, %r131, 1;

BB8_102:
setp.lt.u32	%p63, %r130, 4;
@%p63 bra BB8_106;

mul.wide.s32 %rd86, %r131, 2;
add.s64 %rd112, %rd20, %rd86;

BB8_104:
rem.s32 %r422, %r131, %r2;
cvt.u64.u32	%rd91, %r422;
add.s64 %rd92, %rd91, %rd5;
cvt.u32.u64	%r423, %rd92;
shl.b32 %r424, %r423, 1;
add.s32 %r425, %r3, %r424;
ld.shared.u16 %rs115, [%r425];

	{ atom.add.noftz.f16 %rs114,[%rd112],%rs115; }


	add.s32 %r426, %r131, 1;
rem.s32 %r427, %r426, %r2;
cvt.u64.u32	%rd93, %r427;
add.s64 %rd94, %rd93, %rd5;
cvt.u32.u64	%r428, %rd94;
shl.b32 %r429, %r428, 1;
add.s32 %r430, %r3, %r429;
ld.shared.u16 %rs117, [%r430];
add.s64 %rd88, %rd112, 2;

	{ atom.add.noftz.f16 %rs116,[%rd88],%rs117; }


	add.s32 %r431, %r131, 2;
rem.s32 %r432, %r431, %r2;
cvt.u64.u32	%rd95, %r432;
add.s64 %rd96, %rd95, %rd5;
cvt.u32.u64	%r433, %rd96;
shl.b32 %r434, %r433, 1;
add.s32 %r435, %r3, %r434;
ld.shared.u16 %rs119, [%r435];
add.s64 %rd89, %rd112, 4;

	{ atom.add.noftz.f16 %rs118,[%rd89],%rs119; }


	add.s32 %r436, %r131, 3;
rem.s32 %r437, %r436, %r2;
cvt.u64.u32	%rd97, %r437;
add.s64 %rd98, %rd97, %rd5;
cvt.u32.u64	%r438, %rd98;
shl.b32 %r439, %r438, 1;
add.s32 %r440, %r3, %r439;
ld.shared.u16 %rs121, [%r440];
add.s64 %rd90, %rd112, 6;

	{ atom.add.noftz.f16 %rs120,[%rd90],%rs121; }


	add.s64 %rd112, %rd112, 8;
add.s32 %r131, %r131, 4;
setp.lt.s32	%p64, %r131, %r149;
@%p64 bra BB8_104;

BB8_106:
add.s32 %r513, %r513, %r9;
setp.lt.s32	%p65, %r513, %r125;
add.s32 %r512, %r512, 1;
@%p65 bra BB8_94;

BB8_107:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_11
)
{
.reg .pred %p<61>;
.reg .b16 %rs<170>;
.reg .b32 %r<536>;
.reg .f64 %fd<2>;
.reg .b64 %rd<73>;


ld.param.u32 %r150, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_0];
ld.param.u32 %r151, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_1];
ld.param.u32 %r152, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_2];
ld.param.u32 %r153, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_3];
ld.param.u64 %rd18, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_4];
ld.param.u64 %rd15, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_5];
ld.param.u64 %rd19, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_7];
ld.param.u64 %rd20, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_8];
ld.param.u64 %rd16, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_9];
ld.param.u64 %rd17, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_10];
cvta.to.global.u64 %rd1, %rd19;
cvta.to.global.u64 %rd2, %rd20;
add.s32 %r1, %r153, 1;
mul.lo.s32 %r154, %r1, %r153;
shr.u32 %r155, %r154, 31;
add.s32 %r156, %r154, %r155;
shr.s32 %r157, %r156, 1;
shl.b32 %r2, %r1, 1;
mov.u32 %r158, %ctaid.x;
mul.lo.s32 %r159, %r157, %r158;
cvt.s64.s32	%rd3, %r159;
cvta.to.global.u64 %rd21, %rd18;
mul.wide.s32 %rd22, %r158, 4;
add.s64 %rd23, %rd21, %rd22;
ld.global.u32 %r3, [%rd23];
ld.global.u32 %r4, [%rd23+4];
mov.u32 %r5, %ntid.x;
mov.u32 %r529, %tid.y;
mov.u32 %r7, %tid.x;
mad.lo.s32 %r487, %r5, %r529, %r7;
mov.u32 %r9, %ntid.y;
mul.lo.s32 %r10, %r9, %r5;
div.u32 %r11, %r151, %r5;
mov.u32 %r12, %ctaid.y;
setp.ge.s32	%p1, %r487, %r1;
@%p1 bra BB9_2;

BB9_1:
shl.b32 %r160, %r487, 2;
mov.u32 %r161, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r162, %r161, %r160;
mov.u32 %r163, -1;
st.shared.u32 [%r162], %r163;
mul.wide.s32 %rd24, %r487, 4;
add.s64 %rd25, %rd1, %rd24;
ld.global.u32 %r164, [%rd25];
add.s32 %r165, %r487, %r153;
shl.b32 %r166, %r165, 2;
add.s32 %r167, %r161, %r166;
st.shared.u32 [%r167+4], %r164;
mul.wide.s32 %rd26, %r165, 4;
add.s64 %rd27, %rd1, %rd26;
ld.global.u32 %r168, [%rd27+4];
add.s32 %r169, %r165, %r2;
shl.b32 %r170, %r169, 2;
add.s32 %r171, %r161, %r170;
st.shared.u32 [%r171+4], %r168;
add.s32 %r487, %r487, %r10;
setp.lt.s32	%p2, %r487, %r1;
@%p2 bra BB9_1;

BB9_2:
add.s32 %r172, %r152, 1;
mul.lo.s32 %r15, %r172, %r153;
setp.ge.s32	%p3, %r529, %r15;
@%p3 bra BB9_10;

shl.b32 %r173, %r153, 4;
mov.u32 %r174, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r175, %r173, %r174;
add.s32 %r176, %r175, 16;
mad.lo.s32 %r16, %r7, 2, %r176;
shl.b32 %r17, %r151, 1;
shl.b32 %r18, %r5, 1;
mad.lo.s32 %r19, %r12, %r151, %r7;
cvta.to.global.u64 %rd4, %rd15;
mov.u32 %r488, %r529;

BB9_4:
setp.lt.s32	%p4, %r11, 1;
@%p4 bra BB9_9;

mad.lo.s32 %r491, %r17, %r488, %r16;
mad.lo.s32 %r490, %r150, %r488, %r19;
mov.u32 %r492, 0;
mov.u32 %r489, %r19;

BB9_6:
setp.ge.s32	%p5, %r489, %r150;
@%p5 bra BB9_8;

mul.wide.s32 %rd28, %r490, 2;
add.s64 %rd29, %rd4, %rd28;
ld.global.u16 %rs27, [%rd29];
st.shared.u16 [%r491], %rs27;

BB9_8:
add.s32 %r491, %r491, %r18;
add.s32 %r490, %r490, %r5;
add.s32 %r489, %r489, %r5;
add.s32 %r492, %r492, 1;
setp.lt.s32	%p6, %r492, %r11;
@%p6 bra BB9_6;

BB9_9:
add.s32 %r488, %r488, %r9;
setp.lt.s32	%p7, %r488, %r15;
@%p7 bra BB9_4;

BB9_10:
bar.sync 0;
add.s32 %r493, %r529, %r3;
setp.ge.s32	%p8, %r493, %r4;
@%p8 bra BB9_29;

cvta.to.global.u64 %rd5, %rd16;
cvt.s64.s32	%rd30, %r2;
add.s64 %rd31, %rd30, %rd30;
mov.u32 %r178, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
{
.reg .u64 %temp; 
cvt.u64.u32 %temp, %r178;
cvta.shared.u64 %rd32, %temp;
}
shl.b64 %rd33, %rd31, 2;
add.s64 %rd6, %rd32, %rd33;
cvt.s64.s32	%rd7, %r7;
mul.lo.s32 %r179, %r152, %r151;
mul.lo.s32 %r180, %r179, %r153;
cvt.s64.s32	%rd34, %r180;
add.s64 %rd8, %rd7, %rd34;
mad.lo.s32 %r33, %r12, %r151, %r7;

BB9_12:
shl.b32 %r181, %r493, 1;
mul.wide.s32 %rd35, %r181, 4;
add.s64 %rd36, %rd5, %rd35;
add.s32 %r182, %r181, 1;
mul.wide.s32 %rd37, %r182, 4;
add.s64 %rd38, %rd5, %rd37;
ld.global.u32 %r183, [%rd38];
add.s32 %r35, %r183, -1;
setp.lt.s32	%p9, %r35, 0;
setp.ge.s32	%p10, %r35, %r153;
or.pred %p11, %p9, %p10;
ld.global.u32 %r36, [%rd36];
add.s32 %r184, %r36, -1;
setp.lt.s32	%p12, %r184, 0;
or.pred %p13, %p11, %p12;
setp.ge.s32	%p14, %r184, %r152;
or.pred %p15, %p13, %p14;
@%p15 bra BB9_28;

setp.ne.s32	%p16, %r7, 0;
@%p16 bra BB9_15;

shl.b32 %r185, %r35, 2;
add.s32 %r187, %r178, %r185;
add.s32 %r483, %r36, -1;
st.shared.u32 [%r187], %r483;

BB9_15:
setp.lt.s32	%p17, %r152, 1;
@%p17 bra BB9_23;

mul.lo.s32 %r37, %r152, %r493;
mov.u32 %r494, 0;

BB9_17:
setp.lt.s32	%p18, %r11, 1;
@%p18 bra BB9_22;

mad.lo.s32 %r192, %r494, %r153, %r35;
mul.lo.s32 %r193, %r192, %r151;
cvt.s64.s32	%rd39, %r193;
add.s64 %rd9, %rd39, %rd7;
add.s32 %r194, %r37, %r494;
mad.lo.s32 %r495, %r150, %r194, %r33;
mov.u32 %r497, 0;
mov.u32 %r496, %r33;
mov.u32 %r498, %r497;

BB9_19:
setp.ge.u32	%p19, %r496, %r150;
@%p19 bra BB9_21;

mul.wide.s32 %rd41, %r495, 2;
add.s64 %rd42, %rd2, %rd41;
ld.global.u16 %rs29, [%rd42];
cvt.u64.u32	%rd43, %r497;
add.s64 %rd44, %rd9, %rd43;
shl.b64 %rd45, %rd44, 1;
add.s64 %rd40, %rd6, %rd45;

	{ atom.add.noftz.f16 %rs28,[%rd40],%rs29; }



BB9_21:
add.s32 %r498, %r498, 1;
add.s32 %r497, %r497, %r5;
add.s32 %r496, %r496, %r5;
add.s32 %r495, %r495, %r5;
setp.lt.s32	%p20, %r498, %r11;
@%p20 bra BB9_19;

BB9_22:
add.s32 %r494, %r494, 1;
setp.lt.s32	%p21, %r494, %r152;
@%p21 bra BB9_17;

BB9_23:
setp.lt.s32	%p22, %r11, 1;
@%p22 bra BB9_28;

mul.lo.s32 %r197, %r35, %r151;
cvt.s64.s32	%rd46, %r197;
add.s64 %rd10, %rd8, %rd46;
mad.lo.s32 %r198, %r152, %r493, %r36;
add.s32 %r199, %r198, -1;
mad.lo.s32 %r499, %r150, %r199, %r33;
mov.u32 %r501, 0;
mov.u32 %r500, %r33;
mov.u32 %r502, %r501;

BB9_25:
setp.ge.u32	%p23, %r500, %r150;
@%p23 bra BB9_27;

mul.wide.s32 %rd48, %r499, 2;
add.s64 %rd49, %rd2, %rd48;
ld.global.u16 %rs31, [%rd49];

	{mul.f16 %rs30,%rs31,%rs31;
}

	cvt.u64.u32	%rd50, %r501;
add.s64 %rd51, %rd10, %rd50;
shl.b64 %rd52, %rd51, 1;
add.s64 %rd47, %rd6, %rd52;

	{ atom.add.noftz.f16 %rs33,[%rd47],%rs30; }



BB9_27:
add.s32 %r502, %r502, 1;
add.s32 %r501, %r501, %r5;
add.s32 %r500, %r500, %r5;
add.s32 %r499, %r499, %r5;
setp.lt.s32	%p24, %r502, %r11;
@%p24 bra BB9_25;

BB9_28:
add.s32 %r493, %r493, %r9;
setp.lt.s32	%p25, %r493, %r4;
@%p25 bra BB9_12;

BB9_29:
bar.sync 0;
or.b32 %r200, %r7, %r529;
add.s32 %r201, %r2, %r153;
shl.b32 %r202, %r201, 2;
mov.u32 %r203, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r59, %r203, %r202;
setp.ne.s32	%p26, %r200, 0;
@%p26 bra BB9_63;

mov.u32 %r513, 0;
setp.lt.s32	%p27, %r153, 1;
@%p27 bra BB9_62;

and.b32 %r211, %r153, 3;
mov.u32 %r504, 0;
setp.eq.s32	%p28, %r211, 0;
@%p28 bra BB9_32;

setp.eq.s32	%p29, %r211, 1;
@%p29 bra BB9_34;
bra.uni BB9_35;

BB9_34:
mov.u32 %r507, %r504;
bra.uni BB9_43;

BB9_32:
mov.u32 %r513, %r504;
bra.uni BB9_47;

BB9_35:
setp.eq.s32	%p30, %r211, 2;
@%p30 bra BB9_36;
bra.uni BB9_37;

BB9_36:
mov.u32 %r503, %r504;
bra.uni BB9_39;

BB9_37:
ld.shared.u32 %r214, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r503, 1;
setp.lt.s32	%p31, %r214, 0;
@%p31 bra BB9_39;

shl.b32 %r217, %r2, 2;
add.s32 %r219, %r203, %r217;
mov.u32 %r220, 0;
st.shared.u32 [%r219], %r220;
mov.u32 %r503, 1;
mov.u32 %r504, %r503;

BB9_39:
shl.b32 %r221, %r503, 2;
add.s32 %r223, %r203, %r221;
ld.shared.u32 %r224, [%r223];
setp.lt.s32	%p32, %r224, 0;
@%p32 bra BB9_40;

add.s32 %r507, %r504, 1;
add.s32 %r225, %r504, %r2;
shl.b32 %r226, %r225, 2;
add.s32 %r228, %r203, %r226;
st.shared.u32 [%r228], %r503;
bra.uni BB9_42;

BB9_40:
mov.u32 %r507, %r504;

BB9_42:
add.s32 %r504, %r503, 1;

BB9_43:
shl.b32 %r229, %r504, 2;
add.s32 %r231, %r203, %r229;
ld.shared.u32 %r232, [%r231];
setp.lt.s32	%p33, %r232, 0;
@%p33 bra BB9_44;

add.s32 %r513, %r507, 1;
add.s32 %r233, %r507, %r2;
shl.b32 %r234, %r233, 2;
add.s32 %r236, %r203, %r234;
st.shared.u32 [%r236], %r504;
bra.uni BB9_46;

BB9_44:
mov.u32 %r513, %r507;

BB9_46:
add.s32 %r504, %r504, 1;

BB9_47:
setp.lt.u32	%p34, %r153, 4;
@%p34 bra BB9_62;

shl.b32 %r237, %r504, 2;
add.s32 %r511, %r203, %r237;

BB9_49:
ld.shared.u32 %r239, [%r511];
setp.lt.s32	%p35, %r239, 0;
@%p35 bra BB9_50;

add.s32 %r514, %r513, 1;
add.s32 %r240, %r513, %r2;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r203, %r241;
st.shared.u32 [%r243], %r504;
bra.uni BB9_52;

BB9_50:
mov.u32 %r514, %r513;

BB9_52:
ld.shared.u32 %r244, [%r511+4];
setp.lt.s32	%p36, %r244, 0;
@%p36 bra BB9_53;

add.s32 %r515, %r514, 1;
add.s32 %r245, %r514, %r2;
shl.b32 %r246, %r245, 2;
add.s32 %r248, %r203, %r246;
add.s32 %r249, %r504, 1;
st.shared.u32 [%r248], %r249;
bra.uni BB9_55;

BB9_53:
mov.u32 %r515, %r514;

BB9_55:
ld.shared.u32 %r250, [%r511+8];
setp.lt.s32	%p37, %r250, 0;
@%p37 bra BB9_56;

add.s32 %r516, %r515, 1;
add.s32 %r251, %r515, %r2;
shl.b32 %r252, %r251, 2;
add.s32 %r254, %r203, %r252;
add.s32 %r255, %r504, 2;
st.shared.u32 [%r254], %r255;
bra.uni BB9_58;

BB9_56:
mov.u32 %r516, %r515;

BB9_58:
ld.shared.u32 %r256, [%r511+12];
setp.lt.s32	%p38, %r256, 0;
@%p38 bra BB9_59;

add.s32 %r513, %r516, 1;
add.s32 %r257, %r516, %r2;
shl.b32 %r258, %r257, 2;
add.s32 %r260, %r203, %r258;
add.s32 %r261, %r504, 3;
st.shared.u32 [%r260], %r261;
bra.uni BB9_61;

BB9_59:
mov.u32 %r513, %r516;

BB9_61:
add.s32 %r504, %r504, 4;
setp.lt.s32	%p39, %r504, %r153;
add.s32 %r511, %r511, 16;
@%p39 bra BB9_49;

BB9_62:
st.shared.u32 [%r59], %r513;

BB9_63:
bar.sync 0;
cvt.s64.s32	%rd53, %r153;
add.s64 %rd11, %rd53, 1;
cvt.u64.u32	%rd54, %r153;
add.s64 %rd55, %rd11, %rd54;
cvt.u32.u64	%r262, %rd55;
add.s32 %r263, %r262, %r2;
shl.b32 %r264, %r263, 2;
add.s32 %r266, %r203, %r264;
ld.shared.u32 %r88, [%r266];
ld.shared.u32 %r89, [%r59];
setp.ge.s32	%p40, %r529, %r89;
@%p40 bra BB9_83;

shl.b32 %r267, %r153, 4;
add.s32 %r269, %r267, %r203;
add.s32 %r270, %r269, 16;
mad.lo.s32 %r90, %r7, 2, %r270;
shl.b32 %r91, %r151, 1;
mul.lo.s32 %r92, %r153, %r152;
shl.b32 %r93, %r5, 1;
mad.lo.s32 %r94, %r12, %r151, %r7;

	{mov.u32 %r286, WARP_SZ;
}

	shl.b32 %r325, %r286, 8;
add.s32 %r326, %r325, -8192;
or.b32 %r290, %r326, 31;
mov.u32 %r519, %r529;

BB9_65:
add.s32 %r272, %r519, %r2;
shl.b32 %r273, %r272, 2;
add.s32 %r275, %r203, %r273;
ld.shared.u32 %r96, [%r275];
cvt.s64.s32	%rd12, %r96;
shl.b32 %r276, %r96, 2;
add.s32 %r277, %r203, %r276;
ld.shared.u32 %r97, [%r277];
add.s32 %r278, %r96, 1;
mul.lo.s32 %r279, %r278, %r96;
shr.u32 %r280, %r279, 31;
add.s32 %r281, %r279, %r280;
shr.s32 %r98, %r281, 1;
mov.u32 %r524, 0;

	cvt.rn.f16.s32 %rs35, %r524;

	setp.lt.s32	%p41, %r11, 1;
mov.u16 %rs161, %rs35;
@%p41 bra BB9_70;

add.s32 %r283, %r92, %r96;
mad.lo.s32 %r522, %r91, %r283, %r90;
mad.lo.s32 %r284, %r153, %r97, %r96;
mad.lo.s32 %r521, %r91, %r284, %r90;
mov.u32 %r523, 0;
mov.u32 %r520, %r94;
mov.u16 %rs161, %rs35;

BB9_67:
setp.ge.u32	%p42, %r520, %r150;
mov.u16 %rs159, %rs35;
mov.u16 %rs160, %rs35;
@%p42 bra BB9_69;

ld.shared.u16 %rs160, [%r521];
ld.shared.u16 %rs159, [%r522];

BB9_69:
mov.f64 %fd1, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs36, %fd1;}


	
	{mul.f16 %rs37,%rs160,%rs160;
}

	
	{sub.f16 %rs40,%rs37,%rs159;
}

	
	{mul.f16 %rs43,%rs36,%rs40;
}

	
	{add.f16 %rs161,%rs161,%rs43;
}

	add.s32 %r522, %r522, %r93;
add.s32 %r521, %r521, %r93;
add.s32 %r520, %r520, %r5;
add.s32 %r523, %r523, 1;
setp.lt.s32	%p43, %r523, %r11;
@%p43 bra BB9_67;

BB9_70:
bar.warp.sync -1;

	{ mov.b32 %r285, {%rs161,%rs161};}


	mov.u32 %r313, 8;
mov.u32 %r289, 1;
mov.u32 %r323, -1;

	{shfl.sync.bfly.b32 %r287,%r285,%r289,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r287;
mov.b16 %rs51, low;}

	
	{add.f16 %rs52,%rs161,%rs51;
}

	
	{ mov.b32 %r293, {%rs52,%rs52};}


	mov.u32 %r297, 2;

	{shfl.sync.bfly.b32 %r295,%r293,%r297,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r295;
mov.b16 %rs57, low;}

	
	{add.f16 %rs58,%rs52,%rs57;
}

	
	{ mov.b32 %r301, {%rs58,%rs58};}


	mov.u32 %r305, 4;

	{shfl.sync.bfly.b32 %r303,%r301,%r305,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r303;
mov.b16 %rs63, low;}

	
	{add.f16 %rs64,%rs58,%rs63;
}

	
	{ mov.b32 %r309, {%rs64,%rs64};}


	
	{shfl.sync.bfly.b32 %r311,%r309,%r313,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r311;
mov.b16 %rs69, low;}

	
	{add.f16 %rs70,%rs64,%rs69;
}

	
	{ mov.b32 %r317, {%rs70,%rs70};}


	mov.u32 %r321, 16;

	{shfl.sync.bfly.b32 %r319,%r317,%r321,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r319;
mov.b16 %rs75, low;}

	
	{add.f16 %rs76,%rs70,%rs75;
}

	setp.ne.s32	%p44, %r7, 0;
@%p44 bra BB9_72;

cvt.s64.s32	%rd57, %r98;
add.s64 %rd58, %rd12, %rd3;
add.s64 %rd59, %rd58, %rd57;
shl.b64 %rd60, %rd59, 1;
add.s64 %rd56, %rd17, %rd60;

	{ atom.add.noftz.f16 %rs79,[%rd56],%rs76; }



BB9_72:
bar.warp.sync -1;
setp.lt.s32	%p45, %r519, 1;
@%p45 bra BB9_82;

cvt.s64.s32	%rd61, %r98;
add.s64 %rd13, %rd61, %rd3;
mul.lo.s32 %r109, %r153, %r97;

BB9_74:
add.s32 %r336, %r524, %r2;
shl.b32 %r337, %r336, 2;
add.s32 %r339, %r203, %r337;
ld.shared.u32 %r111, [%r339];
cvt.s64.s32	%rd14, %r111;
mov.u16 %rs165, %rs35;
@%p41 bra BB9_79;

shl.b32 %r341, %r111, 2;
add.s32 %r343, %r203, %r341;
ld.shared.u32 %r344, [%r343];
cvt.u32.u64	%r345, %rd14;
cvt.u32.u64	%r346, %rd12;
mad.lo.s32 %r347, %r153, %r344, %r346;
mad.lo.s32 %r527, %r91, %r347, %r90;
add.s32 %r348, %r109, %r345;
mad.lo.s32 %r526, %r91, %r348, %r90;
mov.u32 %r528, 0;
mov.u32 %r525, %r94;
mov.u16 %rs165, %rs35;

BB9_76:
setp.ge.u32	%p47, %r525, %r150;
mov.u16 %rs163, %rs35;
mov.u16 %rs164, %rs35;
@%p47 bra BB9_78;

ld.shared.u16 %rs164, [%r526];
ld.shared.u16 %rs163, [%r527];

BB9_78:

	{mul.f16 %rs81,%rs164,%rs163;
}

	
	{add.f16 %rs165,%rs165,%rs81;
}

	add.s32 %r527, %r527, %r93;
add.s32 %r526, %r526, %r93;
add.s32 %r525, %r525, %r5;
add.s32 %r528, %r528, 1;
setp.lt.s32	%p48, %r528, %r11;
@%p48 bra BB9_76;

BB9_79:
bar.warp.sync -1;
mov.u32 %r486, 1;

	{ mov.b32 %r349, {%rs165,%rs165};}


	
	{shfl.sync.bfly.b32 %r351,%r349,%r486,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r351;
mov.b16 %rs89, low;}

	
	{add.f16 %rs90,%rs165,%rs89;
}

	
	{ mov.b32 %r357, {%rs90,%rs90};}


	
	{shfl.sync.bfly.b32 %r359,%r357,%r297,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r359;
mov.b16 %rs95, low;}

	
	{add.f16 %rs96,%rs90,%rs95;
}

	
	{ mov.b32 %r365, {%rs96,%rs96};}


	
	{shfl.sync.bfly.b32 %r367,%r365,%r305,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r367;
mov.b16 %rs101, low;}

	
	{add.f16 %rs102,%rs96,%rs101;
}

	
	{ mov.b32 %r373, {%rs102,%rs102};}


	
	{shfl.sync.bfly.b32 %r375,%r373,%r313,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r375;
mov.b16 %rs107, low;}

	
	{add.f16 %rs108,%rs102,%rs107;
}

	
	{ mov.b32 %r381, {%rs108,%rs108};}


	
	{shfl.sync.bfly.b32 %r383,%r381,%r321,%r290,%r323;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r383;
mov.b16 %rs113, low;}

	
	{add.f16 %rs114,%rs108,%rs113;
}

	@%p44 bra BB9_81;

add.s64 %rd63, %rd13, %rd14;
shl.b64 %rd64, %rd63, 1;
add.s64 %rd62, %rd17, %rd64;

	{ atom.add.noftz.f16 %rs117,[%rd62],%rs114; }



BB9_81:
bar.warp.sync -1;
add.s32 %r524, %r524, 1;
setp.lt.s32	%p50, %r524, %r519;
@%p50 bra BB9_74;

BB9_82:
add.s32 %r519, %r519, %r9;
setp.lt.s32	%p51, %r519, %r89;
@%p51 bra BB9_65;

BB9_83:
setp.ge.s32	%p52, %r529, %r88;
@%p52 bra BB9_97;

mov.u32 %r485, %ctaid.y;
ld.param.u32 %r484, [_ZN8nvinfer119sparse_fipnn_shared21SparseFIPNNGpuShareV2I6__halfEEviiiiPiPT_S5_S3_PKS4_PKiS5_S5__param_1];
shl.b32 %r399, %r153, 4;
add.s32 %r401, %r399, %r203;
add.s32 %r402, %r401, 16;
mad.lo.s32 %r124, %r7, 2, %r402;
shl.b32 %r125, %r484, 1;
shl.b32 %r126, %r5, 1;
mad.lo.s32 %r129, %r485, %r484, %r7;

	{mov.u32 %r427, WARP_SZ;
}

	shl.b32 %r466, %r427, 8;
add.s32 %r467, %r466, -8192;
or.b32 %r431, %r467, 31;

BB9_85:
cvt.u64.u32	%rd65, %r529;
add.s64 %rd66, %rd65, %rd11;
cvt.u32.u64	%r403, %rd66;
add.s32 %r404, %r403, %r2;
shl.b32 %r405, %r404, 2;
add.s32 %r407, %r203, %r405;
ld.shared.u32 %r531, [%r407];
setp.lt.s32	%p53, %r89, 1;
@%p53 bra BB9_96;

add.s32 %r409, %r531, %r1;
shl.b32 %r410, %r409, 2;
add.s32 %r412, %r203, %r410;
ld.shared.u32 %r413, [%r412];
mul.lo.s32 %r132, %r153, %r413;
mov.u32 %r408, 0;
mov.u32 %r530, %r408;

BB9_87:
add.s32 %r415, %r530, %r2;
shl.b32 %r416, %r415, 2;
add.s32 %r418, %r203, %r416;
ld.shared.u32 %r135, [%r418];
min.s32 %r136, %r135, %r531;

	cvt.rn.f16.s32 %rs119, %r408;

	setp.lt.s32	%p54, %r11, 1;
@%p54 bra BB9_88;

shl.b32 %r420, %r135, 2;
add.s32 %r422, %r203, %r420;
ld.shared.u32 %r423, [%r422];
mad.lo.s32 %r424, %r153, %r423, %r531;
mad.lo.s32 %r534, %r125, %r424, %r124;
add.s32 %r425, %r132, %r135;
mad.lo.s32 %r533, %r125, %r425, %r124;
mov.u32 %r535, 0;
mov.u32 %r532, %r129;
mov.u16 %rs169, %rs119;

BB9_90:
setp.ge.u32	%p55, %r532, %r150;
mov.u16 %rs167, %rs119;
mov.u16 %rs168, %rs119;
@%p55 bra BB9_92;

ld.shared.u16 %rs167, [%r533];
ld.shared.u16 %rs168, [%r534];

BB9_92:

	{mul.f16 %rs120,%rs167,%rs168;
}

	
	{add.f16 %rs169,%rs169,%rs120;
}

	add.s32 %r534, %r534, %r126;
add.s32 %r533, %r533, %r126;
add.s32 %r532, %r532, %r5;
add.s32 %r535, %r535, 1;
setp.lt.s32	%p56, %r535, %r11;
@%p56 bra BB9_90;
bra.uni BB9_93;

BB9_88:
mov.u16 %rs169, %rs119;

BB9_93:
bar.warp.sync -1;

	{ mov.b32 %r426, {%rs169,%rs169};}


	mov.u32 %r454, 8;
mov.u32 %r430, 1;
mov.u32 %r464, -1;

	{shfl.sync.bfly.b32 %r428,%r426,%r430,%r431,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r428;
mov.b16 %rs128, low;}

	
	{add.f16 %rs129,%rs169,%rs128;
}

	
	{ mov.b32 %r434, {%rs129,%rs129};}


	mov.u32 %r438, 2;

	{shfl.sync.bfly.b32 %r436,%r434,%r438,%r431,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r436;
mov.b16 %rs134, low;}

	
	{add.f16 %rs135,%rs129,%rs134;
}

	
	{ mov.b32 %r442, {%rs135,%rs135};}


	mov.u32 %r446, 4;

	{shfl.sync.bfly.b32 %r444,%r442,%r446,%r431,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r444;
mov.b16 %rs140, low;}

	
	{add.f16 %rs141,%rs135,%rs140;
}

	
	{ mov.b32 %r450, {%rs141,%rs141};}


	
	{shfl.sync.bfly.b32 %r452,%r450,%r454,%r431,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r452;
mov.b16 %rs146, low;}

	
	{add.f16 %rs147,%rs141,%rs146;
}

	
	{ mov.b32 %r458, {%rs147,%rs147};}


	mov.u32 %r462, 16;

	{shfl.sync.bfly.b32 %r460,%r458,%r462,%r431,%r464;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r460;
mov.b16 %rs152, low;}

	
	{add.f16 %rs153,%rs147,%rs152;
}

	add.s32 %r476, %r531, 1;
mul.lo.s32 %r477, %r476, %r531;
setp.lt.s32	%p57, %r135, %r531;
add.s32 %r478, %r135, 1;
mul.lo.s32 %r479, %r478, %r135;
selp.b32	%r147, %r477, %r479, %p57;
setp.ne.s32	%p58, %r7, 0;
@%p58 bra BB9_95;

shr.u32 %r480, %r147, 31;
add.s32 %r481, %r147, %r480;
shr.s32 %r482, %r481, 1;
cvt.s64.s32	%rd68, %r482;
cvt.s64.s32	%rd69, %r136;
add.s64 %rd70, %rd69, %rd3;
add.s64 %rd71, %rd70, %rd68;
shl.b64 %rd72, %rd71, 1;
add.s64 %rd67, %rd17, %rd72;

	{ atom.add.noftz.f16 %rs156,[%rd67],%rs153; }



BB9_95:
bar.warp.sync -1;
add.s32 %r530, %r530, 1;
setp.lt.s32	%p59, %r530, %r89;
mov.u32 %r531, %r136;
@%p59 bra BB9_87;

BB9_96:
add.s32 %r529, %r529, %r9;
setp.lt.s32	%p60, %r529, %r88;
@%p60 bra BB9_85;

BB9_97:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_3,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_10,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_11
)
{
.reg .pred %p<38>;
.reg .b16 %rs<24>;
.reg .f32 %f<3>;
.reg .b32 %r<169>;
.reg .b64 %rd<91>;


ld.param.u32 %r52, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_0];
ld.param.u32 %r53, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_1];
ld.param.u32 %r54, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_2];
ld.param.u32 %r55, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_3];
ld.param.u32 %r56, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_4];
ld.param.u64 %rd31, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_5];
ld.param.u64 %rd25, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_6];
ld.param.u64 %rd26, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_7];
ld.param.u64 %rd27, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_8];
ld.param.u64 %rd28, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_9];
ld.param.u64 %rd29, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_10];
ld.param.u64 %rd30, [_ZN8nvinfer119sparse_fipnn_shared17ProcessCommonPartI6__halfLi32EEEviiiiiPiPKT_PKiPS4_S9_S3_S9__param_11];
cvta.to.global.u64 %rd32, %rd31;
mov.u32 %r57, %ntid.x;
mov.u32 %r58, %tid.y;
mul.lo.s32 %r1, %r57, %r58;
mov.u32 %r59, %tid.x;
add.s32 %r60, %r1, %r59;
ld.global.u32 %r2, [%rd32];
setp.ge.s32	%p1, %r60, %r55;
@%p1 bra BB10_24;

add.s32 %r61, %r55, -1;
sub.s32 %r65, %r61, %r1;
mov.u32 %r66, %tid.x;
sub.s32 %r67, %r65, %r66;
shr.u32 %r68, %r67, 10;
add.s32 %r69, %r68, 1;
and.b32 %r3, %r69, 3;
setp.eq.s32	%p2, %r3, 0;
add.s32 %r158, %r1, %r66;
@%p2 bra BB10_13;

setp.eq.s32	%p3, %r3, 1;
add.s32 %r156, %r1, %r66;
@%p3 bra BB10_10;

setp.eq.s32	%p4, %r3, 2;
add.s32 %r155, %r1, %r66;
@%p4 bra BB10_7;

mov.u32 %r72, 0;

	cvt.rn.f16.s32 %rs1, %r72;

	add.s32 %r7, %r1, %r66;
cvta.to.global.u64 %rd33, %rd27;
mul.wide.s32 %rd34, %r7, 2;
add.s64 %rd35, %rd33, %rd34;
st.global.u16 [%rd35], %rs1;
setp.ge.s32	%p5, %r7, %r54;
@%p5 bra BB10_6;

cvta.to.global.u64 %rd36, %rd29;
mul.wide.s32 %rd37, %r7, 4;
add.s64 %rd38, %rd36, %rd37;
mov.u32 %r74, -1;
st.global.u32 [%rd38], %r74;

BB10_6:
add.s32 %r155, %r7, 1024;

BB10_7:
mov.u32 %r77, 0;

	cvt.rn.f16.s32 %rs2, %r77;

	cvta.to.global.u64 %rd39, %rd27;
cvt.s64.s32	%rd1, %r155;
mul.wide.s32 %rd40, %r155, 2;
add.s64 %rd41, %rd39, %rd40;
st.global.u16 [%rd41], %rs2;
setp.ge.s32	%p6, %r155, %r54;
@%p6 bra BB10_9;

cvta.to.global.u64 %rd42, %rd29;
mul.wide.s32 %rd43, %r155, 4;
add.s64 %rd44, %rd42, %rd43;
mov.u32 %r78, -1;
st.global.u32 [%rd44], %r78;

BB10_9:
cvt.u32.u64	%r79, %rd1;
add.s32 %r156, %r79, 1024;

BB10_10:
mov.u32 %r80, 0;

	cvt.rn.f16.s32 %rs3, %r80;

	cvta.to.global.u64 %rd45, %rd27;
cvt.s64.s32	%rd2, %r156;
mul.wide.s32 %rd46, %r156, 2;
add.s64 %rd47, %rd45, %rd46;
st.global.u16 [%rd47], %rs3;
setp.ge.s32	%p7, %r156, %r54;
@%p7 bra BB10_12;

cvta.to.global.u64 %rd48, %rd29;
mul.wide.s32 %rd49, %r156, 4;
add.s64 %rd50, %rd48, %rd49;
mov.u32 %r81, -1;
st.global.u32 [%rd50], %r81;

BB10_12:
cvt.u32.u64	%r82, %rd2;
add.s32 %r158, %r82, 1024;

BB10_13:
setp.lt.u32	%p8, %r69, 4;
@%p8 bra BB10_24;

cvta.to.global.u64 %rd51, %rd29;
mul.wide.s32 %rd52, %r158, 4;
add.s64 %rd85, %rd51, %rd52;
cvta.to.global.u64 %rd53, %rd27;
mul.wide.s32 %rd54, %r158, 2;
add.s64 %rd84, %rd53, %rd54;

BB10_15:
mov.u32 %r89, 0;

	cvt.rn.f16.s32 %rs4, %r89;

	st.global.u16 [%rd84], %rs4;
setp.ge.s32	%p9, %r158, %r54;
@%p9 bra BB10_17;

mov.u32 %r90, -1;
st.global.u32 [%rd85], %r90;

BB10_17:

	cvt.rn.f16.s32 %rs5, %r89;

	st.global.u16 [%rd84+2048], %rs5;
add.s32 %r15, %r158, 1024;
setp.ge.s32	%p10, %r15, %r54;
@%p10 bra BB10_19;

mov.u32 %r92, -1;
st.global.u32 [%rd85+4096], %r92;

BB10_19:

	cvt.rn.f16.s32 %rs6, %r89;

	st.global.u16 [%rd84+4096], %rs6;
add.s32 %r16, %r15, 1024;
setp.ge.s32	%p11, %r16, %r54;
@%p11 bra BB10_21;

mov.u32 %r94, -1;
st.global.u32 [%rd85+8192], %r94;

BB10_21:

	cvt.rn.f16.s32 %rs7, %r89;

	st.global.u16 [%rd84+6144], %rs7;
add.s32 %r17, %r16, 1024;
setp.ge.s32	%p12, %r17, %r54;
@%p12 bra BB10_23;

mov.u32 %r96, -1;
st.global.u32 [%rd85+12288], %r96;

BB10_23:
add.s64 %rd85, %rd85, 16384;
add.s32 %r158, %r17, 1024;
setp.lt.s32	%p13, %r158, %r55;
add.s64 %rd84, %rd84, 8192;
@%p13 bra BB10_15;

BB10_24:
mad.lo.s32 %r100, %r57, %r58, %r59;
setp.ge.s32	%p14, %r100, %r56;
@%p14 bra BB10_34;

add.s32 %r101, %r56, -1;
sub.s32 %r105, %r101, %r1;
mov.u32 %r106, %tid.x;
sub.s32 %r107, %r105, %r106;
shr.u32 %r108, %r107, 10;
add.s32 %r19, %r108, 1;
and.b32 %r20, %r19, 3;
setp.eq.s32	%p15, %r20, 0;
add.s32 %r162, %r1, %r106;
@%p15 bra BB10_31;

setp.eq.s32	%p16, %r20, 1;
mad.lo.s32 %r160, %r57, %r58, %r106;
@%p16 bra BB10_30;

setp.eq.s32	%p17, %r20, 2;
mad.lo.s32 %r159, %r57, %r58, %r106;
@%p17 bra BB10_29;

mov.u32 %r115, 0;

	cvt.rn.f16.s32 %rs8, %r115;

	mad.lo.s32 %r119, %r57, %r58, %r106;
cvta.to.global.u64 %rd55, %rd30;
mul.wide.s32 %rd56, %r119, 2;
add.s64 %rd57, %rd55, %rd56;
st.global.u16 [%rd57], %rs8;
add.s32 %r159, %r119, 1024;

BB10_29:
mov.u32 %r120, 0;

	cvt.rn.f16.s32 %rs9, %r120;

	cvta.to.global.u64 %rd58, %rd30;
mul.wide.s32 %rd59, %r159, 2;
add.s64 %rd60, %rd58, %rd59;
st.global.u16 [%rd60], %rs9;
add.s32 %r160, %r159, 1024;

BB10_30:
mov.u32 %r121, 0;

	cvt.rn.f16.s32 %rs10, %r121;

	cvta.to.global.u64 %rd61, %rd30;
mul.wide.s32 %rd62, %r160, 2;
add.s64 %rd63, %rd61, %rd62;
st.global.u16 [%rd63], %rs10;
add.s32 %r162, %r160, 1024;

BB10_31:
setp.lt.u32	%p18, %r19, 4;
@%p18 bra BB10_34;

cvta.to.global.u64 %rd64, %rd30;
mul.wide.s32 %rd65, %r162, 2;
add.s64 %rd86, %rd64, %rd65;

BB10_33:
mov.u32 %r125, 0;

	cvt.rn.f16.s32 %rs11, %r125;

	st.global.u16 [%rd86], %rs11;

	cvt.rn.f16.s32 %rs12, %r125;

	st.global.u16 [%rd86+2048], %rs12;

	cvt.rn.f16.s32 %rs13, %r125;

	st.global.u16 [%rd86+4096], %rs13;

	cvt.rn.f16.s32 %rs14, %r125;

	st.global.u16 [%rd86+6144], %rs14;
add.s64 %rd86, %rd86, 8192;
add.s32 %r162, %r162, 4096;
setp.lt.s32	%p19, %r162, %r56;
@%p19 bra BB10_33;

BB10_34:
mov.u32 %r126, %ctaid.x;
shl.b32 %r127, %r126, 5;
add.s32 %r163, %r127, %r58;
setp.ge.s32	%p20, %r163, %r2;
@%p20 bra BB10_53;

add.s32 %r130, %r52, 31;
shr.s32 %r131, %r130, 31;
shr.u32 %r132, %r131, 27;
add.s32 %r133, %r130, %r132;
and.b32 %r32, %r133, -32;
cvta.to.global.u64 %rd66, %rd26;

BB10_36:
shl.b32 %r137, %r163, 1;
mul.wide.s32 %rd67, %r137, 4;
add.s64 %rd68, %rd66, %rd67;
ld.global.u32 %r35, [%rd68];
add.s32 %r36, %r35, -1;
ld.global.u32 %r37, [%rd68+4];
add.s32 %r38, %r37, -1;
setp.lt.s32	%p21, %r38, 0;
setp.ge.s32	%p22, %r38, %r54;
or.pred %p23, %p21, %p22;
setp.lt.s32	%p24, %r36, 0;
or.pred %p25, %p23, %p24;
setp.ge.s32	%p26, %r36, %r53;
or.pred %p27, %p25, %p26;
@%p27 bra BB10_52;

setp.ne.s32	%p28, %r59, 0;
@%p28 bra BB10_39;

cvta.to.global.u64 %rd69, %rd29;
mul.wide.s32 %rd70, %r38, 4;
add.s64 %rd71, %rd69, %rd70;
add.s32 %r154, %r35, -1;
st.global.u32 [%rd71], %r154;

BB10_39:
setp.lt.s32	%p29, %r53, 1;
@%p29 bra BB10_47;

mov.u32 %r164, 0;

BB10_41:
setp.lt.s32	%p30, %r52, 1;
@%p30 bra BB10_46;

mad.lo.s32 %r141, %r53, %r163, %r164;
mov.u32 %r165, %tid.x;
mad.lo.s32 %r142, %r52, %r141, %r165;
cvta.to.global.u64 %rd72, %rd25;
mul.wide.s32 %rd73, %r142, 2;
add.s64 %rd88, %rd72, %rd73;
mad.lo.s32 %r143, %r54, %r164, %r37;
add.s32 %r144, %r143, -1;
mul.lo.s32 %r145, %r52, %r144;
mul.wide.s32 %rd74, %r165, 2;
add.s64 %rd75, %rd27, %rd74;
mul.wide.s32 %rd76, %r145, 2;
add.s64 %rd87, %rd75, %rd76;
mov.u32 %r166, 0;

BB10_43:
setp.ge.s32	%p31, %r165, %r52;
@%p31 bra BB10_45;

ld.global.u16 %rs16, [%rd88];

	{ atom.add.noftz.f16 %rs15,[%rd87],%rs16; }



BB10_45:
add.s32 %r166, %r166, 32;
add.s64 %rd88, %rd88, 64;
add.s32 %r165, %r165, 32;
add.s64 %rd87, %rd87, 64;
setp.lt.s32	%p32, %r166, %r32;
@%p32 bra BB10_43;

BB10_46:
add.s32 %r164, %r164, 1;
setp.lt.s32	%p33, %r164, %r53;
@%p33 bra BB10_41;

BB10_47:
setp.lt.s32	%p34, %r52, 1;
@%p34 bra BB10_52;

mad.lo.s32 %r147, %r53, %r163, %r35;
add.s32 %r148, %r147, -1;
mad.lo.s32 %r149, %r52, %r148, %r59;
cvta.to.global.u64 %rd78, %rd25;
mul.wide.s32 %rd79, %r149, 2;
add.s64 %rd90, %rd78, %rd79;
mul.lo.s32 %r151, %r52, %r38;
mul.wide.s32 %rd80, %r59, 2;
add.s64 %rd81, %rd28, %rd80;
mul.wide.s32 %rd82, %r151, 2;
add.s64 %rd89, %rd81, %rd82;
mov.u32 %r168, 0;
mov.u32 %r167, %r59;

BB10_49:
setp.ge.s32	%p35, %r167, %r52;
@%p35 bra BB10_51;

ld.global.u16 %rs18, [%rd90];

	{mul.f16 %rs17,%rs18,%rs18;
}

	
	{ cvt.f32.f16 %f1, %rs17;}


	
	{ cvt.rn.f16.f32 %rs21, %f1;}


	
	{ atom.add.noftz.f16 %rs22,[%rd89],%rs21; }



BB10_51:
add.s32 %r168, %r168, 32;
add.s64 %rd90, %rd90, 64;
add.s32 %r167, %r167, 32;
add.s64 %rd89, %rd89, 64;
setp.lt.s32	%p36, %r168, %r32;
@%p36 bra BB10_49;

BB10_52:
mov.u32 %r152, %nctaid.x;
shl.b32 %r153, %r152, 5;
add.s32 %r163, %r163, %r153;
setp.lt.s32	%p37, %r163, %r2;
@%p37 bra BB10_36;

BB10_53:
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_2,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_6
)
{
.reg .b16 %rs<4>;
.reg .b32 %r<20>;
.reg .b64 %rd<18>;


ld.param.u32 %r1, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_1];
ld.param.u32 %r2, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_2];
ld.param.u32 %r3, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_3];
ld.param.u64 %rd1, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_4];
ld.param.u64 %rd2, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_5];
ld.param.u64 %rd3, [_ZN8nvinfer119sparse_fipnn_shared19BroadcastCommonPartI6__halfEEviiiiPT_S4_S4__param_6];
cvta.to.global.u64 %rd4, %rd1;
cvta.to.global.u64 %rd5, %rd3;
cvta.to.global.u64 %rd6, %rd2;
mov.u32 %r4, %tid.x;
mov.u32 %r5, %ctaid.x;
rem.u32 %r6, %r5, %r3;
mad.lo.s32 %r7, %r6, %r1, %r4;
div.u32 %r8, %r5, %r3;
mul.wide.s32 %rd7, %r7, 2;
add.s64 %rd8, %rd6, %rd7;
ld.global.u16 %rs1, [%rd8];
add.s32 %r9, %r2, 1;
mul.lo.s32 %r10, %r9, %r1;
mul.lo.s32 %r11, %r10, %r3;
mul.lo.s32 %r12, %r11, %r8;
mul.lo.s32 %r13, %r2, %r1;
mad.lo.s32 %r14, %r13, %r3, %r12;
add.s32 %r15, %r14, %r7;
mul.wide.s32 %rd9, %r15, 2;
add.s64 %rd10, %rd5, %rd9;
st.global.u16 [%rd10], %rs1;
add.s64 %rd11, %rd4, %rd7;
ld.global.u16 %rs2, [%rd11];
mul.lo.s32 %r16, %r3, %r1;
add.s32 %r17, %r7, %r16;
mul.wide.s32 %rd12, %r17, 2;
add.s64 %rd13, %rd4, %rd12;
ld.global.u16 %rs3, [%rd13];
add.s32 %r18, %r12, %r7;
mul.wide.s32 %rd14, %r18, 2;
add.s64 %rd15, %rd5, %rd14;
st.global.u16 [%rd15], %rs2;
add.s32 %r19, %r18, %r16;
mul.wide.s32 %rd16, %r19, 2;
add.s64 %rd17, %rd5, %rd16;
st.global.u16 [%rd17], %rs3;
ret;
}


.visible .entry _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5_(
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_0,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_1,
.param .u32 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_2,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_3,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_4,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_5,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_6,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_7,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_8,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_9,
.param .u64 _ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_10
)
{
.local .align 8 .b8 __local_depot12[24];
.reg .b64 %SP;
.reg .b64 %SPL;
.reg .pred %p<83>;
.reg .b16 %rs<260>;
.reg .f32 %f<3>;
.reg .b32 %r<505>;
.reg .f64 %fd<8>;
.reg .b64 %rd<136>;


mov.u64 %SPL, __local_depot12;
ld.param.u32 %r116, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_0];
ld.param.u32 %r117, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_1];
ld.param.u32 %r118, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_2];
ld.param.u64 %rd48, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_3];
ld.param.u64 %rd49, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_6];
ld.param.u64 %rd50, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_7];
ld.param.u64 %rd46, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_8];
ld.param.u64 %rd51, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_9];
ld.param.u64 %rd47, [_ZN8nvinfer119sparse_fipnn_shared14SparseFIPNNGpuI6__halfLi32EEEviiiPiPT_S5_S3_PKS4_PKiS5_S5__param_10];
cvta.to.global.u64 %rd1, %rd49;
cvta.to.global.u64 %rd2, %rd50;
cvta.to.global.u64 %rd3, %rd47;
cvta.to.global.u64 %rd4, %rd51;
add.u64 %rd5, %SPL, 0;
add.s32 %r119, %r118, 1;
mul.lo.s32 %r120, %r119, %r118;
shr.u32 %r121, %r120, 31;
add.s32 %r122, %r120, %r121;
shr.s32 %r123, %r122, 1;
mov.u32 %r1, %ctaid.x;
mul.lo.s32 %r2, %r123, %r1;
add.s32 %r124, %r117, 1;
mul.lo.s32 %r125, %r124, %r116;
mul.lo.s32 %r3, %r125, %r118;
mul.lo.s32 %r4, %r3, %r1;
mul.lo.s32 %r126, %r117, %r116;
mad.lo.s32 %r5, %r126, %r118, %r4;
cvta.to.global.u64 %rd53, %rd48;
mul.wide.s32 %rd54, %r1, 4;
add.s64 %rd55, %rd53, %rd54;
ld.global.u32 %r6, [%rd55];
ld.global.u32 %r7, [%rd55+4];
mov.u32 %r127, %ntid.x;
mov.u32 %r491, %tid.y;
mov.u32 %r9, %tid.x;
mad.lo.s32 %r10, %r127, %r491, %r9;
setp.lt.s32	%p1, %r118, 1;
@%p1 bra BB12_23;

add.s32 %r132, %r118, -1;
shr.u32 %r133, %r132, 10;
add.s32 %r11, %r133, 1;
and.b32 %r131, %r11, 3;
mov.u32 %r463, 0;
setp.eq.s32	%p2, %r131, 0;
@%p2 bra BB12_12;

setp.eq.s32	%p3, %r131, 1;
@%p3 bra BB12_9;

setp.eq.s32	%p4, %r131, 2;
@%p4 bra BB12_6;

mov.u32 %r463, 1024;
setp.ge.s32	%p5, %r10, %r118;
@%p5 bra BB12_6;

mul.wide.s32 %rd56, %r10, 4;
add.s64 %rd57, %rd1, %rd56;
ld.global.u32 %r136, [%rd57];
shl.b32 %r137, %r10, 2;
mov.u32 %r138, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r139, %r138, %r137;
st.shared.u32 [%r139], %r136;

BB12_6:
add.s32 %r13, %r10, %r463;
setp.ge.s32	%p6, %r13, %r118;
@%p6 bra BB12_8;

mul.wide.s32 %rd58, %r13, 4;
add.s64 %rd59, %rd1, %rd58;
ld.global.u32 %r140, [%rd59];
shl.b32 %r141, %r13, 2;
mov.u32 %r142, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r143, %r142, %r141;
st.shared.u32 [%r143], %r140;

BB12_8:
add.s32 %r463, %r463, 1024;

BB12_9:
add.s32 %r16, %r10, %r463;
setp.ge.s32	%p7, %r16, %r118;
@%p7 bra BB12_11;

mul.wide.s32 %rd60, %r16, 4;
add.s64 %rd61, %rd1, %rd60;
ld.global.u32 %r144, [%rd61];
shl.b32 %r145, %r16, 2;
mov.u32 %r146, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r147, %r146, %r145;
st.shared.u32 [%r147], %r144;

BB12_11:
add.s32 %r463, %r463, 1024;

BB12_12:
setp.lt.u32	%p8, %r11, 4;
@%p8 bra BB12_23;

add.s32 %r466, %r10, %r463;
shl.b32 %r148, %r466, 2;
mov.u32 %r149, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r467, %r149, %r148;
mul.wide.s32 %rd62, %r466, 4;
add.s64 %rd126, %rd1, %rd62;

BB12_14:
setp.ge.s32	%p9, %r466, %r118;
@%p9 bra BB12_16;

ld.global.u32 %r150, [%rd126];
st.shared.u32 [%r467], %r150;

BB12_16:
add.s32 %r151, %r466, 1024;
setp.ge.s32	%p10, %r151, %r118;
@%p10 bra BB12_18;

ld.global.u32 %r152, [%rd126+4096];
st.shared.u32 [%r467+4096], %r152;

BB12_18:
add.s32 %r153, %r466, 2048;
setp.ge.s32	%p11, %r153, %r118;
@%p11 bra BB12_20;

ld.global.u32 %r154, [%rd126+8192];
st.shared.u32 [%r467+8192], %r154;

BB12_20:
add.s32 %r155, %r466, 3072;
setp.ge.s32	%p12, %r155, %r118;
@%p12 bra BB12_22;

ld.global.u32 %r156, [%rd126+12288];
st.shared.u32 [%r467+12288], %r156;

BB12_22:
add.s32 %r463, %r463, 4096;
add.s64 %rd126, %rd126, 16384;
add.s32 %r466, %r466, 4096;
setp.lt.s32	%p13, %r463, %r118;
add.s32 %r467, %r467, 16384;
@%p13 bra BB12_14;

BB12_23:
bar.sync 0;
add.s32 %r469, %r491, %r6;
setp.ge.s32	%p14, %r469, %r7;
@%p14 bra BB12_42;

cvta.to.global.u64 %rd9, %rd46;
add.s32 %r157, %r116, 31;
shr.s32 %r158, %r157, 31;
shr.u32 %r159, %r158, 27;
add.s32 %r160, %r157, %r159;
and.b32 %r28, %r160, -32;
mad.lo.s32 %r161, %r3, %r1, %r9;
mul.wide.s32 %rd63, %r161, 2;
add.s64 %rd10, %rd47, %rd63;
add.s32 %r162, %r5, %r9;
mul.wide.s32 %rd64, %r162, 2;
add.s64 %rd11, %rd47, %rd64;

BB12_25:
shl.b32 %r163, %r469, 1;
mul.wide.s32 %rd65, %r163, 4;
add.s64 %rd66, %rd9, %rd65;
add.s32 %r164, %r163, 1;
mul.wide.s32 %rd67, %r164, 4;
add.s64 %rd68, %rd9, %rd67;
ld.global.u32 %r30, [%rd68];
add.s32 %r165, %r30, -1;
setp.lt.s32	%p15, %r165, 0;
setp.ge.s32	%p16, %r165, %r118;
or.pred %p17, %p15, %p16;
ld.global.u32 %r31, [%rd66];
add.s32 %r166, %r31, -1;
setp.lt.s32	%p18, %r166, 0;
or.pred %p19, %p17, %p18;
setp.ge.s32	%p20, %r166, %r117;
or.pred %p21, %p19, %p20;
@%p21 bra BB12_41;

setp.ne.s32	%p22, %r9, 0;
@%p22 bra BB12_28;

shl.b32 %r167, %r30, 2;
mov.u32 %r168, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r169, %r167, %r168;
add.s32 %r462, %r31, -1;
st.shared.u32 [%r169+-4], %r462;

BB12_28:
setp.lt.s32	%p23, %r117, 1;
@%p23 bra BB12_36;

mul.lo.s32 %r32, %r117, %r469;
mov.u32 %r470, 0;

BB12_30:
setp.lt.s32	%p24, %r116, 1;
@%p24 bra BB12_35;

add.s32 %r173, %r32, %r470;
mad.lo.s32 %r174, %r116, %r173, %r9;
mul.wide.s32 %rd69, %r174, 2;
add.s64 %rd128, %rd2, %rd69;
mad.lo.s32 %r175, %r118, %r470, %r165;
mul.lo.s32 %r176, %r116, %r175;
mul.wide.s32 %rd70, %r176, 2;
add.s64 %rd127, %rd10, %rd70;
mov.u32 %r472, 0;
mov.u32 %r471, %r9;

BB12_32:
setp.ge.s32	%p25, %r471, %r116;
@%p25 bra BB12_34;

ld.global.u16 %rs54, [%rd128];

	{ atom.add.noftz.f16 %rs53,[%rd127],%rs54; }



BB12_34:
add.s32 %r472, %r472, 32;
add.s64 %rd128, %rd128, 64;
add.s32 %r471, %r471, 32;
add.s64 %rd127, %rd127, 64;
setp.lt.s32	%p26, %r472, %r28;
@%p26 bra BB12_32;

BB12_35:
add.s32 %r470, %r470, 1;
setp.lt.s32	%p27, %r470, %r117;
@%p27 bra BB12_30;

BB12_36:
setp.lt.s32	%p28, %r116, 1;
@%p28 bra BB12_41;

mad.lo.s32 %r178, %r117, %r469, %r31;
add.s32 %r179, %r178, -1;
mad.lo.s32 %r180, %r116, %r179, %r9;
mul.wide.s32 %rd72, %r180, 2;
add.s64 %rd130, %rd2, %rd72;
mul.lo.s32 %r182, %r116, %r165;
mul.wide.s32 %rd73, %r182, 2;
add.s64 %rd129, %rd11, %rd73;
mov.u32 %r474, 0;
mov.u32 %r473, %r9;

BB12_38:
setp.ge.s32	%p29, %r473, %r116;
@%p29 bra BB12_40;

ld.global.u16 %rs56, [%rd130];

	{mul.f16 %rs55,%rs56,%rs56;
}

	
	{ cvt.f32.f16 %f1, %rs55;}


	
	{ cvt.rn.f16.f32 %rs59, %f1;}


	
	{ atom.add.noftz.f16 %rs60,[%rd129],%rs59; }



BB12_40:
add.s32 %r474, %r474, 32;
add.s64 %rd130, %rd130, 64;
add.s32 %r473, %r473, 32;
add.s64 %rd129, %rd129, 64;
setp.lt.s32	%p30, %r474, %r28;
@%p30 bra BB12_38;

BB12_41:
add.s32 %r469, %r469, 32;
setp.lt.s32	%p31, %r469, %r7;
@%p31 bra BB12_25;

BB12_42:
bar.sync 0;
or.b32 %r183, %r9, %r491;
add.s32 %r184, %r118, %r118;
shl.b32 %r185, %r184, 2;
mov.u32 %r186, _ZN8nvinfer119sparse_fipnn_shared9smem_poolE;
add.s32 %r45, %r186, %r185;
setp.ne.s32	%p32, %r183, 0;
@%p32 bra BB12_76;

mov.u32 %r485, 0;
@%p1 bra BB12_75;

and.b32 %r194, %r118, 3;
mov.u32 %r476, 0;
setp.eq.s32	%p34, %r194, 0;
@%p34 bra BB12_45;

setp.eq.s32	%p35, %r194, 1;
@%p35 bra BB12_47;
bra.uni BB12_48;

BB12_47:
mov.u32 %r479, %r476;
bra.uni BB12_56;

BB12_45:
mov.u32 %r485, %r476;
bra.uni BB12_60;

BB12_48:
setp.eq.s32	%p36, %r194, 2;
@%p36 bra BB12_49;
bra.uni BB12_50;

BB12_49:
mov.u32 %r475, %r476;
bra.uni BB12_52;

BB12_50:
ld.shared.u32 %r197, [_ZN8nvinfer119sparse_fipnn_shared9smem_poolE];
mov.u32 %r475, 1;
setp.lt.s32	%p37, %r197, 0;
@%p37 bra BB12_52;

shl.b32 %r200, %r118, 2;
add.s32 %r202, %r186, %r200;
mov.u32 %r203, 0;
st.shared.u32 [%r202], %r203;
mov.u32 %r475, 1;
mov.u32 %r476, %r475;

BB12_52:
shl.b32 %r204, %r475, 2;
add.s32 %r206, %r186, %r204;
ld.shared.u32 %r207, [%r206];
setp.lt.s32	%p38, %r207, 0;
@%p38 bra BB12_53;

add.s32 %r479, %r476, 1;
add.s32 %r208, %r476, %r118;
shl.b32 %r209, %r208, 2;
add.s32 %r211, %r186, %r209;
st.shared.u32 [%r211], %r475;
bra.uni BB12_55;

BB12_53:
mov.u32 %r479, %r476;

BB12_55:
add.s32 %r476, %r475, 1;

BB12_56:
shl.b32 %r212, %r476, 2;
add.s32 %r214, %r186, %r212;
ld.shared.u32 %r215, [%r214];
setp.lt.s32	%p39, %r215, 0;
@%p39 bra BB12_57;

add.s32 %r485, %r479, 1;
add.s32 %r216, %r479, %r118;
shl.b32 %r217, %r216, 2;
add.s32 %r219, %r186, %r217;
st.shared.u32 [%r219], %r476;
bra.uni BB12_59;

BB12_57:
mov.u32 %r485, %r479;

BB12_59:
add.s32 %r476, %r476, 1;

BB12_60:
setp.lt.u32	%p40, %r118, 4;
@%p40 bra BB12_75;

shl.b32 %r220, %r476, 2;
add.s32 %r483, %r186, %r220;

BB12_62:
ld.shared.u32 %r222, [%r483];
setp.lt.s32	%p41, %r222, 0;
@%p41 bra BB12_63;

add.s32 %r486, %r485, 1;
add.s32 %r223, %r485, %r118;
shl.b32 %r224, %r223, 2;
add.s32 %r226, %r186, %r224;
st.shared.u32 [%r226], %r476;
bra.uni BB12_65;

BB12_63:
mov.u32 %r486, %r485;

BB12_65:
ld.shared.u32 %r227, [%r483+4];
setp.lt.s32	%p42, %r227, 0;
@%p42 bra BB12_66;

add.s32 %r487, %r486, 1;
add.s32 %r228, %r486, %r118;
shl.b32 %r229, %r228, 2;
add.s32 %r231, %r186, %r229;
add.s32 %r232, %r476, 1;
st.shared.u32 [%r231], %r232;
bra.uni BB12_68;

BB12_66:
mov.u32 %r487, %r486;

BB12_68:
ld.shared.u32 %r233, [%r483+8];
setp.lt.s32	%p43, %r233, 0;
@%p43 bra BB12_69;

add.s32 %r488, %r487, 1;
add.s32 %r234, %r487, %r118;
shl.b32 %r235, %r234, 2;
add.s32 %r237, %r186, %r235;
add.s32 %r238, %r476, 2;
st.shared.u32 [%r237], %r238;
bra.uni BB12_71;

BB12_69:
mov.u32 %r488, %r487;

BB12_71:
ld.shared.u32 %r239, [%r483+12];
setp.lt.s32	%p44, %r239, 0;
@%p44 bra BB12_72;

add.s32 %r485, %r488, 1;
add.s32 %r240, %r488, %r118;
shl.b32 %r241, %r240, 2;
add.s32 %r243, %r186, %r241;
add.s32 %r244, %r476, 3;
st.shared.u32 [%r243], %r244;
bra.uni BB12_74;

BB12_72:
mov.u32 %r485, %r488;

BB12_74:
add.s32 %r476, %r476, 4;
setp.lt.s32	%p45, %r476, %r118;
add.s32 %r483, %r483, 16;
@%p45 bra BB12_62;

BB12_75:
st.shared.u32 [%r45], %r485;

BB12_76:
bar.sync 0;
add.s32 %r245, %r116, 31;
shr.s32 %r246, %r245, 31;
shr.u32 %r247, %r246, 27;
add.s32 %r248, %r245, %r247;
and.b32 %r73, %r248, -32;
ld.shared.u32 %r74, [%r45];
setp.ge.s32	%p46, %r491, %r74;
@%p46 bra BB12_143;

setp.gt.s32	%p47, %r73, 32;
add.s32 %r249, %r73, -1;
shr.u32 %r250, %r249, 5;
add.s32 %r251, %r250, 1;
selp.b32	%r75, %r251, 1, %p47;
and.b32 %r77, %r75, 3;
mul.wide.s32 %rd75, %r4, 2;
add.s64 %rd24, %rd3, %rd75;
mul.lo.s32 %r253, %r1, %r118;
mul.lo.s32 %r254, %r253, %r116;
mul.lo.s32 %r256, %r254, %r124;
mul.wide.s32 %rd76, %r256, 2;
add.s64 %rd27, %rd3, %rd76;
mul.lo.s32 %r257, %r118, %r117;
mad.lo.s32 %r258, %r257, %r116, %r256;
mul.wide.s32 %rd77, %r258, 2;
add.s64 %rd28, %rd3, %rd77;

	{mov.u32 %r336, WARP_SZ;
}

	shl.b32 %r375, %r336, 8;
add.s32 %r376, %r375, -8192;
or.b32 %r340, %r376, 31;

BB12_78:
mov.u16 %rs259, 0;
st.local.v4.u16 [%rd5], {%rs259, %rs259, %rs259, %rs259};
st.local.v4.u16 [%rd5+8], {%rs259, %rs259, %rs259, %rs259};
st.local.v4.u16 [%rd5+16], {%rs259, %rs259, %rs259, %rs259};
add.s32 %r260, %r491, %r118;
shl.b32 %r261, %r260, 2;
add.s32 %r263, %r186, %r261;
ld.shared.u32 %r79, [%r263];
shl.b32 %r264, %r79, 2;
add.s32 %r265, %r186, %r264;
ld.shared.u32 %r80, [%r265];
add.s32 %r266, %r79, 2;
add.s32 %r267, %r79, 1;
mul.lo.s32 %r268, %r266, %r267;
shr.u32 %r269, %r268, 31;
add.s32 %r270, %r268, %r269;
shr.s32 %r271, %r270, 1;
sub.s32 %r81, %r271, %r267;
mov.u32 %r259, 0;

	cvt.rn.f16.s32 %rs62, %r259;

	st.local.u16 [%rd5], %rs62;
setp.lt.s32	%p48, %r116, 1;
@%p48 bra BB12_101;

mad.lo.s32 %r82, %r79, %r116, %r9;
add.s32 %r275, %r79, %r118;
mad.lo.s32 %r83, %r275, %r116, %r9;
mov.u32 %r492, 0;
setp.eq.s32	%p49, %r77, 0;
@%p49 bra BB12_90;

setp.eq.s32	%p50, %r77, 1;
@%p50 bra BB12_87;

setp.eq.s32	%p51, %r77, 2;
@%p51 bra BB12_84;

mov.u32 %r492, 32;
setp.ge.s32	%p52, %r9, %r116;
@%p52 bra BB12_84;

add.s32 %r278, %r82, %r4;
mul.wide.s32 %rd80, %r278, 2;
add.s64 %rd81, %rd3, %rd80;
ld.global.u16 %rs64, [%rd81];
st.local.u16 [%rd5], %rs64;
add.s32 %r279, %r83, %r4;
mul.wide.s32 %rd82, %r279, 2;
add.s64 %rd83, %rd3, %rd82;
ld.global.u16 %rs65, [%rd83];
st.local.u16 [%rd5+12], %rs65;

BB12_84:
add.s32 %r280, %r492, %r9;
setp.ge.s32	%p53, %r280, %r116;
@%p53 bra BB12_86;

shr.u32 %r281, %r492, 5;
add.s32 %r282, %r82, %r492;
add.s32 %r283, %r282, %r4;
mul.wide.s32 %rd84, %r283, 2;
add.s64 %rd85, %rd3, %rd84;
ld.global.u16 %rs66, [%rd85];
mul.wide.u32 %rd86, %r281, 2;
add.s64 %rd87, %rd5, %rd86;
st.local.u16 [%rd87], %rs66;
add.s32 %r284, %r83, %r492;
add.s32 %r285, %r284, %r4;
mul.wide.s32 %rd88, %r285, 2;
add.s64 %rd89, %rd3, %rd88;
ld.global.u16 %rs67, [%rd89];
st.local.u16 [%rd87+12], %rs67;

BB12_86:
add.s32 %r492, %r492, 32;

BB12_87:
add.s32 %r286, %r492, %r9;
setp.ge.s32	%p54, %r286, %r116;
@%p54 bra BB12_89;

shr.s32 %r287, %r492, 31;
shr.u32 %r288, %r287, 27;
add.s32 %r289, %r492, %r288;
shr.s32 %r290, %r289, 5;
add.s32 %r291, %r82, %r492;
add.s32 %r292, %r291, %r4;
mul.wide.s32 %rd90, %r292, 2;
add.s64 %rd91, %rd3, %rd90;
ld.global.u16 %rs68, [%rd91];
mul.wide.s32 %rd92, %r290, 2;
add.s64 %rd93, %rd5, %rd92;
st.local.u16 [%rd93], %rs68;
add.s32 %r293, %r83, %r492;
add.s32 %r294, %r293, %r4;
mul.wide.s32 %rd94, %r294, 2;
add.s64 %rd95, %rd3, %rd94;
ld.global.u16 %rs69, [%rd95];
st.local.u16 [%rd93+12], %rs69;

BB12_89:
add.s32 %r492, %r492, 32;

BB12_90:
setp.lt.u32	%p55, %r75, 4;
@%p55 bra BB12_101;

add.s32 %r495, %r9, %r492;
mad.lo.s32 %r295, %r116, %r79, %r495;
mul.wide.s32 %rd96, %r295, 2;
add.s64 %rd131, %rd27, %rd96;
add.s32 %r296, %r118, %r79;
mad.lo.s32 %r297, %r116, %r296, %r495;
mul.wide.s32 %rd97, %r297, 2;
add.s64 %rd132, %rd27, %rd97;

BB12_92:
setp.ge.s32	%p56, %r495, %r116;
@%p56 bra BB12_94;

shr.s32 %r298, %r492, 31;
shr.u32 %r299, %r298, 27;
add.s32 %r300, %r492, %r299;
shr.s32 %r301, %r300, 5;
ld.global.u16 %rs70, [%rd131];
mul.wide.s32 %rd98, %r301, 2;
add.s64 %rd99, %rd5, %rd98;
st.local.u16 [%rd99], %rs70;
ld.global.u16 %rs71, [%rd132];
st.local.u16 [%rd99+12], %rs71;

BB12_94:
add.s32 %r302, %r495, 32;
setp.ge.s32	%p57, %r302, %r116;
@%p57 bra BB12_96;

add.s32 %r303, %r492, 32;
shr.s32 %r304, %r303, 31;
shr.u32 %r305, %r304, 27;
add.s32 %r306, %r303, %r305;
shr.s32 %r307, %r306, 5;
ld.global.u16 %rs72, [%rd131+64];
mul.wide.s32 %rd100, %r307, 2;
add.s64 %rd101, %rd5, %rd100;
st.local.u16 [%rd101], %rs72;
ld.global.u16 %rs73, [%rd132+64];
st.local.u16 [%rd101+12], %rs73;

BB12_96:
add.s32 %r308, %r495, 64;
setp.ge.s32	%p58, %r308, %r116;
@%p58 bra BB12_98;

add.s32 %r309, %r492, 64;
shr.s32 %r310, %r309, 31;
shr.u32 %r311, %r310, 27;
add.s32 %r312, %r309, %r311;
shr.s32 %r313, %r312, 5;
ld.global.u16 %rs74, [%rd131+128];
mul.wide.s32 %rd102, %r313, 2;
add.s64 %rd103, %rd5, %rd102;
st.local.u16 [%rd103], %rs74;
ld.global.u16 %rs75, [%rd132+128];
st.local.u16 [%rd103+12], %rs75;

BB12_98:
add.s32 %r314, %r495, 96;
setp.ge.s32	%p59, %r314, %r116;
@%p59 bra BB12_100;

add.s32 %r315, %r492, 96;
shr.s32 %r316, %r315, 31;
shr.u32 %r317, %r316, 27;
add.s32 %r318, %r315, %r317;
shr.s32 %r319, %r318, 5;
ld.global.u16 %rs76, [%rd131+192];
mul.wide.s32 %rd104, %r319, 2;
add.s64 %rd105, %rd5, %rd104;
st.local.u16 [%rd105], %rs76;
ld.global.u16 %rs77, [%rd132+192];
st.local.u16 [%rd105+12], %rs77;

BB12_100:
add.s32 %r492, %r492, 128;
add.s32 %r495, %r495, 128;
setp.lt.s32	%p60, %r492, %r73;
add.s64 %rd131, %rd131, 256;
add.s64 %rd132, %rd132, 256;
@%p60 bra BB12_92;

BB12_101:
setp.lt.s32	%p61, %r491, 1;
@%p61 bra BB12_113;

mul.lo.s32 %r94, %r118, %r80;
mov.u32 %r497, 0;

BB12_103:
add.s32 %r321, %r497, %r118;
shl.b32 %r322, %r321, 2;
add.s32 %r324, %r186, %r322;
ld.shared.u32 %r96, [%r324];
mov.u16 %rs239, %rs62;
@%p48 bra BB12_110;

shl.b32 %r326, %r96, 2;
add.s32 %r328, %r186, %r326;
ld.shared.u32 %r97, [%r328];
add.s32 %r329, %r94, %r96;
mad.lo.s32 %r330, %r116, %r329, %r9;
mul.wide.s32 %rd106, %r330, 2;
add.s64 %rd133, %rd24, %rd106;
mov.u32 %r499, 0;
mov.u32 %r498, %r9;
mov.u16 %rs239, %rs62;

BB12_105:
setp.ge.s32	%p63, %r498, %r116;
mov.u16 %rs237, %rs62;
mov.u16 %rs238, %rs62;
@%p63 bra BB12_109;

ld.global.u16 %rs238, [%rd133];
shr.s32 %r331, %r499, 31;
shr.u32 %r332, %r331, 27;
add.s32 %r333, %r499, %r332;
shr.s32 %r334, %r333, 5;
mul.wide.s32 %rd107, %r334, 2;
add.s64 %rd38, %rd5, %rd107;
setp.eq.s32	%p64, %r97, 0;
@%p64 bra BB12_108;
bra.uni BB12_107;

BB12_108:
ld.local.u16 %rs237, [%rd38];
bra.uni BB12_109;

BB12_107:
ld.local.u16 %rs237, [%rd38+12];

BB12_109:

	{mul.f16 %rs78,%rs238,%rs237;
}

	
	{add.f16 %rs239,%rs239,%rs78;
}

	add.s32 %r498, %r498, 32;
add.s64 %rd133, %rd133, 64;
add.s32 %r499, %r499, 32;
setp.lt.s32	%p65, %r499, %r73;
@%p65 bra BB12_105;

BB12_110:
bar.warp.sync -1;

	{ mov.b32 %r335, {%rs239,%rs239};}


	mov.u32 %r363, 8;
mov.u32 %r339, 1;
mov.u32 %r373, -1;

	{shfl.sync.bfly.b32 %r337,%r335,%r339,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r337;
mov.b16 %rs86, low;}

	
	{add.f16 %rs87,%rs239,%rs86;
}

	
	{ mov.b32 %r343, {%rs87,%rs87};}


	mov.u32 %r347, 2;

	{shfl.sync.bfly.b32 %r345,%r343,%r347,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r345;
mov.b16 %rs92, low;}

	
	{add.f16 %rs93,%rs87,%rs92;
}

	
	{ mov.b32 %r351, {%rs93,%rs93};}


	mov.u32 %r355, 4;

	{shfl.sync.bfly.b32 %r353,%r351,%r355,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r353;
mov.b16 %rs98, low;}

	
	{add.f16 %rs99,%rs93,%rs98;
}

	
	{ mov.b32 %r359, {%rs99,%rs99};}


	
	{shfl.sync.bfly.b32 %r361,%r359,%r363,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r361;
mov.b16 %rs104, low;}

	
	{add.f16 %rs105,%rs99,%rs104;
}

	
	{ mov.b32 %r367, {%rs105,%rs105};}


	mov.u32 %r371, 16;

	{shfl.sync.bfly.b32 %r369,%r367,%r371,%r340,%r373;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r369;
mov.b16 %rs110, low;}

	
	{add.f16 %rs111,%rs105,%rs110;
}

	setp.ne.s32	%p66, %r9, 0;
@%p66 bra BB12_112;

add.s32 %r385, %r96, %r81;
add.s32 %r386, %r385, %r2;
mul.wide.s32 %rd108, %r386, 2;
add.s64 %rd109, %rd4, %rd108;
st.global.u16 [%rd109], %rs111;

BB12_112:
bar.warp.sync -1;
add.s32 %r497, %r497, 1;
setp.lt.s32	%p67, %r497, %r491;
@%p67 bra BB12_103;

BB12_113:
@%p48 bra BB12_114;

mad.lo.s32 %r390, %r80, %r118, %r79;
mad.lo.s32 %r103, %r390, %r116, %r9;
mad.lo.s32 %r104, %r79, %r116, %r9;
setp.eq.s32	%p69, %r77, 0;
@%p69 bra BB12_116;

setp.eq.s32	%p70, %r77, 1;
@%p70 bra BB12_118;
bra.uni BB12_119;

BB12_118:
mov.u16 %rs245, %rs62;
bra.uni BB12_126;

BB12_114:
mov.u16 %rs259, %rs62;
bra.uni BB12_140;

BB12_116:
mov.u16 %rs248, %rs62;
bra.uni BB12_129;

BB12_119:
setp.eq.s32	%p71, %r77, 2;
mov.u16 %rs242, %rs62;
@%p71 bra BB12_123;

setp.ge.s32	%p72, %r9, %r116;
mov.u16 %rs240, %rs62;
mov.u16 %rs241, %rs62;
@%p72 bra BB12_122;

add.s32 %r391, %r103, %r4;
mul.wide.s32 %rd110, %r391, 2;
add.s64 %rd111, %rd3, %rd110;
ld.global.u16 %rs241, [%rd111];
add.s32 %r392, %r104, %r5;
mul.wide.s32 %rd112, %r392, 2;
add.s64 %rd113, %rd3, %rd112;
ld.global.u16 %rs240, [%rd113];

BB12_122:
mov.f64 %fd1, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs115, %fd1;}


	
	{mul.f16 %rs116,%rs241,%rs241;
}

	
	{sub.f16 %rs119,%rs116,%rs240;
}

	
	{mul.f16 %rs122,%rs115,%rs119;
}

	
	{add.f16 %rs242,%rs62,%rs122;
}

	mov.u32 %r259, 32;

BB12_123:
add.s32 %r394, %r259, %r9;
setp.ge.s32	%p73, %r394, %r116;
mov.u16 %rs243, %rs62;
mov.u16 %rs244, %rs62;
@%p73 bra BB12_125;

add.s32 %r395, %r103, %r259;
add.s32 %r396, %r395, %r4;
mul.wide.s32 %rd114, %r396, 2;
add.s64 %rd115, %rd3, %rd114;
ld.global.u16 %rs244, [%rd115];
add.s32 %r397, %r104, %r259;
add.s32 %r398, %r397, %r5;
mul.wide.s32 %rd116, %r398, 2;
add.s64 %rd117, %rd3, %rd116;
ld.global.u16 %rs243, [%rd117];

BB12_125:
mov.f64 %fd2, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs128, %fd2;}


	
	{mul.f16 %rs129,%rs244,%rs244;
}

	
	{sub.f16 %rs132,%rs129,%rs243;
}

	
	{mul.f16 %rs135,%rs128,%rs132;
}

	
	{add.f16 %rs245,%rs242,%rs135;
}

	add.s32 %r259, %r259, 32;

BB12_126:
add.s32 %r399, %r259, %r9;
setp.ge.s32	%p74, %r399, %r116;
mov.u16 %rs246, %rs62;
mov.u16 %rs247, %rs62;
@%p74 bra BB12_128;

add.s32 %r400, %r103, %r259;
add.s32 %r401, %r400, %r4;
mul.wide.s32 %rd118, %r401, 2;
add.s64 %rd119, %rd3, %rd118;
ld.global.u16 %rs247, [%rd119];
add.s32 %r402, %r104, %r259;
add.s32 %r403, %r402, %r5;
mul.wide.s32 %rd120, %r403, 2;
add.s64 %rd121, %rd3, %rd120;
ld.global.u16 %rs246, [%rd121];

BB12_128:
mov.f64 %fd3, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs141, %fd3;}


	
	{mul.f16 %rs142,%rs247,%rs247;
}

	
	{sub.f16 %rs145,%rs142,%rs246;
}

	
	{mul.f16 %rs148,%rs141,%rs145;
}

	
	{add.f16 %rs248,%rs245,%rs148;
}

	add.s32 %r259, %r259, 32;
mov.u16 %rs259, %rs248;

BB12_129:
setp.lt.u32	%p75, %r75, 4;
@%p75 bra BB12_140;

add.s32 %r503, %r9, %r259;
mad.lo.s32 %r404, %r118, %r80, %r79;
mad.lo.s32 %r405, %r116, %r404, %r503;
mul.wide.s32 %rd122, %r405, 2;
add.s64 %rd134, %rd27, %rd122;
mad.lo.s32 %r406, %r116, %r79, %r503;
mul.wide.s32 %rd123, %r406, 2;
add.s64 %rd135, %rd28, %rd123;
mov.u16 %rs259, %rs248;

BB12_131:
setp.ge.s32	%p76, %r503, %r116;
mov.u16 %rs251, %rs62;
mov.u16 %rs252, %rs62;
@%p76 bra BB12_133;

ld.global.u16 %rs252, [%rd134];
ld.global.u16 %rs251, [%rd135];

BB12_133:
mov.f64 %fd4, 0d3FE0000000000000;

	{ cvt.rn.f16.f64 %rs154, %fd4;}


	
	{mul.f16 %rs155,%rs252,%rs252;
}

	
	{sub.f16 %rs158,%rs155,%rs251;
}

	
	{mul.f16 %rs161,%rs154,%rs158;
}

	
	{add.f16 %rs164,%rs259,%rs161;
}

	add.s32 %r407, %r503, 32;
setp.ge.s32	%p77, %r407, %r116;
mov.u16 %rs253, %rs62;
mov.u16 %rs254, %rs62;
@%p77 bra BB12_135;

ld.global.u16 %rs254, [%rd134+64];
ld.global.u16 %rs253, [%rd135+64];

BB12_135:

	{ cvt.rn.f16.f64 %rs167, %fd4;}


	
	{mul.f16 %rs168,%rs254,%rs254;
}

	
	{sub.f16 %rs171,%rs168,%rs253;
}

	
	{mul.f16 %rs174,%rs167,%rs171;
}

	
	{add.f16 %rs177,%rs164,%rs174;
}

	add.s32 %r408, %r503, 64;
setp.ge.s32	%p78, %r408, %r116;
mov.u16 %rs255, %rs62;
mov.u16 %rs256, %rs62;
@%p78 bra BB12_137;

ld.global.u16 %rs256, [%rd134+128];
ld.global.u16 %rs255, [%rd135+128];

BB12_137:

	{ cvt.rn.f16.f64 %rs180, %fd4;}


	
	{mul.f16 %rs181,%rs256,%rs256;
}

	
	{sub.f16 %rs184,%rs181,%rs255;
}

	
	{mul.f16 %rs187,%rs180,%rs184;
}

	
	{add.f16 %rs190,%rs177,%rs187;
}

	add.s32 %r409, %r503, 96;
setp.ge.s32	%p79, %r409, %r116;
mov.u16 %rs257, %rs62;
mov.u16 %rs258, %rs62;
@%p79 bra BB12_139;

ld.global.u16 %rs258, [%rd134+192];
ld.global.u16 %rs257, [%rd135+192];

BB12_139:

	{ cvt.rn.f16.f64 %rs193, %fd4;}


	
	{mul.f16 %rs194,%rs258,%rs258;
}

	
	{sub.f16 %rs197,%rs194,%rs257;
}

	
	{mul.f16 %rs200,%rs193,%rs197;
}

	
	{add.f16 %rs259,%rs190,%rs200;
}

	add.s32 %r503, %r503, 128;
add.s32 %r259, %r259, 128;
setp.lt.s32	%p80, %r259, %r73;
add.s64 %rd134, %rd134, 256;
add.s64 %rd135, %rd135, 256;
@%p80 bra BB12_131;

BB12_140:
bar.warp.sync -1;

	{ mov.b32 %r410, {%rs259,%rs259};}


	mov.u32 %r438, 8;
mov.u32 %r414, 1;
mov.u32 %r448, -1;

	{shfl.sync.bfly.b32 %r412,%r410,%r414,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r412;
mov.b16 %rs208, low;}

	
	{add.f16 %rs209,%rs259,%rs208;
}

	
	{ mov.b32 %r418, {%rs209,%rs209};}


	mov.u32 %r422, 2;

	{shfl.sync.bfly.b32 %r420,%r418,%r422,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r420;
mov.b16 %rs214, low;}

	
	{add.f16 %rs215,%rs209,%rs214;
}

	
	{ mov.b32 %r426, {%rs215,%rs215};}


	mov.u32 %r430, 4;

	{shfl.sync.bfly.b32 %r428,%r426,%r430,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r428;
mov.b16 %rs220, low;}

	
	{add.f16 %rs221,%rs215,%rs220;
}

	
	{ mov.b32 %r434, {%rs221,%rs221};}


	
	{shfl.sync.bfly.b32 %r436,%r434,%r438,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r436;
mov.b16 %rs226, low;}

	
	{add.f16 %rs227,%rs221,%rs226;
}

	
	{ mov.b32 %r442, {%rs227,%rs227};}


	mov.u32 %r446, 16;

	{shfl.sync.bfly.b32 %r444,%r442,%r446,%r340,%r448;
}

	
	{.reg .f16 low,high;
mov.b32 {low,high}, %r444;
mov.b16 %rs232, low;}

	
	{add.f16 %rs233,%rs227,%rs232;
}

	setp.ne.s32	%p81, %r9, 0;
@%p81 bra BB12_142;

add.s32 %r460, %r81, %r79;
add.s32 %r461, %r460, %r2;
mul.wide.s32 %rd124, %r461, 2;
add.s64 %rd125, %rd4, %rd124;
st.global.u16 [%rd125], %rs233;

BB12_142:
bar.warp.sync -1;
add.s32 %r491, %r491, 32;
setp.lt.s32	%p82, %r491, %r74;
@%p82 bra BB12_78;

BB12_143:
ret;
}


