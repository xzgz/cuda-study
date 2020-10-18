.global main
main:
mov $0x1000000, %rax
vxorps %ymm0, %ymm0, %ymm0
vxorps %ymm1, %ymm1, %ymm1
vxorps %ymm2, %ymm2, %ymm2
vxorps %ymm3, %ymm3, %ymm3
loop:
vmulps %ymm1, %ymm1, %ymm0
vaddps %ymm3, %ymm3, %ymm2
subq $0x1, %rax
jne loop
