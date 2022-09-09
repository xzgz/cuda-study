nvcc -c ../kernels.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
nvcc -c ../kernels.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep --dryrun
cuobjdump -ptx kernels.o > kernels.o.ptx
cuobjdump -sass kernels.sm_75.cubin > kernels.sm_75.cubin.sass
cuobjdump kernels.o -lelf

nvcc -I/usr/local/cuda-10.2/include -I/home/yckj3211/workspace/OpenBLAS/install/include -I../../include \
     -L/usr/local/cuda-10.2/lib64 -L/home/yckj3211/workspace/OpenBLAS/install/lib -lopenblas -lcublas -std=c++11 \
     -c ../interfaces.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
nvcc -I/usr/local/cuda-10.2/include -I../../include  -std=c++11      -c ../interfaces.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
cuobjdump -ptx interfaces.o > interfaces.o.ptx
cuobjdump -sass interfaces.sm_75.cubin > interfaces.sm_75.cubin.sass
cuobjdump interfaces.o -lelf

nvcc -I../../include -c ../cudaTensorCoreGemm.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep

nvcc -o clk_8x32x16HHF_RR ../clk_8x32x16HHF_RR.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep

