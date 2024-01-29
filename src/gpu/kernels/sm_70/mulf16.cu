
#include<cuda_fp16.h>

__global__ void mulf16(__half* x1, __half* x2, __half* y) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    y[index] = __hmul(x1[index], x2[index]);
}