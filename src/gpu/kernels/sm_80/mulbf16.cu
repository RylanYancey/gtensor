
#include<cuda_bf16.h>

__global__ void mulbf16(__nv_bfloat16* x1, __nv_bfloat16* x2, __nv_bfloat16* y) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    y[index] = __hmul(x1[index], x2[index]);
}