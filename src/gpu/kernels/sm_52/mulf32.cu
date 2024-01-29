
__global__ void mulf32(float* x1, float* x2, float* y) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    y[index] = x1[index] * x2[index];
}
