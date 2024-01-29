
__global__ void mulf64(double* x1, double* x2, double* y) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    y[index] = x1[index] * x2[index];
}