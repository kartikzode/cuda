#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) {
        output[x] = (input[x] > 0.0 ? input[x]: 0.0f );
    }

}