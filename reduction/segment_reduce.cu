#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void SharedMemoryReduction(float* input, float* output, int n) {
    __shared__ float input_s[BLOCK_DIM]; 
    unsigned int segment = 2* blockDim.x * blockIdx.x;
    unsigned int idx = segment + threadIdx.x; // global index
    unsigned int t = threadIdx.x; // index within tile

    // Load elements into shared memory
    float val1 = (idx < n) ? input[idx] : 0.0f;
    float val2 = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    input_s[t] = val1 + val2;


    // Reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    // Reduction across blocks using atomic add
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    // Size of the input data
    const int size = 1<<20;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    float zero = 0.0f;
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    int eff_coverage = 2*BLOCK_DIM;
    int numBlocks = (size + eff_coverage - 1) / eff_coverage;
    SharedMemoryReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum is " << *h_output << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}