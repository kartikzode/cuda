#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 256
#define COARSE_FACTOR 2

__global__ void CoarsenedReduction(float* input, float* output, int size) {

    __shared__ float input_s[BLOCK_DIM];
    int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    
    float sum = 0.0f;
    if (i < size) {
        sum = input[i];
    }

    // Reduce within a thread
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        unsigned int index = i + tile * blockDim.x;
        if (index < size) {
            sum += input[index];
        }
    }

    input_s[t] = sum;
    
    //Reduce within a block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    __syncthreads();
    //Reduce over blocks
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    const int size = 10325;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;  // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));  // Initialize output to 0

    // Launch the kernel with coarsening
    int eff_coverage = BLOCK_DIM*COARSE_FACTOR*2;
    int numBlocks = (size + eff_coverage - 1) / eff_coverage;
    CoarsenedReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);

    cudaDeviceSynchronize();
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