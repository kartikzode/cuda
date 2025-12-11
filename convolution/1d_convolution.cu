#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define KERNEL_SIZE 3

// Declare constant memory for the kernel filter
__constant__ float d_kernel[KERNEL_SIZE];

// 1D Convolution kernel using constant memory
__global__ void conv1d_kernel(float *input, float *output, 
                               int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size - KERNEL_SIZE + 1) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum += input[idx + k] * d_kernel[k];
        }
        output[idx] = sum;
    }
}

int main() {
    // Host data
    int input_size = 10;
    int output_size = input_size - KERNEL_SIZE + 1;
    
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float h_kernel[] = {0.5f, 0.3f, 0.2f};
    float h_output[output_size];
    float *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float)));
    
    // Launch kernel
    int threads_per_block = 32;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    conv1d_kernel<<<blocks, threads_per_block>>>(d_input, d_output, input_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    printf("\nInput: ");
    for (int i = 0; i < input_size; i++) {
        printf("%.1f ", h_input[i]);
    }
    printf("\n");
    
    printf("Kernel (in constant memory): ");
    for (int i = 0; i < KERNEL_SIZE; i++) {
        printf("%.1f ", h_kernel[i]);
    }
    printf("\n");
    
    printf("Output: ");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");
    
    // Verify with manual calculation for first element
    printf("\nVerification:\n");
    float expected = h_input[0] * h_kernel[0] + h_input[1] * h_kernel[1] + h_input[2] * h_kernel[2];
    printf("First output element - Expected: %.2f, Got: %.2f\n", expected, h_output[0]);
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\nKernel execution completed successfully!\n");
    return 0;
}
