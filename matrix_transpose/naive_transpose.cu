#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    // Calculate the row and column index of the element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Perform the transposition if within bounds
    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

// Host function to check for CUDA errors
static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {
    int width = WIDTH;
    int height = HEIGHT;

    // Allocate host memory
    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize the input matrix with some values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    check(cudaMalloc((void**)&d_input, size), "Failed to allocate device memory for input");
    check(cudaMalloc((void**)&d_output, size), "Failed to allocate device memory for output");

    // Copy data from host to device
    check(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Failed to copy input data to device");

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    check(cudaGetLastError(), "Kernel launch failed");
    check(cudaDeviceSynchronize(), "Kernel sync");

    // Copy the result back to the host
    check(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost), "Failed to copy output data to host");

    // Verify the result
    bool success = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
                break;
            }
        }
    }
    if (success) {
        printf("Matrix transposition successful!\n");
    } else {
        printf("Matrix transposition failed!\n");
    }



    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}