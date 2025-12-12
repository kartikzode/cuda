#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 1024
#define TILE_DIM 32
#define BLOCK_ROWS 8

// CUDA kernel for coalesced matrix transposition
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {

    __shared__ float tile [TILE_DIM] [TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM  + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i+= BLOCK_ROWS) {
        
        tile[threadIdx.y + i][threadIdx.x] = input[(y+i) * width + x];
    }
    __syncthreads();

    //transpose block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // write to output
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
        output[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }


}

// kernel with tile size equal to block sizwe
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile [TILE_DIM] [TILE_DIM];

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y*cols + x];
    } 
    
    __syncthreads();

    //transpose offset
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        output[y*rows + x] = tile [threadIdx.x][threadIdx.y];
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
    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // // Define block and grid sizes
    // dim3 blockSize(TILE_DIM, TILE_DIM);
    // dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

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