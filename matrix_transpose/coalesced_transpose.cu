#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(float *odata, const float *idata, int width, int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
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
    dim3 blockSize(TILE_DIM, BLOCK_ROWS, 1);
    dim3 gridSize((width / TILE_DIM), (height / TILE_DIM), 1);

    // Launch the kernel
    transposeCoalesced<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    check(cudaGetLastError(), "Kernel launch failed");
    check(cudaDeviceSynchronize(), "Kernel sync");


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}