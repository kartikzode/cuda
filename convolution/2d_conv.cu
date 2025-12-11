#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2*FILTER_RADIUS)
#define FILTER_SIZE (2*FILTER_RADIUS + 1)
__constant__ float d_filter[FILTER_SIZE][FILTER_SIZE];

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// 2D Tiled Convolution Kernel with constant memory
__global__ void convolution_tiled_2d_const_mem_kernel(float *N, float *P, int width, int height) {
    
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    // Load input tile into shared memory
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Calculate output
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && 
        tileRow >= 0 && tileRow < OUT_TILE_DIM) {
        
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                Pvalue += d_filter[fRow][fCol] * 
                         N_s[tileRow + fRow][tileCol + fCol];
            }
        }
        P[row * width + col] = Pvalue;
        }
    }
}

int main() {
    printf("========== 2D Tiled Convolution with Constant Memory ==========\n\n");
    
    // Image dimensions
    int width = 64;
    int height = 64;
    int output_width = width - 2*FILTER_RADIUS;
    int output_height = height - 2*FILTER_RADIUS;
    
    printf("Input image: %d x %d\n", width, height);
    printf("Filter size: %d x %d (FILTER_RADIUS = %d)\n", FILTER_SIZE, FILTER_SIZE, FILTER_RADIUS);
    printf("Output image: %d x %d\n\n", output_width, output_height);
    
    // Allocate host memory
    size_t input_size = width * height * sizeof(float);
    size_t output_size = output_width * output_height * sizeof(float);
    size_t filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(output_size);
    float *h_filter = (float*)malloc(filter_size);
    
    // Initialize host input with some data
    printf("Initializing input image...\n");
    for (int i = 0; i < width * height; i++) {
        h_input[i] = sinf(i * 0.01f) + 2.0f;  // Some pattern
    }
    
    // Initialize host filter (simple blur-like kernel)
    printf("Initializing filter...\n");
    float filter_data[FILTER_SIZE][FILTER_SIZE] = {
        {0.04f, 0.06f, 0.08f, 0.06f, 0.04f},
        {0.06f, 0.09f, 0.12f, 0.09f, 0.06f},
        {0.08f, 0.12f, 0.16f, 0.12f, 0.08f},
        {0.06f, 0.09f, 0.12f, 0.09f, 0.06f},
        {0.04f, 0.06f, 0.08f, 0.06f, 0.04f}
    };
    memcpy(h_filter, filter_data, filter_size);
    
    // Print filter
    printf("Filter (5x5):\n");
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            printf("%.2f ", h_filter[i * FILTER_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, filter_size));
    
    printf("Data copied to device (filter in constant memory)\n\n");
    
    // Launch kernel
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((output_width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (output_height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    printf("Launching kernel with:\n");
    printf("  Block size: %d x %d = %d threads\n", threads.x, threads.y, threads.x * threads.y);
    printf("  Grid size: %d x %d = %d blocks\n\n", blocks.x, blocks.y, blocks.x * blocks.y);
    
    convolution_tiled_2d_const_mem_kernel<<<blocks, threads>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Kernel execution completed!\n\n");
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Print sample results
    printf("Sample input values (first 5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", h_input[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Sample output values (first 5x5):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.4f ", h_output[i * output_width + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Manual verification of output[0][0]
    // This corresponds to input region starting at input[FILTER_RADIUS][FILTER_RADIUS]
    printf("Verification of output[0][0]:\n");
    printf("(This uses input region [%d:%d, %d:%d])\n\n", 
           FILTER_RADIUS, FILTER_RADIUS + FILTER_SIZE,
           FILTER_RADIUS, FILTER_RADIUS + FILTER_SIZE);
    
    float expected = 0.0f;
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
            int inRow = FILTER_RADIUS + fRow;
            int inCol = FILTER_RADIUS + fCol;
            expected += h_filter[fRow * FILTER_SIZE + fCol] * 
                       h_input[inRow * width + inCol];
        }
    }
    printf("  Expected: %.4f\n", expected);
    printf("  Got: %.4f\n", h_output[0 * output_width + 0]);
    printf("  Match: %s\n\n", fabs(expected - h_output[0 * output_width + 0]) < 0.001f ? "YES" : "NO");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_filter);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("Test completed successfully!\n");
    return 0;
}
