#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2*FILTER_RADIUS)
#define FILTER_SIZE (2*FILTER_RADIUS + 1)
__constant__ float d_filter[FILTER_SIZE][FILTER_SIZE];

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
