#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define OUT_TILE_DIM 8
#define RADIUS 1

// Stencil coefficients
__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

// stencil coefficients in host memory
float h_c0 = 0.5f;
float h_c1 = 0.1f;
float h_c2 = 0.1f;
float h_c3 = 0.1f;
float h_c4 = 0.1f;
float h_c5 = 0.05f;
float h_c6 = 0.05f;



#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }


__global__ void stencilTiled(float *u_in, float *u_out, int N) {
    // Input tile with halo cells
    __shared__ float tile[OUT_TILE_DIM+2*RADIUS][OUT_TILE_DIM+2*RADIUS][OUT_TILE_DIM+2*RADIUS];
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Global indices
    int i = blockIdx.x * OUT_TILE_DIM + tx - RADIUS;
    int j = blockIdx.y * OUT_TILE_DIM + ty - RADIUS;
    int k = blockIdx.z * OUT_TILE_DIM + tz - RADIUS;
    
    // Load input tile into shared memory
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        tile[tz][ty][tx] = u_in[k*N*N + j*N + i];
    } else {
        tile[tz][ty][tx] = 0.0f;
    }
    
    // Synchronize all threads to ensure tile is fully loaded
    __syncthreads();
    
    // Compute output (only for interior threads)
    if (tx >= RADIUS && tx < OUT_TILE_DIM + RADIUS &&
        ty >= RADIUS && ty < OUT_TILE_DIM + RADIUS &&
        tz >= RADIUS && tz < OUT_TILE_DIM + RADIUS &&
        i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        
        float result = c0 * tile[tz][ty][tx]
                     + c1 * tile[tz][ty][tx-1]         // i-1
                     + c2 * tile[tz][ty][tx+1]           // i+1
                     + c3 * tile[tz][ty-1][tx]     // j-1
                     + c4 * tile[tz][ty+1][tx]      // j+1
                     + c5 * tile[tz+1][ty][tx]  // k-1
                     + c6 * tile[tz-1][ty][tx]; // k+1
        
        // Write to global memory
        int global_idx = k*N*N + j*N + i;
        u_out[global_idx] = result;
    }
}

int main() {
    const unsigned int N = 128; // Size of the 3D grid
    const size_t size = N * N * N * sizeof(float);
    
    // Allocate host memory
    float *h_u_in = (float*)malloc(size);
    float *h_u_out = (float*)malloc(size);
    
    // Initialize input data
    for (unsigned int i = 0; i < N*N*N; i++) {
        h_u_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_u_in, *d_u_out;
    CUDA_CHECK(cudaMalloc((void**)&d_u_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_u_out, size));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_u_in, h_u_in, size, cudaMemcpyHostToDevice));
    
    // Define block and grid sizes
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y,
                  (N + blockSize.z - 1) / blockSize.z);
    
    // Launch the stencil kernel
    stencilTiled<<<gridSize, blockSize>>>(d_u_in, d_u_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy output data back to host
    cudaMemcpy(h_u_out, d_u_out, size, cudaMemcpyDeviceToHost);

    // verify a few output values
    for (unsigned int k = 1; k < 3; k++) {
        for (unsigned int j = 1; j < 3; j++) {
            for (unsigned int i = 1; i < 3; i++) {
                unsigned int idx = k*N*N + j*N + i;
                h_u_out[idx] == h_c0*h_u_in[idx] + h_c1*h_u_in[k*N*N + j*N + (i-1)] + h_c2*h_u_in[k*N*N + j*N + (i+1)]
                   + h_c3*h_u_in[k*N*N + (j-1)*N + i] + h_c4*h_u_in[k*N*N + (j+1)*N + i]
                   + h_c5*h_u_in[(k-1)*N*N + j*N + i] + h_c6*h_u_in[(k+1)*N*N + j*N + i];
                printf("stencil output at (%u, %u, %u): %f\n", i, j, k, h_u_out[idx]);
                printf("expected output at (%u, %u, %u): %f\n", i, j, k,
                    h_c0*h_u_in[idx] + h_c1*h_u_in[k*N*N + j*N + (i-1)] + h_c2*h_u_in[k*N*N + j*N + (i+1)]
                   + h_c3*h_u_in[k*N*N + (j-1)*N + i] + h_c4*h_u_in[k*N*N + (j+1)*N + i]
                   + h_c5*h_u_in[(k-1)*N*N + j*N + i] + h_c6*h_u_in[(k+1)*N*N + j*N + i]);
                printf("\n");
            }
        }
    }
    
    // Free device memory
    cudaFree(d_u_in);
    cudaFree(d_u_out);
    
    // Free host memory
    free(h_u_in);
    free(h_u_out);
    
    return 0;
}
