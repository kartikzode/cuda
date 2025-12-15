#include <cuda_runtime.h>

#define RADIUS 1
#define TILE_SIZE 8

// Stencil coefficients
__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

__global__ void stencilTiled(float *u_in, float *u_out, int N) {
    // Input tile with halo cells
    __shared__ float tile[(TILE_SIZE+2)*(TILE_SIZE+2)*(TILE_SIZE+2)];
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Global indices
    int i = blockIdx.x * TILE_SIZE + tx - RADIUS;
    int j = blockIdx.y * TILE_SIZE + ty - RADIUS;
    int k = blockIdx.z * TILE_SIZE + tz - RADIUS;
    
    // Load input tile into shared memory
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        int global_idx = k*N*N + j*N + i;
        int local_idx = (tz)*(TILE_SIZE+2)*(TILE_SIZE+2) + 
                       (ty)*(TILE_SIZE+2) + tx;
        tile[local_idx] = u_in[global_idx];
    } else {
        // Boundary condition (zero for ghost cells)
        int local_idx = (tz)*(TILE_SIZE+2)*(TILE_SIZE+2) + 
                       (ty)*(TILE_SIZE+2) + tx;
        tile[local_idx] = 0.0f;
    }
    
    // Synchronize all threads to ensure tile is fully loaded
    __syncthreads();
    
    // Compute output (only for interior threads)
    if (tx >= RADIUS && tx < TILE_SIZE + RADIUS &&
        ty >= RADIUS && ty < TILE_SIZE + RADIUS &&
        tz >= RADIUS && tz < TILE_SIZE + RADIUS &&
        i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        
        int local_idx = (tz)*(TILE_SIZE+2)*(TILE_SIZE+2) + 
                       (ty)*(TILE_SIZE+2) + tx;
        int tile_size_plus = TILE_SIZE + 2;
        
        float result = c0 * tile[local_idx]
                     + c1 * tile[local_idx - 1]           // i-1
                     + c2 * tile[local_idx + 1]           // i+1
                     + c3 * tile[local_idx - tile_size_plus]      // j-1
                     + c4 * tile[local_idx + tile_size_plus]      // j+1
                     + c5 * tile[local_idx - tile_size_plus*tile_size_plus]  // k-1
                     + c6 * tile[local_idx + tile_size_plus*tile_size_plus]; // k+1
        
        // Write to global memory
        int global_idx = k*N*N + j*N + i;
        u_out[global_idx] = result;
    }
}
