#include <cuda_runtime.h>

#define TILE_SIZE 32
#define COARSENING_FACTOR 4

// Stencil coefficients
__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

__global__ void stencilCoarsened(float *u_in, float *u_out, int N) {
    // Shared memory for current and adjacent z-planes
    __shared__ float inPrev[TILE_SIZE+2][TILE_SIZE+2];
    __shared__ float inCurr[TILE_SIZE+2][TILE_SIZE+2];
    __shared__ float inNext[TILE_SIZE+2][TILE_SIZE+2];
    
    // 2D thread block (32x32)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Starting coordinates
    int i = blockIdx.x * TILE_SIZE + tx - 1;
    int j = blockIdx.y * TILE_SIZE + ty - 1;
    int k_start = blockIdx.z * COARSENING_FACTOR - 1;
    
    // Load initial z-planes
    // Load k_start plane into inPrev
    if (i >= 0 && i < N && j >= 0 && j < N) {
        int gk = k_start;
        if (gk >= 0 && gk < N) {
            inPrev[ty][tx] = u_in[gk*N*N + j*N + i];
        } else {
            inPrev[ty][tx] = 0.0f;
        }
    }
    
    // Load k_start+1 plane into inCurr
    if (i >= 0 && i < N && j >= 0 && j < N) {
        int gk = k_start + 1;
        if (gk >= 0 && gk < N) {
            inCurr[ty][tx] = u_in[gk*N*N + j*N + i];
        } else {
            inCurr[ty][tx] = 0.0f;
        }
    }
    
    // Main iteration loop in z-direction
    for (int kiter = 0; kiter < COARSENING_FACTOR; kiter++) {
        int k_out = k_start + 1 + kiter;
        int k_next = k_start + 2 + kiter;
        
        __syncthreads();
        
        // Load next z-plane into inNext
        if (tx < TILE_SIZE+2 && ty < TILE_SIZE+2) {
            if (i >= 0 && i < N && j >= 0 && j < N) {
                if (k_next >= 0 && k_next < N) {
                    inNext[ty][tx] = u_in[k_next*N*N + j*N + i];
                } else {
                    inNext[ty][tx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute stencil for this z-plane
        if (tx >= 1 && tx < TILE_SIZE+1 && ty >= 1 && ty < TILE_SIZE+1 &&
            i >= 1 && i < N-1 && j >= 1 && j < N-1 && 
            k_out >= 1 && k_out < N-1) {
            
            float result = c0 * inCurr[ty][tx]
                         + c1 * inCurr[ty][tx-1]      // i-1
                         + c2 * inCurr[ty][tx+1]      // i+1
                         + c3 * inCurr[ty-1][tx]      // j-1
                         + c4 * inCurr[ty+1][tx]      // j+1
                         + c5 * inPrev[ty][tx]         // k-1
                         + c6 * inNext[ty][tx];        // k+1
            
            int global_idx = k_out*N*N + j*N + i;
            u_out[global_idx] = result;
        }
        
        // Shift planes for next iteration
        if (kiter < COARSENING_FACTOR - 1) {
            __syncthreads();
            if (tx < TILE_SIZE+2 && ty < TILE_SIZE+2) {
                float temp = inCurr[ty][tx];
                inCurr[ty][tx] = inNext[ty][tx];
                inPrev[ty][tx] = temp;
            }
        }
    }
}
