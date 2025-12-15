#include <cuda_runtime.h>


// Stencil coefficients
__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

__global__ void stencilBasic(float *u_in, float *u_out, int N) {
    // 3D thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Boundary check
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        int idx = k*N*N + j*N + i;
        int idx_left = k*N*N + j*N + (i-1);
        int idx_right = k*N*N + j*N + (i+1);
        int idx_down = k*N*N + (j-1)*N + i;
        int idx_up = k*N*N + (j+1)*N + i;
        int idx_front = (k-1)*N*N + j*N + i;
        int idx_back = (k+1)*N*N + j*N + i;
        
        u_out[idx] = c0*u_in[idx] + c1*u_in[idx_left] + c2*u_in[idx_right]
                   + c3*u_in[idx_down] + c4*u_in[idx_up]
                   + c5*u_in[idx_front] + c6*u_in[idx_back];
    }
}
