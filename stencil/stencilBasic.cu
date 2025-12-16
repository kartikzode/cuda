#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>


// Stencil coefficients in constant memory
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

__global__ void stencilBasic(float *u_in, float *u_out, unsigned int N) {
    // 3D thread indices
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    
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
    stencilBasic<<<gridSize, blockSize>>>(d_u_in, d_u_out, N);
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