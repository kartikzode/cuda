#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16


__global__ void matrixMulKernel(float* M, float* N, float* P,
                                int Width,
                                unsigned Mds_sz, unsigned Nds_sz) {
    // Declare dynamic shared memory
    extern __shared__ float shared_mem[];


    // blockDim.x acts as our Tile Width (assuming square blocks)
    int tw = blockDim.x; 
    
    float* Mds = &shared_mem[0];           // Starts at index 0
    float* Nds = &shared_mem[tw * tw];     // Starts after Mds
    
    // Standard coordinates
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int row = by * tw + ty;
    int col = bx * tw + tx;

    float value = 0.0f;

    for (int ph = 0; ph < (Width + tw - 1) / tw; ++ph) {
                
        // Load M
        if (row < Width && (ph * tw + tx) < Width)
            Mds[ty * tw + tx] = M[row * Width + ph * tw + tx];
        else
            Mds[ty * tw + tx] = 0.0f;

        // Load N
        if ((ph * tw + ty) < Width && col < Width)
            Nds[ty * tw + tx] = N[(ph * tw + ty) * Width + col];
        else
            Nds[ty * tw + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tw; k++) {
            value += Mds[ty * tw + k] * Nds[k * tw + tx];
        }
        
        __syncthreads();
    }

    if (row < Width && col < Width) {
        P[row * Width + col] = value;
    }
}

static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // ----- problem size -----
    const int Width = 1024;      // square matrices Width x Width
    const int numElems = Width * Width;
    const size_t bytes = numElems * sizeof(float);

    // ----- host allocations -----
    float *h_M = (float*)malloc(bytes);
    float *h_N = (float*)malloc(bytes);
    float *h_P = (float*)malloc(bytes);
    if (!h_M || !h_N || !h_P) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // initialize input matrices (simple pattern)
    for (int i = 0; i < numElems; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    // ----- device allocations -----
    float *d_M, *d_N, *d_P;
    check(cudaMalloc((void**)&d_M, bytes), "cudaMalloc d_M");
    check(cudaMalloc((void**)&d_N, bytes), "cudaMalloc d_N");
    check(cudaMalloc((void**)&d_P, bytes), "cudaMalloc d_P");

    // copy inputs to device
    check(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice), "Memcpy h_M->d_M");
    check(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice), "Memcpy h_N->d_N");

    // ----- device properties -----
    cudaDeviceProp devProp;
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    check(cudaGetDeviceProperties(&devProp, dev), "cudaGetDeviceProperties");

    // shared memory per block
    size_t maxShared = devProp.sharedMemPerBlock;

    size_t size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float); 
    if (size > maxShared) {
        fprintf(stderr,
                "Requested shared memory %zu exceeds device limit %zu\n",
                size, maxShared);
        return EXIT_FAILURE;
    }

    unsigned Mds_sz = (unsigned)(size / 2);
    unsigned Nds_sz = (unsigned)(size / 2);

    // ----- launch configuration -----
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    // ----- kernel launch -----
    matrixMulKernel<<<dimGrid, dimBlock, size>>>(d_M, d_N, d_P,
                                                 Width,
                                                 Mds_sz, Nds_sz);
    check(cudaGetLastError(), "Kernel launch");
    check(cudaDeviceSynchronize(), "Kernel sync");

    // ----- copy result back & basic check -----
    check(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost), "Memcpy d_P->h_P");

    printf("P[0] = %f\n", h_P[0]);  // quick sanity print

    // ----- cleanup -----
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
