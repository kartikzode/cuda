#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16
#define COARSE_FACTOR 4

//CUDA kernel for matric multiplication
__global__
void matmul_tile( float* A, float * B, float* C, int Width) {
    
    __shared__ float Ads [TILE_WIDTH] [TILE_WIDTH];
    __shared__ float Bds [TILE_WIDTH] [TILE_WIDTH];

    int bx = blockIdx.x ; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Get the row and first column of the C matrix to work on
    int row = by*TILE_WIDTH + ty;
    int col_start = COARSE_FACTOR*bx*TILE_WIDTH + tx;

    //Initialize value for all output elements
    float value[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; c++) {
        value[c] = 0.0f;
    }

    // Loop over the A and B tiles required to compute C elements
    for(int ph = 0; ph < ceil(Width/(float) TILE_WIDTH); ++ph) {
        
        // Collaborative loading of A and B tiles into Shares Memory
        Ads[ty][tx] =  (((row < Width) && ((ph*TILE_WIDTH + tx) < Width)) ? A[row*Width + ph*TILE_WIDTH + tx] : 0.f);

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = col_start + c*TILE_WIDTH;
            Bds[ty][tx] = ((((ph*TILE_WIDTH + ty) < Width) && (col < Width)) ? B[(ph*TILE_WIDTH + ty)*Width + col] : 0.f);
        
            __syncthreads();

            for(int k=0; k < TILE_WIDTH; k++) {
                value[c] += Ads[ty][k] * Bds[k][tx];
            }
            __syncthreads();
    }

    }
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = col_start + c * TILE_WIDTH;
        if (row < Width && col < Width) {
        C[row*Width + col] = value[c];
    }

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

    // // ----- device properties -----
    // cudaDeviceProp devProp;
    // int dev = 0;
    // check(cudaGetDevice(&dev), "cudaGetDevice");
    // check(cudaGetDeviceProperties(&devProp, dev), "cudaGetDeviceProperties");

    // // shared memory per block
    // size_t maxShared = devProp.sharedMemPerBlock;

    // size_t size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float); 
    // if (size > maxShared) {
    //     fprintf(stderr,
    //             "Requested shared memory %zu exceeds device limit %zu\n",
    //             size, maxShared);
    //     return EXIT_FAILURE;
    // }

    // unsigned Mds_sz = (unsigned)(size / 2);
    // unsigned Nds_sz = (unsigned)(size / 2);

    // ----- launch configuration -----
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    // ----- kernel launch -----
    matmul_tile<<<dimGrid, dimBlock>>>(d_M, d_N, d_P,
                                                 Width);
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
