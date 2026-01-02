#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCKSIZE 32

// Shared Memory GEMM Kernel
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
        const float *A, const float *B, float beta, float *C) {
                                        
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float Asub[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bsub[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        
        Asub[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bsub[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
        tmp += Asub[threadRow * BLOCKSIZE + dotIdx] *
                Bsub[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}