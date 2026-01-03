#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// GEMM Kernel with 1D block tiling
template<const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blockTiling(int M, int N, int K, float alpha,
        const float *A, const float *B, float beta, float *C) {
                                        
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    // Number of threads in one block is equal to the the number of elements in one tile of B
    // One to One mapping from thread block to B tile.
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float Asub[BM * BK];
    __shared__ float Bsub[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Number of threads = BN*BK, where as elements in A = BM*BK
    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;


    float threadResults[TM] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        Asub[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bsub[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();
        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bsub[dotIdx * BN + threadCol];

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += Asub[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }
    // write the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] +
                                                        beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}