#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// GEMM Kernel with 2D block tiling
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blockTiling(int M, int N, int K, float alpha,
        const float *A, const float *B, float beta, float *C) {
                                        
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    // C_tile size (BM*BN)
    // blockdim (BM*BN / (TM*TN))
    // Number of threads to span the columns (BN/TN)
    const uint threadCol = threadIdx.x % (BN/TN);
    const uint threadRow = threadIdx.x / (BN/TN);
    const uint numThreads = (BM*BN) / (TM*TN);

    __shared__ float Asub[BM * BK];
    __shared__ float Bsub[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    // since we have numThreads (64) for loading A_tile (BM*BK) into Asub and B_tile (BK*BN) into Bsub
    // it will be done inside a for loop that will have a stride equal to the number of rows 
    // spanned in each iteration, 8 in case of A_tile and 1 in case of B_tile.
    const uint strideA = numThreads / BK;
    const uint strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regB[TN] = {0.0f};
    float regA[TM] = {0.0f};

    // outer most loop over the block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        // loading a_tile into smem
        for (uint offset = 0; offset < BM; offset += strideA) {
            Asub[(innerRowA + offset) * BK + innerColA] = A[(innerRowA + offset) * K + innerColA];
        }
        
        // loading b_tile into smem
        for (uint offset = 0; offset < BK; offset += strideB) {
            Bsub[(innerRowB + offset) * BN + innerColB] = B[(innerRowB + offset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // load the relevant values into register memory and compute the dot product
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // loading Bs
            for (uint resIdxB = 0; resIdxB < TN; ++resIdxB) {
                regB[resIdxB] = Bsub[dotIdx*BN + threadCol*TN  + resIdxB];
            }

            // loading As
            for (uint resIdxA = 0; resIdxA < TM; ++ resIdxA) {
                regA[resIdxA] = Asub[(threadRow*TM + resIdxA)*BK + dotIdx];
            }
            
            // computing the dot product
            for (uint resIdxA = 0; resIdxA < TM; ++resIdxA) {
                for (uint resIdxB = 0; resIdxB < TN; ++resIdxB) {
                    threadResults[resIdxA*TN + resIdxB] += regB[resIdxB] * regA[resIdxA];
                }
            }

        }
        __syncthreads();
    }
    // write the results
    for (uint resIdxA = 0; resIdxA < TM; ++resIdxA) {
        for (uint resIdxB = 0; resIdxB < TN; ++resIdxB) {
            C[(threadRow*TM + resIdxA)*N + threadCol*TN + resIdxB] = 
            alpha * threadResults[resIdxA*TN + resIdxB] +
            beta * C[(threadRow*TM + resIdxA)*N + threadCol*TN + resIdxB];
        }
        
    }
}