#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// GEMM Kernel with 1D block tiling
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_1d_blockTiling(int M, int N, int K, float alpha,
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

    // loading 4 values (128 bits/16bytes) per instruction using vectorized memory operation
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);


    float threadResults[TM * TN] = {0.0f};
    float regB[TN] = {0.0f};
    float regA[TM] = {0.0f};

    // outer most loop over the block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // load a_tile into smem as a transpose 
        float4 tmp = reinterpret_cast<float4*> (&A[innerRowA*K + innerColA*4])[0];
        Asub[(innerColA*4 + 0)*BM + innerRowA] = tmp.x;
        Asub[(innerColA*4 + 1)*BM + innerRowA] = tmp.y;
        Asub[(innerColA*4 + 2)*BM + innerRowA] = tmp.z;
        Asub[(innerColA*4 + 3)*BM + innerRowA] = tmp.w;

        //load b_tile into smem
        reinterpret_cast<float4*>(&Bsub[innerRowB*BN + innerColB*4])[0] = 
                    reinterpret_cast<float4*>(&B[innerRowB*N + innerColB*4])[0]; 
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
                regA[resIdxA] = Asub[dotIdx*BM + threadRow*TM + resIdxA];
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
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB])[0];
            // perform GEMM update in reg
            tmp.x = alpha * threadResults[resIdxA * TN + resIdxB] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxA * TN + resIdxB + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxA * TN + resIdxB + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxA * TN + resIdxB + 3] + beta * tmp.w;
            // write back
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB])[0] =
                tmp;
        }
        
    }
}