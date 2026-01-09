#include <stdio.h>
#include <cuda_runtime.h>

// GEMM Kernel with warp tiling
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
                                        
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Pointers to A_tile and B_tile
    A += cRow * BM * K;
    B += cCol * BN;
    // Pointer to the warptile inside C_tile
    c += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // index init for As/Bs inside smem
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowA = threadIdx.x / (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    
    // warp location in the c_tile
    const uint warpIdx = threadIdx.x / warpSize;                                // !/!
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // warp subtile size
    constexpr uint WMITER = (WM * WN) / (TM * TN * warpSize * WNITER);
    const uint WSUBM = WM / WMITER;
    const uint WSUBN = WN / WNITER;

    // thread location in the warp subtile
    const uint threadInWarp = threadIdx.x % warpSize;                           // !%!
    const uint threadRowInWarp = threadInWarp / (WSUBN / TN);
    const uint threadColInWarp = threadInWarp % (WSUBN / TN);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER*TM * WNITER*TN] = {0.0};
    
    // allocate cache for a_frag and b_frag
    float regM[WMITER*TM] = {0.0};
    float regN[WNITER*TN] = {0.0};


    // outer most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // load a_tile from gmem to smem
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {

            const float4 tmp = reinterpret_cast<const float4*>(&A[(innerRowA + offset) * K + innerColA*4])[0];
            As[(innercolA*4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innercolA*4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innercolA*4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innercolA*4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        __syncthread();
        // load b_tile from gmem into smem
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {

            reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * BN + innerColB*4])[0] = 
                    reinterpret_cast<const float4*>(&B[(innerRowB + offset) * N + innerColB*4])[0];
        }
        
        // moving the pointers to the next tiles
        A += BK;
        B += BK*N;
        
        //synchronizing the threads
        __syncthreads();

        for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {

            // load a_frag from As(smem) into regM(thread-local registerfile)
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (uint i = 0; i < TM; i++) {
                    regM[wSubRowIdx*TM + i] = As[dotIdx*BM + warpRow*WM + wSubRowIdx*WSUBM + threadRowInWarp*TM + i];
                }
            }

            // load b_frag from Bs into regN
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                for (uint i = 0; i < TN; i++) {
                    regN[wSubColIdx*TN + i] = Bs[dotIdx*BN + warpCol*WN + wSubColIdx*WSUBN + threadColInWarp*TN + i];
                }
            }

            //execute the warptile matmul
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; WNITER++) {
                    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
                        for (uint resIdxN = 0; resIdxN < TN; resIdxN++) {
                            threadResults[(wSubRowIdx*TM + resIdxM) * (WNITER*TN) + wSubColIdx*TN + resIdxN] = 
                                regM[wSubRowIdx*TM + resIdxM] * regN[wSubColIdx*TN + resIdxN];
                        }
                    }
                }
            }

            // writing the results 
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    
                    // pointer to current warp subtile
                    float* C_subWarp = C + (wSubRowIdx*WSUBM)*N + wSubColIdx*WSUBN;
                    for(uint resIdxM = 0; resIdxM < TM; resIdxM++) {
                        for(uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                            // load previous iteration's C outputs
                            float4 temp = reinterpret_cast<float4 *>( 
                                &C_subWarp[(threadRowInWarp*TM + resIdxM) * N 
                                    + threadColInWarp*TN + resIdxN])[0];
                            
                            // GEMM Update
                            const int i = (wSubRowIdx*TM + resIdxM) * (WNITER*TN) + wSubColIdx*TN + resIdxN;
                            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;

                            // writing bacl to gmem
                            reinterpret_cast<float4*>(
                                &C_subWarp[(threadRowInWarp*TM + resIdxM) * N
                                    + threadColInWarp*TN + resIdxN])[0] = tmp;
                        }
                    }
                }
            }

        }
    }
}
    