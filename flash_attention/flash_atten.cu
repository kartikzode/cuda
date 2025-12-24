#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__
void flash_atten_kernel(const float* Q, const float*K, const float* V, 
    float* out, float scaling, int M, int N, int T_r, int T_c, int B_r, int B_c, int d)
    {
        int tx = threadIdx.x;
        int bix = blockIdx.x;
        int bdx = blockDim.x;
        int col = bix*bdx + tx;

        // Check bounds - each block processes one row of Q
        if (bix >= M) return;
        if (tx >= d) return;


        // Shared memory buffers for Q, K, V blocks
        extern __shared__ float array[];

        float* Q_i = array;
        float* K_j = array + d;
        float* V_j = K_j + (B_c * d);
        float* S = V_j + (B_c*d);

        //local accumulators
        float O_i = 0.0f;
        float l_i = 0.0f;
        float m_i = -1e30f;

        //load q tile
        Q_i[tx] = Q[bix*bdx + tx];

        __syncthreads();
        // Debug: print loaded Q values
        if (bix == 0 && tx == 0) {
            printf("Loaded Q values:\n");
            for (int k = 0; k < d; k++) {
                printf("Q[%d]=%f\n", k, Q_i[k]);
            }
        }
        __syncthreads();

        // loop over T_c tiles of K/V
        for (int j = 0; j < T_c; j++) {
            printf("Processing tile %d by block %d\n", j, bix);
            // load one k/v tile 
            for (int jj = 0; jj < B_c; jj++) {
                printf("Block %d, Thread %d loading K and V for jj=%d\n", bix, tx, jj);
                // load K and V with boundary check
                K_j[jj*d + tx] = K[(j*B_c + jj)*d + tx];
                V_j[jj*d + tx] = V[(j*B_c + jj)*d + tx];
            }

            __syncthreads();

            // Debug: print loaded K and V values
            if (bix == 0 && tx == 0) {
                printf("Loaded K values for tile %d:\n", j);
                for (int jj = 0; jj < B_c; jj++) {
                    printf("K[%d]: ", jj);
                    for (int k = 0; k < d; k++) {
                        printf("%f ", K_j[jj*d + k]);
                    }
                    printf("\n");
                }
                printf("Loaded V values for tile %d:\n", j);
                for (int jj = 0; jj < B_c; jj++) {
                    printf("V[%d]: ", jj);
                    for (int k = 0; k < d; k++) {
                        printf("%f ", V_j[jj*d + k]);
                    }
                    printf("\n");
                }
            }
            __syncthreads();

            // compute S
            for (int jj = tx; jj < B_c; jj += bdx) {
                float dot = 0.0f;
                for (int k = 0; k < d; k++) {
                    dot += Q_i[k] * K_j[jj*d + k];
                }
                S[jj] = scaling * dot;
            }
            __syncthreads();

            // // print the s values for debugging
            // if (bix == 0 && tx == 0) {
            //     printf("Tile %d S values:\n", j);
            //     for (int jj = 0; jj < B_c; jj++) {
            //         printf("S[%d]=%f\n", jj, S[jj]);
            //     }
            // }
            // __syncthreads();
            
            // get m_i
            float m = m_i;
            float last_m = m;
            for (int jj = 0; jj < B_c; jj++) {
                m = fmax(m, S[jj]);
            }
            // new row max
            m_i = m;
            // rescaling the old denominator
            float l = exp(last_m - m_i) * l_i;
            // Scale O_i
            O_i *= expf(last_m - m_i);
            // Compute P_ij
            for (int jj = 0; jj < B_c; jj++) {
                float P_ij = expf(S[jj] - m_i);
                l += P_ij;
                O_i += P_ij * V_j[jj*d + tx];

            }
            l_i = l;
            __syncthreads();
        }

        // write output to global memory 
        if (bix < M && tx < d) {
            out[bix * d + tx] = O_i / l_i;
        }
    }

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {

    dim3 blockSize(d);
    int gridSize = M;

    int B_r = 1;   // number of rows to process for one q_tile
    int B_c = 32;  // number of columns to process for one k_tile and v_tile

    int T_r = (M + B_r - 1) / B_r;  // number of q_tiles
    if (N < B_c) {
        // Adjust k/v tile size if N is smaller than B_c
        B_c = N;
    } 
    int T_c = (N + B_c - 1) / B_c;   // number of K/V tiles
    //scaling factor
    float scaling = 1.0f / sqrtf(d);

    int q_tile_sz = B_r * d;
    int k_tile_sz = B_c * d;
    int v_tile_sz = B_c * d;
    int s_tile_sz = B_r * B_c;

    size_t size = (q_tile_sz + k_tile_sz + v_tile_sz + s_tile_sz) * sizeof(float);

    flash_atten_kernel<<<gridSize, blockSize, size>>>(Q, K, V, output, scaling, M, N, T_r, T_c, B_r, B_c, d);

}