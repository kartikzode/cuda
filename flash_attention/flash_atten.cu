#include <cuda_runtime.h>
#include <cmath>

constexpr int B_r = 1;   // number of rows to process for one q_tile
constexpr int B_c = 32;  // number of columns to process for one k_tile and v_tile

__global__
void flash_atten_kernel(const float* Q, const float*K, const float* V, 
    float* out, float scaling, int M, int N, int T_r, int T_c, int d)
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
        float* V_j = array + (B_c * d);
        float* S = V_j + 2*(B_c*d);

        //local accumulators
        float O_i = 0.0f;
        float l_i = 0.0f;
        float m_i = -1e30f;

        //load q tile
        Q_i[tx] = Q[bix*bdx + tx];

        // loop over T_c tiles of K/V
        for (int j = 0; j < T_c; j++) {
            // load one k/v tile 
            for (int jj = 0; jj < B_c; jj++) {
                // load K and V with boundary check
                if (N < jj + j*B_c) {
                    K_j[jj*d + tx] = 0.0f;
                    V_j[jj*d + tx] = 0.0f;
                } else {
                    K_j[jj*d + tx] = K[(j*B_c + jj)*d + tx];
                    V_j[jj*d + tx] = V[(j*B_c + jj)*d + tx];
                }
            }

            __syncthreads();

            // compute S
            for (int jj = tx; jj < B_c; jj += bdx) {
                float dot = 0.0f;
                if (N < jj + j*B_c) {
                    for (int k = 0; k < d; k++) {
                        dot += Q_i[k] * K_j[jj*d + k];
                    }
                }
                S[jj] = scaling * dot;
            }
            
            // get m_i
            float m = m_i;
            float last_m = m;
            for (int jj = 0; jj < B_c; jj++) {
                m = fmax(m, S[jj]);
            }
            m_i = m;   // new row max

            // rescaling the old denominator
            float l = exp(last_m - m_i) * l_i;

            // Scale O_i
            O_i *= expf(last_m - m_i);

            // Compute P_ij  (one value per thread)
            S[tx] = expf(S[tx] - m_i);  //P_ij, stored in shared meme so that every thread can access it

            // compute row sum (calculated by every thread)
            for (int jj = 0; jj < bdx; jj++) {
                l += S[jj];
            }
            
            //New O_i
            for (int jj = 0; jj < B_c; jj++) {
                O_i += S[tx] * V_j[jj*d + tx];
            }
            l_i = l;
            __syncthreads();
        }

        // write output to global memory 
        if (col < M) {
            out[bix*bdx + tx] = O_i / l_i;
        }
    }

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {

    dim3 blockSize(d);
    int gridSize = M;

    int T_r = (M + B_r - 1) / B_r;  // number of q_tiles
    int T_c = (N + B_c - 1) / B_c;   // number of K/V tiles
    //scaling factor
    float scaling = 1.0f / sqrtf(d);

    int q_tile_sz = B_r * d;
    int k_tile_sz = B_c * d;
    int v_tile_sz = B_c * d;
    int s_tile_sz = B_r * B_c;

    size_t size = (q_tile_sz + k_tile_sz + v_tile_sz + s_tile_sz) * sizeof(float);

    flash_atten_kernel<<<gridSize, blockSize, size>>>(Q, K, V, output, scaling, M, N, T_r, T_c, d);

}