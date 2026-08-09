// Async (TMA) tiled matrix multiplication.
//
// Uses cuda::barrier + cp.async.bulk (TMA) to overlap global->shared memory
// copies of the next A/B tile with compute on the current tile
// (double-buffered software pipeline). Requires Hopper (sm_90a) since
// cp.async.bulk is a Hopper-only instruction.
//
// Build: nvcc -arch=sm_90a async_matmul.cu -o async_matmul

#include <cuda/barrier>
#include <cuda/ptx>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#define M 512
#define K 512
#define N 512
#define TILE 32          // square tile size (TILE x TILE)
#define NUM_STAGES 2      // double buffering

__device__ inline bool is_elected() {
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
}

// Copies a TILE x TILE sub-block of `src` (row-major, `ld` columns wide)
// starting at (row0, col0) into `dst` (contiguous TILE x TILE), via TMA.
__device__ inline void tma_load_tile(
    float* dst, const float* src, int ld, int row0, int col0, barrier& bar) {
    if (is_elected()) {
        for (int r = 0; r < TILE; r++) {
            ptx::cp_async_bulk(
                ptx::space_shared, ptx::space_global,
                dst + r * TILE, src + (row0 + r) * ld + col0,
                TILE * sizeof(float));
        }
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_release, ptx::scope_cta, ptx::space_shared,
            cuda::device::barrier_native_handle(bar), TILE * TILE * sizeof(float));
    }
}

__global__ void async_matmul_kernel(const float* A, const float* B, float* C,
                                     int m, int k, int n) {
    __shared__ alignas(16) float As[NUM_STAGES][TILE][TILE];
    __shared__ alignas(16) float Bs[NUM_STAGES][TILE][TILE];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[NUM_STAGES];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int s = 0; s < NUM_STAGES; s++) init(&bar[s], blockDim.x * blockDim.y);
    }
    __syncthreads();

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;
    int num_tiles = k / TILE;

    float acc = 0.0f;

    // Prime the pipeline: kick off the load for tile 0 into stage 0.
    int cur_stage = 0;
    tma_load_tile(&As[cur_stage][0][0], A, k, block_row, 0, bar[cur_stage]);
    tma_load_tile(&Bs[cur_stage][0][0], B, n, 0, block_col, bar[cur_stage]);
    barrier::arrival_token token = bar[cur_stage].arrive();

    for (int t = 0; t < num_tiles; t++) {
        int stage = t % NUM_STAGES;
        int next_stage = (t + 1) % NUM_STAGES;

        // Wait for the tile we're about to compute on.
        bar[stage].wait(std::move(token));

        // Kick off the next tile's load before computing this one, so the
        // copy overlaps with compute.
        if (t + 1 < num_tiles) {
            tma_load_tile(&As[next_stage][0][0], A, k, block_row, (t + 1) * TILE, bar[next_stage]);
            tma_load_tile(&Bs[next_stage][0][0], B, n, (t + 1) * TILE, block_col, bar[next_stage]);
            token = bar[next_stage].arrive();
        }

        __syncwarp();
        for (int l = 0; l < TILE; l++) {
            acc += As[stage][threadIdx.y][l] * Bs[stage][l][threadIdx.x];
        }
        __syncthreads();
    }

    C[(block_row + threadIdx.y) * n + (block_col + threadIdx.x)] = acc;
}

void matmul_cpu(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) mat[i] = (float)rand() / RAND_MAX;
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    h_C_ref = (float*)malloc(size_C);

    srand((unsigned)time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim(N / TILE, M / TILE);

    async_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("Computing CPU reference...\n");
    matmul_cpu(h_A, h_B, h_C_ref, M, K, N);

    double max_err = 0.0;
    for (int i = 0; i < M * N; i++) {
        double err_i = fabs(h_C[i] - h_C_ref[i]);
        if (err_i > max_err) max_err = err_i;
    }
    printf("Max error vs CPU reference: %e\n", max_err);

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
