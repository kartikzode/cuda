#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmul_wmma_simple(
    half *A,
    half *B,
    float *C,
    int m,
    int n,
    int k
) {
    // Block and thread indices
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    int thread_row = threadIdx.x / (WMMA_N / 16);
    int thread_col = threadIdx.x % (WMMA_N / 16);
    
    // Output tile position
    int out_row = block_row * WMMA_M + thread_row * 16;
    int out_col = block_col * WMMA_N + thread_col * 16;
    
    // Shared memory for tiling
    __shared__ half smem_A[WMMA_M * WMMA_K];  // 16×16 tile of A
    __shared__ half smem_B[WMMA_K * WMMA_N];  // 16×16 tile of B
    
    // Fragment declarations (registers for storing tiles)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension in chunks of WMMA_K
    for (int k_chunk = 0; k_chunk < k; k_chunk += WMMA_K) {
        // ============================================================
        // Phase 1: Load A tile into shared memory (cooperative load)
        // ============================================================
        // Each thread loads part of the A tile
        for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += blockDim.x) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            
            int global_row = out_row + row;
            int global_col = k_chunk + col;
            
            // Bounds checking
            if (global_row < m && global_col < k) {
                smem_A[row * WMMA_K + col] = A[global_row * k + global_col];
            } else {
                smem_A[row * WMMA_K + col] = 0.0f;
            }
        }
        
        // ============================================================
        // Phase 2: Load B tile into shared memory (cooperative load)
        // ============================================================
        for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += blockDim.x) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            
            int global_row = k_chunk + row;
            int global_col = out_col + col;
            
            // Bounds checking
            if (global_row < k && global_col < n) {
                smem_B[row * WMMA_N + col] = B[global_row + global_col * k];
            } else {
                smem_B[row * WMMA_N + col] = 0.0f;
            }
        }
        
        // Synchronize to ensure all threads have loaded data
        __syncthreads();
        
        // ============================================================
        // Phase 3: Load fragments from shared memory
        // ============================================================
        // Each warp loads its tile from shared memory into registers
        load_matrix_sync(a_frag, smem_A, WMMA_K);
        load_matrix_sync(b_frag, smem_B, WMMA_N);
        
        // ============================================================
        // Phase 4: Perform matrix multiply-accumulate (THE TENSOR CORE)
        // ============================================================
        // This is where the actual tensor core operation happens
        // D = A * B + C (warp-synchronous operation)
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // Synchronize before next iteration
        __syncthreads();
    }
    
    // ============================================================
    // Phase 5: Store results from registers to global memory
    // ============================================================
    // Store the accumulated tile back to global memory
    for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        
        int global_row = out_row + row;
        int global_col = out_col + col;
        
        if (global_row < m && global_col < n) {
            C[global_row * n + global_col] = c_frag.x[i];
        }
    }
}

// ============================================================
// Host code to test the kernel
// ============================================================

void print_matrix(float *mat, int rows, int cols, const char *name) {
    printf("\n%s (first 4x4):\n", name);
    for (int i = 0; i < min(4, rows); i++) {
        for (int j = 0; j < min(4, cols); j++) {
            printf("%7.3f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Matrix dimensions
    int m = 64;
    int n = 64;
    int k = 64;
    
    printf("Matrix Multiplication using WMMA (Tensor Cores)\n");
    printf("Matrix A: %d × %d\n", m, k);
    printf("Matrix B: %d × %d\n", k, n);
    printf("Matrix C: %d × %d\n", m, n);
    printf("Using WMMA tiles of %d × %d\n\n", WMMA_M, WMMA_N);
    
    // Allocate host memory
    half *h_A = (half *)malloc(m * k * sizeof(half));
    half *h_B = (half *)malloc(k * n * sizeof(half));
    float *h_C = (float *)malloc(m * n * sizeof(float));
    float *h_C_ref = (float *)malloc(m * n * sizeof(float));
    
    // Initialize with simple values for verification
    // A[i][j] = i + j
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = __float2half(1.0f);
        }
    }
    
    // B[i][j] = 1 (column-major)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i + j * k] = __float2half(1.0f);
        }
    }
    
    // Reference computation (CPU): C[i][j] = sum of A[i][l] * B[l][j] over l
    // Since all A and B elements are 1, C[i][j] = k
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_C_ref[i * n + j] = (float)k;
        }
    }
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, m * k * sizeof(half));
    cudaMalloc(&d_B, k * n * sizeof(half));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(half), cudaMemcpyHostToDevice);
    
    // Launch kernel
    // Grid: (m/WMMA_M) × (n/WMMA_N) blocks
    // Block: 32 threads (one warp, minimum for WMMA)
    dim3 grid((m + WMMA_M - 1) / WMMA_M, (n + WMMA_N - 1) / WMMA_N);
    dim3 block(32);
    
    printf("Grid: %d × %d blocks\n", grid.x, grid.y);
    printf("Block: %d threads\n\n", block.x);
    
    // Warm up
    matmul_wmma_simple<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_wmma_simple<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance
    long long operations = (long long)m * n * k * 2;  // multiply-add
    double gflops = (operations / 1e9) / (milliseconds / 1000.0);
    
    printf("Kernel execution time (10 runs): %.3f ms\n", milliseconds);
    printf("Average time per run: %.3f ms\n", milliseconds / 10.0);
    printf("Performance: %.1f GFLOPS\n\n", gflops);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float error = fabsf(h_C[i] - h_C_ref[i]);
        if (error > max_error) max_error = error;
        if (error > 0.1f) {  // Tolerance for floating point
            correct = false;
        }
    }
    
    printf("Result verification:\n");
    if (correct) {
        printf("✓ PASSED - All results within tolerance\n");
    } else {
        printf("✗ FAILED - Results differ from reference\n");
    }
    printf("Max error: %.6f\n\n", max_error);
    
    // Print sample results
    print_matrix(h_C, m, n, "Computed C");
    print_matrix(h_C_ref, m, n, "Reference C");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
