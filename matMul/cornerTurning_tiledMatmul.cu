#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_WIDTH 32
#define WIDTH 512

// CUDA kernel
__global__
void matmul_tile(float* A, float* B, float* C, int Width) {
    
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Get the row and column value of the C matrix to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Loop over the A and B tiles required to compute C elements
    float value = 0.0;
    for(int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {
        
        // Collaborative loading of A and B tiles into Shared Memory
        Ads[ty][tx] = (((row < Width) && ((ph*TILE_WIDTH + tx) < Width)) ? 
                       A[row*Width + ph*TILE_WIDTH + tx] : 0.f);
        
        // B is in column-major order
        Bds[ty][tx] = ((((ph*TILE_WIDTH + ty) < Width) && (col < Width)) ? 
                       B[col*Width + (ph*TILE_WIDTH + ty)] : 0.f);
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            value += Ads[ty][k] * Bds[tx][k];
        }
        __syncthreads();
    }
    
    if(row < Width && col < Width) {
        C[row*Width + col] = value;
    }
}

// // Host function to initialize matrix (row-major)
// void initMatrix(float* matrix, int rows, int cols, int seed) {
//     srand(seed);
//     for(int i = 0; i < rows * cols; i++) {
//         matrix[i] = (float)(rand() % 100) / 10.0f;
//     }
// }

// // Host function to initialize matrix (column-major)
// void initMatrixColMajor(float* matrix, int rows, int cols, int seed) {
//     srand(seed);
//     for(int c = 0; c < cols; c++) {
//         for(int r = 0; r < rows; r++) {
//             matrix[c * rows + r] = (float)(rand() % 100) / 10.0f;
//         }
//     }
// }

// // CPU reference implementation (A row-major, B column-major)
// void matmulCPU(float* A, float* B, float* C, int Width) {
//     for(int i = 0; i < Width; i++) {
//         for(int j = 0; j < Width; j++) {
//             float sum = 0.0f;
//             for(int k = 0; k < Width; k++) {
//                 // A is row-major: A[i][k] = A[i*Width + k]
//                 // B is column-major: B[j][k] = B[j*Width + k]
//                 sum += A[i*Width + k] * B[j*Width + k];
//             }
//             C[i*Width + j] = sum;
//         }
//     }
// }

// // Verify results
// bool verifyResults(float* GPU_result, float* CPU_result, int size, float tolerance = 1e-4) {
//     for(int i = 0; i < size; i++) {
//         float diff = fabs(GPU_result[i] - CPU_result[i]);
//         if(diff > tolerance) {
//             printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n", i, GPU_result[i], CPU_result[i], diff);
//             return false;
//         }
//     }
//     return true;
// }

// int main() {
//     int Width = WIDTH;
//     int size = Width * Width;
    
//     // Host memory allocation
//     float* h_A = (float*)malloc(size * sizeof(float));
//     float* h_B = (float*)malloc(size * sizeof(float));
//     float* h_C_GPU = (float*)malloc(size * sizeof(float));
//     float* h_C_CPU = (float*)malloc(size * sizeof(float));
    
//     if(!h_A || !h_B || !h_C_GPU || !h_C_CPU) {
//         printf("Host memory allocation failed\n");
//         return 1;
//     }

//     // Initialize matrices
//     printf("Initializing matrices...\n");
//     initMatrix(h_A, Width, Width, 1);          // A is row-major
//     initMatrixColMajor(h_B, Width, Width, 2);  // B is column-major
//     memset(h_C_GPU, 0, size * sizeof(float));
//     memset(h_C_CPU, 0, size * sizeof(float));

//     // Compute reference result on CPU
//     printf("Computing CPU result...\n");
//     matmulCPU(h_A, h_B, h_C_CPU, Width);

//     // Device memory allocation
//     float* d_A, * d_B, * d_C;
//     cudaMalloc((void**)&d_A, size * sizeof(float));
//     cudaMalloc((void**)&d_B, size * sizeof(float));
//     cudaMalloc((void**)&d_C, size * sizeof(float));

//     // Copy data to device
//     printf("Copying data to device...\n");
//     cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemset(d_C, 0, size * sizeof(float));

//     // Configure grid and blocks
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
//     dim3 gridDim((Width + TILE_WIDTH - 1) / TILE_WIDTH, 
//                  (Width + TILE_WIDTH - 1) / TILE_WIDTH);

//     printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

//     // Launch kernel
//     printf("Launching kernel...\n");
//     matmul_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, Width);
    
//     // Check for kernel launch errors
//     cudaError_t err = cudaGetLastError();
//     if(err != cudaSuccess) {
//         printf("Kernel launch error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // Wait for kernel to finish
//     cudaDeviceSynchronize();

//     // Copy result back to host
//     printf("Copying result back to host...\n");
//     cudaMemcpy(h_C_GPU, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

//     // Verify results
//     printf("Verifying results...\n");
//     if(verifyResults(h_C_GPU, h_C_CPU, size)) {
//         printf("✓ PASS: GPU and CPU results match!\n");
//     } else {
//         printf("✗ FAIL: GPU and CPU results do not match!\n");
//     }

//     // Print sample results
//     printf("\nSample results (first 5x5 block of C matrix):\n");
//     printf("GPU Results:\n");
//     for(int i = 0; i < 5 && i < Width; i++) {
//         for(int j = 0; j < 5 && j < Width; j++) {
//             printf("%8.2f ", h_C_GPU[i*Width + j]);
//         }
//         printf("\n");
//     }
    
//     printf("\nCPU Results:\n");
//     for(int i = 0; i < 5 && i < Width; i++) {
//         for(int j = 0; j < 5 && j < Width; j++) {
//             printf("%8.2f ", h_C_CPU[i*Width + j]);
//         }
//         printf("\n");
//     }

//     // Cleanup
//     free(h_A);
//     free(h_B);
//     free(h_C_GPU);
//     free(h_C_CPU);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     printf("\nTest completed!\n");
//     return 0;
// }
