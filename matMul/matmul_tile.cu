#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 1024  // number of rows in A & C
#define K 1000  // number of columns in A and number of rows in B
#define N 1024 // number of columns in B & C
#define TILE_WIDTH 64

// CPU matrix multiplication
void matmul_cpu(float* A, float* B, float* C, int m, int k, int n) {
    for (int i=0 ; i < m; i++) {
        for (int j=0; j < n ; j++) {
            float sum = 0.0f;
            for (int l=0; l < k; l++) {
                sum += A[i*k + l] * B[l*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

//CUDA kernel for matric multiplication
__global__
void matmul_tile( float* A, float * B, float* C, int Width) {
    
    __shared__ float Ads [TILE_WIDTH] [TILE_WIDTH];
    __shared__ float Bds [TILE_WIDTH] [TILE_WIDTH];

    int bx = blockIdx.x ; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Get the row and column value of the C matrix to work on
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    // Loop over the A and B tiles required to compute C elements
    float value = 0.0;
    for(int ph = 0; ph < ceil(Width/(float) TILE_WIDTH); ++ph) {
        
        // Collaborative loading of A and B tiles into Shares Memory
        Ads[ty][tx] =  (((row < Width) && ((ph*TILE_WIDTH + tx) < Width)) ? A[row*Width + ph*TILE_WIDTH + tx] : 0.f);
        Bds[ty][tx] = ((((ph*TILE_WIDTH + ty) < Width) && (col < Width)) ? B[(ph*TILE_WIDTH + ty)*Width + col] : 0.f);
        __syncthreads();

        for(int k=0; k < TILE_WIDTH; k++) {
            value += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if (row < Width && col < Width) {
        C[row*Width + col] = value;
    }
    
}

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    cudaMalloc((void**) &d_A, size_A);
    cudaMalloc((void**) &d_B, size_B);
    cudaMalloc((void**) &d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        // matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M);
        cudaDeviceSynchronize();
    }

    // // Benchmark CPU implementation
    // printf("Benchmarking CPU implementation...\n");
    // double cpu_total_time = 0.0;
    // for (int i = 0; i < 20; i++) {
    //     double start_time = get_time();
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    //     double end_time = get_time();
    //     cpu_total_time += end_time - start_time;
    // }
    // double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    // printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    // printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}