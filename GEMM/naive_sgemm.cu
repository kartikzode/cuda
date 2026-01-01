#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }

    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launch_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  dim3 blockSize(32, 32);
  dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                (N + blockSize.y - 1) / blockSize.y);

  sgemm_naive<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
}

// initialize the matrices with random values
    void init_matrix(float* mat, int rows, int cols) {
        for (int i = 0; i < rows * cols; ++i) {
            mat[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    init_matrix(h_C, M, N);

    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    launch_sgemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}