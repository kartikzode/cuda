#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


// Each thread computes one element of the output
__global__ void naiveSoftmaxKernel(float *input, float *output, int N, int D) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * D) {
        int row = idx / D;
        int col = idx % D;
        
        // Compute sum of exponentials for this row
        float sum_exp = 0.0f;
        for (int i = 0; i < D; i++) {
            sum_exp += expf(input[row * D + i]);
        }
        
        // Compute softmax: exp(x) / sum(exp(x))
        output[idx] = expf(input[idx]) / sum_exp;
    }
}


void computeSoftmax(float *h_input, float *h_output, int N, int D) {
    float *d_input, *d_output;
    size_t bytes = N * D * sizeof(float);
    
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int numBlocks = (N * D + threadsPerBlock - 1) / threadsPerBlock;
    
    naiveSoftmaxKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, D);
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int N = 4;
    int D = 8;
    
    float *h_input = (float*)malloc(N * D * sizeof(float));
    float *h_output = (float*)malloc(N * D * sizeof(float));
    
    for (int i = 0; i < N * D; i++) {
        h_input[i] = (float)(i % 10) - 5.0f;
    }
    
    printf("Input matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            printf("%7.3f ", h_input[i * D + j]);
        }
        printf("\n");
    }
    
    computeSoftmax(h_input, h_output, N, D);
    
    printf("\nSoftmax output:\n");
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < D; j++) {
            printf("%7.3f ", h_output[i * D + j]);
            row_sum += h_output[i * D + j];
        }
        printf(" (sum: %.6f)\n", row_sum);
    }
    
    free(h_input);
    free(h_output);
    return 0;
}
