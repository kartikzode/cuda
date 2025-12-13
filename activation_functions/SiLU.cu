#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }


__global__ void silu_kernel(const float* input, float* output, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}
void silu_cpu(const std::vector<float>& input, std::vector<float>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sigmoid;
    }
}

int main() {

    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;
    size_t bytes = N * sizeof(float);

    std::cout << "Testing SiLU Activation Kernel..." << std::endl;
    std::cout << "Vector size: " << N << " elements" << std::endl;

    std::vector<float> h_input(N);
    std::vector<float> h_output_gpu(N);
    std::vector<float> h_output_cpu(N);

    // Input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = -5.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10.0f));
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    silu_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    silu_cpu(h_input, h_output_cpu);

    float max_error = 0.0f;
    const float TOLERANCE = 1e-5f;
    bool match = true;

    for (int i = 0; i < N; ++i) {
        float diff = std::abs(h_output_cpu[i] - h_output_gpu[i]);
        if (diff > max_error) max_error = diff;
        
        if (diff > TOLERANCE) {
            if (match) { // Print details for the first mismatch only
                std::cerr << "Mismatch at index " << i << ": "
                          << "CPU=" << h_output_cpu[i] << ", "
                          << "GPU=" << h_output_gpu[i] << ", "
                          << "Diff=" << diff << std::endl;
            }
            match = false;
        }
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    if (match) {
        std::cout << "PASSED" << std::endl;
        std::cout << "Max Error: " << max_error << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
