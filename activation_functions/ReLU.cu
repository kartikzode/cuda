#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void relu_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // Host memory
    std::vector<float> h_x(N), h_y(N);

    // Initialize input with a mix of positive/negative values
    for (int i = 0; i < N; ++i) {
        h_x[i] = (i - 8);  // values from -8 to 7
    }

    // Device pointers
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // Copy input to device
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);

    // Configure and launch kernel
    int threads = 32;
    int blocks  = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_x, d_y, N);

    // Wait for kernel to finish and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Copy results back
    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    // Print a few values
    std::cout << "x:\n";
    for (int i = 0; i < 10; ++i) std::cout << h_x[i] << " ";
    std::cout << "\n\ny = ReLU(x):\n";
    for (int i = 0; i < 10; ++i) std::cout << h_y[i] << " ";
    std::cout << "\n";

    // Simple correctness check
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expect = h_x[i] > 0.0f ? h_x[i] : 0.0f;
        if (h_y[i] != expect) {
            std::cerr << "Mismatch at " << i
                      << ": got " << h_y[i]
                      << ", expected " << expect << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "Test PASSED\n" : "Test FAILED\n");

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    return ok ? 0 : 1;
}
