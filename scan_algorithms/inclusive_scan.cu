#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// Koge stone segmented scan implementation
__global__ void segmentedScanKernel(const int* input, int* output, int n) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        shared[tid] = input[gid];
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            int val = 0;
            if (tid >= offset) {
                val = shared[tid - offset];
            }
            __syncthreads();
            shared[tid] += val;
            __syncthreads();
        }

        output[gid] = shared[tid];
    }
}

// Host function to launch the segmented scan kernel
void segmentedScan(const std::vector<int>& input, std::vector<int>& output) {
    int *d_input, *d_output;
    int n = input.size();
    size_t size = n * sizeof(int);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    segmentedScanKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);
    cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output(input.size());

    segmentedScan(input, output);

    std::cout << "Segmented Scan Output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
