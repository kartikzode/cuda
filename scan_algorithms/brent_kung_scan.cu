#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define SECTION_SIZE 10

//helper function to check for CUDA errors
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Brent Kung Scan implementation
__global__ void brentKungScanKernel(const float* input, float* output, float n) {
    __shared__ float shared[SECTION_SIZE];
    
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = segment + tid;
    

    if (i < n) {shared[tid] = input[i];}
    if (i + blockDim.x < n) {shared[tid + blockDim.x] = input[i + blockDim.x];}

    // reduction tree
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int indx = (tid + 1)*2*stride - 1;
        if (indx < SECTION_SIZE) {
            shared[indx] += shared[indx - stride];
        }
    }

    // reverse tree
    for (unsigned int stride = (2*blockDim.x)/4; stride > 0; stride >>= 1) {
        __syncthreads();
        unsigned int indx = (tid + 1)*2*stride - 1;
        if (indx < SECTION_SIZE) {
            shared[indx + stride] += shared[indx]; 
        }
    }
    __syncthreads(); 
    if (i < n) output[i] = shared[threadIdx.x];       
    if (i + blockDim.x < n) output[i + blockDim.x] = shared[threadIdx.x + blockDim.x];
}

void brentKungScan(const std::vector<float>& input, std::vector<float>& output) {
    float *d_input, *d_output;
    size_t size = input.size() * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = SECTION_SIZE / 2;
    int blocksPerGrid = (input.size() + SECTION_SIZE - 1) / SECTION_SIZE;

    brentKungScanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, input.size());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int dataSize = 1 << 20; // Example size
    std::vector<float> input(dataSize, 2.0f); // Initialize input with 1.0f
    std::vector<float> output(dataSize, 0.0f);

    brentKungScan(input, output);

    std::cout << "Output: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
