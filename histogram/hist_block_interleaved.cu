#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_BINS 26
#define CFACTOR 4


__global__ void histogram_coarsened_interleaved(
    char* data, 
    unsigned int length, 
    unsigned int* hist) 
{
    // Initialize bins
    __shared__ unsigned int hist_s[NUM_BINS];
    
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        hist_s[bin] = 0;
    }
    __syncthreads();
    
    // Histogram
    unsigned int startIdx = blockIdx.x * blockDim.x * CFACTOR;
    
    for (int ph = 0; ph < CFACTOR; ph++) {
        unsigned int tid = startIdx + ph*blockDim.x + threadIdx.x;
        if (tid < length) {
            int pos = data[tid] - 'a';
            if (pos >= 0 && pos < NUM_BINS) {
                atomicAdd(&hist_s[pos], 1);
        }
        }
    }
    
    // Synchronize: ensure all threads finish updating private histogram
    __syncthreads();
    
    if (blockIdx.x > 0) {
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = hist_s[bin];
            if (binValue > 0) {
                atomicAdd(&hist[bin], binValue);
            }
        }
    }
}


void computeHistogram(
    char* h_data,
    unsigned int data_length,
    unsigned int* h_hist,
    unsigned int num_bins) 
{
    char* d_data = nullptr;
    unsigned int* d_hist = nullptr;
    
    // Allocate device memory
    cudaMalloc(&d_data, data_length * sizeof(char));
    cudaMalloc(&d_hist, num_bins * sizeof(unsigned int));
    
    // Copy input data to device
    cudaMemcpy(d_data, h_data, data_length * sizeof(char), cudaMemcpyHostToDevice);
    
    // Compute histogram
    // Configuration: 256 threads per block
    int blockdim = 256;
    int gridsize  = (data_length + blockdim * CFACTOR - 1) / (blockdim * CFACTOR);

    histogram_coarsened_interleaved<<<gridsize, blockdim>>>(
        d_data, 
        data_length, 
        d_hist);
    
    // Copy result back to host
    cudaMemcpy(h_hist, d_hist, num_bins * sizeof(unsigned int), 
               cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_hist);
}

// Main function demonstrating usage
int main() {
    const unsigned int DATA_SIZE = 1000000;
    
    // Allocate host memory
    char* h_data = (char*)malloc(DATA_SIZE * sizeof(char));
    unsigned int* h_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    
    // Initialize with sample data
    for (unsigned int i = 0; i < DATA_SIZE; i++) {
        h_data[i] = 'a' + (rand() % 26);
    }
    
    // Compute histogram
    computeHistogram(h_data, DATA_SIZE, h_hist, NUM_BINS);
    
    // Display results
    printf("Histogram Results:\n");
    for (unsigned int i = 0; i < NUM_BINS; i++) {
        printf("'%c': %u\n", 'a' + i, h_hist[i]);
    }
    
    // Cleanup
    free(h_data);
    free(h_hist);
    
    return 0;
}
