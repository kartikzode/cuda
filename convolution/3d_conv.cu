#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define FILTER_RADIUS 1
#define FILTER_SIZE (2*FILTER_RADIUS + 1)
#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2*FILTER_RADIUS)

__constant__ float d_filter[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];

// 3D Convolution kernel
__global__ void convolution_3d_tiled_kernel(float *input, float *output,
                                             int depth, int height, int width) {

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    int out_d = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
    int out_h = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int out_w = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    
    // Load input tile into shared memory
    if (out_d >= 0 && out_d < depth && 
        out_h >= 0 && out_h < height && 
        out_w >= 0 && out_w < width) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = 
            input[out_d * height * width + out_h * width + out_w];
    } else {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    int tile_d = threadIdx.z - FILTER_RADIUS;
    int tile_h = threadIdx.y - FILTER_RADIUS;
    int tile_w = threadIdx.x - FILTER_RADIUS;
    
    if (tile_d >= 0 && tile_d < OUT_TILE_DIM &&
        tile_h >= 0 && tile_h < OUT_TILE_DIM &&
        tile_w >= 0 && tile_w < OUT_TILE_DIM) {
        
        int out_d_final = blockIdx.z * OUT_TILE_DIM + tile_d;
        int out_h_final = blockIdx.y * OUT_TILE_DIM + tile_h;
        int out_w_final = blockIdx.x * OUT_TILE_DIM + tile_w;
        
        int out_depth = depth - 2*FILTER_RADIUS;
        int out_height = height - 2*FILTER_RADIUS;
        int out_width = width - 2*FILTER_RADIUS;
        
        if (out_d_final < out_depth && out_h_final < out_height && out_w_final < out_width) {
            float result = 0.0f;
            
            for (int fd = 0; fd < FILTER_SIZE; fd++) {
                for (int fh = 0; fh < FILTER_SIZE; fh++) {
                    for (int fw = 0; fw < FILTER_SIZE; fw++) {
                        result += d_filter[fd][fh][fw] * 
                                 tile[tile_d + fd][tile_h + fh][tile_w + fw];
                    }
                }
            }
            
            // Write output
            int output_idx = out_d_final * out_height * out_width + 
                           out_h_final * out_width + out_w_final;
            output[output_idx] = result;
        }
    }
}

int main() {
    
    int depth = 16;
    int height = 16;
    int width = 16;
    
    int out_depth = depth - 2*FILTER_RADIUS;
    int out_height = height - 2*FILTER_RADIUS;
    int out_width = width - 2*FILTER_RADIUS;
    
    printf("Input volume: %d x %d x %d\n", depth, height, width);
    printf("Filter size: %d x %d x %d (FILTER_RADIUS = %d)\n", 
           FILTER_SIZE, FILTER_SIZE, FILTER_SIZE, FILTER_RADIUS);
    printf("Output volume: %d x %d x %d\n\n", out_depth, out_height, out_width);
    
    // Allocate host memory
    size_t input_size = depth * height * width * sizeof(float);
    size_t output_size = out_depth * out_height * out_width * sizeof(float);
    size_t filter_size = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE * sizeof(float);
    
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(output_size);
    float *h_filter = (float*)malloc(filter_size);
    
    // Initialize host input
    printf("Initializing 3D input volume...\n");
    for (int i = 0; i < depth * height * width; i++) {
        h_input[i] = sinf(i * 0.01f) + 2.0f;
    }
    
    // Initialize host filter (Gaussian-like kernel)
    printf("Initializing 3D filter...\n");
    float filter_data[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];
    for (int d = 0; d < FILTER_SIZE; d++) {
        for (int h = 0; h < FILTER_SIZE; h++) {
            for (int w = 0; w < FILTER_SIZE; w++) {
                // Simple Gaussian-like kernel
                float dist = sqrtf((d-1)*(d-1) + (h-1)*(h-1) + (w-1)*(w-1));
                filter_data[d][h][w] = expf(-dist*dist / 2.0f) / 9.0f;
            }
        }
    }
    memcpy(h_filter, filter_data, filter_size);
    
    // Print sample of filter
    printf("Sample filter slice [middle plane]:\n");
    for (int h = 0; h < FILTER_SIZE; h++) {
        for (int w = 0; w < FILTER_SIZE; w++) {
            printf("%.4f ", filter_data[FILTER_RADIUS][h][w]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, filter_size));
    
    printf("Data copied to device (filter in constant memory)\n\n");
    
    // Launch tiled kernel
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((out_width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (out_height + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (out_depth + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    printf("Launching 3D tiled kernel with:\n");
    printf("  Block size: %d x %d x %d = %d threads\n", 
           threads.x, threads.y, threads.z, threads.x * threads.y * threads.z);
    printf("  Grid size: %d x %d x %d = %d blocks\n\n", 
           blocks.x, blocks.y, blocks.z, blocks.x * blocks.y * blocks.z);
    
    convolution_3d_tiled_kernel<<<blocks, threads>>>(d_input, d_output, depth, height, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Kernel execution completed!\n\n");
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Print sample results
    printf("Sample input values at [0][0][0:5]:\n");
    for (int w = 0; w < 5 && w < width; w++) {
        printf("%.4f ", h_input[0 * height * width + 0 * width + w]);
    }
    printf("\n\n");
    
    printf("Sample output values at [0][0][0:5]:\n");
    for (int w = 0; w < 5 && w < out_width; w++) {
        printf("%.4f ", h_output[0 * out_height * out_width + 0 * out_width + w]);
    }
    printf("\n\n");
    
    // Manual verification of output[0][0][0]
    printf("Verification of output[0][0][0]:\n");
    printf("(This uses input region [1:4, 1:4, 1:4] for filter application)\n\n");
    
    float expected = 0.0f;
    for (int fd = 0; fd < FILTER_SIZE; fd++) {
        for (int fh = 0; fh < FILTER_SIZE; fh++) {
            for (int fw = 0; fw < FILTER_SIZE; fw++) {
                int in_d = FILTER_RADIUS + fd;
                int in_h = FILTER_RADIUS + fh;
                int in_w = FILTER_RADIUS + fw;
                int input_idx = in_d * height * width + in_h * width + in_w;
                expected += h_filter[fd * FILTER_SIZE * FILTER_SIZE + fh * FILTER_SIZE + fw] * h_input[input_idx];
            }
        }
    }
    
    printf("  Expected: %.6f\n", expected);
    printf("  Got: %.6f\n", h_output[0]);
    printf("  Match: %s\n\n", fabsf(expected - h_output[0]) < 0.001f ? "YES ✓" : "NO ✗");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_filter);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("Test completed successfully!\n");
    return 0;
}
