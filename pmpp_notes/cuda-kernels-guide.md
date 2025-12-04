# CUDA Kernels with PyTorch: Complete Guide

## Table of Contents
1. [PyTorch load_inline Overview](#pytorch-loadinline-overview)
2. [Understanding the Wrapper Function](#understanding-the-wrapper-function)
3. [CUDA Kernel Fundamentals](#cuda-kernel-fundamentals)
4. [Memory Layout and Data Access](#memory-layout-and-data-access)
5. [Complete Examples](#complete-examples)

---

## PyTorch load_inline Overview

### What is load_inline?

**`load_inline`** is a just-in-time (JIT) compilation utility in PyTorch that allows you to write C++/CUDA code as strings directly in Python. It automatically compiles and binds them into a Python module without needing separate build systems.

### The Three Components

When using `load_inline`, you define three key parts:

1. **CUDA Kernel** (`cuda_sources`): The actual GPU computation code with `__global__` functions
2. **C++ Wrapper** (`cpp_sources`): Regular C++ functions that call the CUDA kernel
3. **Python Binding** (`functions`): List of C++ functions to expose to Python

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `name` | Unique name for your module |
| `cpp_sources` | String or list of C++ source code |
| `cuda_sources` | String or list of CUDA kernel code |
| `functions` | List of function names to expose to Python |
| `extra_cuda_cflags` | Compiler flags for CUDA compilation (e.g., `['-O2']`) |
| `build_directory` | Where to store build artifacts |
| `verbose` | Enable detailed compilation output |
| `with_pytorch_error_handling` | Wrap functions with PyTorch error handling (default: True) |

### Workflow

```
Python code
    ↓
C++ wrapper function (handles tensor conversion, grid/block setup)
    ↓
CUDA kernel (raw GPU computation)
    ↓
Result tensor returned to Python
```

---

## Understanding the Wrapper Function

The **wrapper function** is a C++ function that bridges Python and your CUDA kernel. It's NOT the kernel itself—it's a helper that:
- Takes PyTorch tensors as arguments
- Validates input (device, dtype)
- Extracts tensor metadata (dimensions, pointers)
- Configures grid and block dimensions
- Launches the CUDA kernel
- Returns PyTorch tensors

### Example: RGB to Grayscale Wrapper

```cpp
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel
__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int outputOffset = row * width + col;
        int inputOffset = (row * width + col) * channels;
        
        unsigned char r = input[inputOffset + 0];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];
        
        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

// C++ Wrapper function (exposed to Python)
torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    // 1. Input validation
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    
    // 2. Extract dimensions
    const auto height = image.size(0);
    const auto width = image.size(1);
    
    // 3. Allocate output tensor
    auto result = torch::empty({height, width, 1}, 
        torch::TensorOptions()
            .dtype(torch::kByte)
            .device(image.device()));
    
    // 4. Configure grid and block dimensions
    dim3 threads_per_block(16, 16);  // 256 threads
    dim3 number_of_blocks(
        (width + 15) / 16,   // ceil(width / 16)
        (height + 15) / 16   // ceil(height / 16)
    );
    
    // 5. Launch kernel
    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, 
        torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );
    
    // 6. Error checking
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    // 7. Return result
    return result;
}
```

### Wrapper Function Breakdown

**Input Validation:**
```cpp
assert(image.device().type() == torch::kCUDA);  // Must be on GPU
assert(image.dtype() == torch::kByte);           // Must be uint8
```

**Extract Metadata:**
```cpp
const auto height = image.size(0);
const auto width = image.size(1);
```

**Allocate Output:**
```cpp
auto result = torch::empty({height, width, 1}, 
    torch::TensorOptions()
        .dtype(torch::kByte)
        .device(image.device()));
```

**Configure Execution:**
```cpp
dim3 threads_per_block(16, 16);  // 16×16 = 256 threads per block
dim3 number_of_blocks(
    (width + 15) / 16,   // Ceiling division for X
    (height + 15) / 16   // Ceiling division for Y
);
```

**Launch Kernel:**
```cpp
kernel_name<<<grid, threads, shared_memory, stream>>>(args...);
```

**Error Checking:**
```cpp
C10_CUDA_KERNEL_LAUNCH_CHECK();  // Calls cudaGetLastError()
```

---

## CUDA Kernel Fundamentals

### What is a CUDA Kernel?

A CUDA kernel is a function marked with `__global__` that runs on the GPU. Each thread executes the same code with different data.

### Thread Organization

```cpp
// Getting thread position
int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column (X-axis)
int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row (Y-axis)
```

**Components:**
- `blockIdx`: Which block is this thread in? (0 to gridDim)
- `blockDim`: How many threads per block? (typically 16×16 = 256)
- `threadIdx`: Which thread within the block? (0 to blockDim)

### Data Types in Kernels

#### unsigned char (8-bit)

Used for standard image processing because:
- Range: 0-255 (perfect for pixel values)
- Memory efficient (1 byte per value vs 4 bytes for float)
- Standard image format (JPEG, PNG use this)

**Type Promotion Example:**
```cpp
unsigned char r = 200;           // 8-bit value
float result = 0.21f * r;        // r promoted to float (200.0f)
                                 // 0.21f * 200.0f = 42.0f
```

Intermediate calculations happen in floating-point, then cast back to char:
```cpp
output = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
```

---

## Memory Layout and Data Access

### Two Common Memory Layouts

#### Channels Last Layout
Used for: **RGB to Grayscale conversion**

```
Memory organization:
[R₀₀, G₀₀, B₀₀, R₀₁, G₀₁, B₀₁, R₀₂, G₀₂, B₀₂, ...]
└─ pixel(0,0) ┘ └─ pixel(0,1) ┘ └─ pixel(0,2) ┘
```

**Tensor shape:** `[height, width, 3]`

**Indexing formula:**
```cpp
int inputOffset = (row * width + col) * channels;
// Access: R = inputOffset + 0, G = inputOffset + 1, B = inputOffset + 2
```

**Why this layout:**
- RGB values grouped together
- Easy to access all channels for a single pixel
- Natural for file I/O (how images are stored on disk)

#### Channels First Layout
Used for: **Mean/blur filters**

```
Memory organization:
[R₀₀, R₀₁, R₀₂, ..., R₍ₕ₋₁₎₍w₋₁₎, G₀₀, G₀₁, ..., G₍ₕ₋₁₎₍w₋₁₎, B₀₀, B₀₁, ..., B₍ₕ₋₁₎₍w₋₁₎]
└─ all red pixels ────────────────┘ └─ all green ──────┘ └─ all blue ──────┘
```

**Tensor shape:** `[3, height, width]`

**Indexing formula:**
```cpp
int channel = threadIdx.z;  // Which channel (0=R, 1=G, 2=B)
int baseOffset = channel * height * width;
// Then access: baseOffset + row * width + col
```

**Why this layout:**
- Each channel can be processed independently
- Better for parallel processing across channels
- Efficient when threads specialize by channel

### Mean Filter Kernel Example

```cpp
__global__
void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;  // Each thread handles one channel
    
    int baseOffset = channel * height * width;  // Start of this channel's data
    
    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;
        
        // Apply mean filter in the neighborhood
        for (int blurRow = -radius; blurRow <= radius; blurRow++) {
            for (int blurCol = -radius; blurCol <= radius; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }
        
        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

### Understanding baseOffset

For a 3×3 RGB image in Channels First format:

```
Memory indices:
Channel 0 (Red):    [0-8]   → baseOffset = 0 * 9 = 0
Channel 1 (Green):  [9-17]  → baseOffset = 1 * 9 = 9
Channel 2 (Blue):   [18-26] → baseOffset = 2 * 9 = 18
```

To access pixel at row=1, col=2 in each channel:
- Red:   `0 + 1*3 + 2 = 5`
- Green: `9 + 1*3 + 2 = 14`
- Blue:  `18 + 1*3 + 2 = 23`

Each thread with different `threadIdx.z` knows its channel's starting position via `baseOffset`.

---

## Complete Examples

### Example 1: RGB to Grayscale

**Input Format:**
```python
x = read_image("image.jpg").permute(1, 2, 0).cuda()
# Shape: [height, width, 3] - Channels Last
```

**Kernel:**
```cpp
__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int outputOffset = row * width + col;
        int inputOffset = (row * width + col) * channels;  // Channels Last indexing
        
        unsigned char r = input[inputOffset + 0];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];
        
        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}
```

**Thread Configuration:**
- 2D threads: `(16, 16)` - one thread per pixel
- Accesses all 3 channels sequentially within thread

### Example 2: Mean Filter

**Input Format:**
```python
x = read_image("image.jpg").contiguous().cuda()
# Shape: [3, height, width] - Channels First
```

**Kernel:**
```cpp
__global__
void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;  // Channels First with parallel processing
    
    int baseOffset = channel * height * width;  // Channels First indexing
    
    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;
        
        for (int blurRow = -radius; blurRow <= radius; blurRow++) {
            for (int blurCol = -radius; blurCol <= radius; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }
        
        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

**Thread Configuration:**
- 3D threads: `(16, 16, 3)` - one thread per pixel-channel combination
- Each channel processed independently in parallel

---

## Key Takeaways

1. **Input format determines kernel design:** Use `.permute()` to match your kernel's expected memory layout

2. **Channels Last** for operations needing all channels together (RGB→Grayscale)

3. **Channels First** for operations independent per channel (blurs, filters)

4. **The wrapper function** handles PyTorch bookkeeping; the kernel handles computation

5. **Memory indexing** must match your data layout—get it wrong and you access garbage data

6. **Type promotion** happens automatically in C++ (char * float = float), so intermediate precision is maintained even when input/output are 8-bit

7. **Thread configuration** (2D vs 3D) should match your parallelization strategy
