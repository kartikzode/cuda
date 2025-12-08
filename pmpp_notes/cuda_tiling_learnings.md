# GPU Architecture & Tiled Matrix Multiplication: Key Concepts

## 1. The Relationship Between Tile Size and Hardware Limits

When designing a tiled CUDA kernel, the choice of **Tile Size** is not arbitrary. It is constrained by three physical hardware limits that must be balanced simultaneously.

### A. The Thread Limit (Block Size)
*   **The Constraint:** Modern NVIDIA GPUs have a hard limit of **1024 threads per block**.
*   **The Implication:** In a naive "1 Thread = 1 Output" implementation, your Tile Size ($T$) determines your Block Size ($T \times T$).
    *   $T=32 \rightarrow 32 \times 32 = 1024$ threads. (Matches Limit $\rightarrow$ **Optimal**)
    *   $T=64 \rightarrow 64 \times 64 = 4096$ threads. (Exceeds Limit $\rightarrow$ **Invalid Argument Error**)
*   **The Lesson:** To use data tiles larger than $32 \times 32$, you cannot simply increase the block dimensions. You must decouple the *Thread Block Size* from the *Data Tile Size* using **Thread Coarsening** (where one thread computes multiple output elements).

### B. The Shared Memory Limit
*   **The Constraint:** GPUs have a fixed amount of Shared Memory per Streaming Multiprocessor (SM) and per Block (e.g., 48 KB or 64 KB).
*   **The Calculation:** Memory usage scales quadratically with tile size.
    $$ \text{Memory} = 2 \times T \times T \times \text{sizeof(float)} $$
*   **The Implication:**
    *   $T=32$: Requires ~8 KB. (Safe)
    *   $T=96$: Requires ~72 KB. (Exceeds 48 KB limit $\rightarrow$ **Compile Time Error** or **Launch Failure**)
*   **The Lesson:** Doubling the tile size quadruples the memory requirement. Even if threads weren't an issue, memory capacity acts as a hard ceiling.

### C. The Register Limit
*   **The Constraint:** Each SM has a fixed pool of registers (e.g., 65,536). These are divided among all active threads.
*   **The Trade-off:** Increasing threads per block decreases the registers available per thread.
*   **The Risk:** If a kernel needs more registers than available, the compiler "spills" data to Local Memory (slow RAM), drastically reducing performance.

---

## 2. Warp Execution and Scheduling

Understanding how the hardware executes the code helps explain performance behaviors and synchronization overhead.

### A. Warps vs. Blocks
*   **Concept:** While software defines 2D blocks (e.g., 32x32), hardware executes 1D **Warps** (groups of 32 threads).
*   **Mapping:** A $32 \times 32$ block is flattened into 32 Warps ($1024 / 32$).
*   **Execution:** The SM does not run all threads simultaneously. It uses a **Warp Scheduler** to time-slice them.

### B. Latency Hiding
*   **The Mechanism:** The SM relies on having many active warps to hide memory latency.
*   **The Flow:**
    1.  Warp A issues a memory load (slow) and goes to sleep.
    2.  The Scheduler immediately switches to Warp B to do math.
    3.  By the time the Scheduler returns to Warp A, the data has arrived.
*   **The Requirement:** You need enough active warps (Occupancy) to keep the pipeline full. However, using too much Shared Memory per block limits the number of blocks that can run on an SM, potentially hurting occupancy.

### C. Synchronization Costs (`__syncthreads`)
*   **The Barrier:** `__syncthreads()` forces **all** warps in a block to reach a specific point before any can proceed.
*   **The Convoy Effect:** If a block has many threads (e.g., the theoretical 4096 case), the entire block moves at the speed of the *slowest* warp.
*   **The Cost:** As block size increases, the probability of one warp stalling the entire group increases, making barriers more expensive.

---

## 3. Summary of Best Practices

1.  **Standard Tile Size:** For square tiles without thread coarsening, **32x32** is the mathematical maximum and usually the most efficient configuration for current hardware.
2.  **Decoupling:** High-performance kernels (like cuBLAS) use large data tiles (e.g., 128x128) but small thread blocks (e.g., 128 or 256 threads) by making each thread compute multiple values (Thread Coarsening/Register Tiling).
3.  **Error Checking:** Always check `cudaGetLastError()` after kernel launches to catch "silent" failures (like invalid block dimensions).
