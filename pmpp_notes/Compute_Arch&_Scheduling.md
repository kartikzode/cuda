# Compute Arcitecture & Scheduling

## Streaming Multiprocessors
 - SMs consist of several processing units called streaming processors/ CUDA cores that share control logic and memory resources.
 - Threads are assiged(simultaneously) to SMs on a block-by-block basis.
 - A block can begin execution only when the runtime system has secured all the resources needed by all the threads in the block to complete execution.
 - Blocks can be executed in any order on the SMs -> led to **transparent scalability**.
 - SMs have a **thread/SM**, **blocks/SM**, **registers/SM** limits.

## Barrier Synchronization
 - method for coordinating parallel activities using __syncthreads() function.
 - imposes execution constraints on threads within a block. 
 - __syncthreads() should be executed either by all threads or no thread in a block
 - ensures all threads have completed execution of a previous phase before any of them start the next phase.

## Thread Sceduling
 - hardware implementation concept
 - once a block is assigned to a SM, it is split into **32-thread** units called ***warps***.
 - A ***Warp*** is a unit of thread scheduling in SMs
 - During *Thread divergence*, the passes may be executed concurrently (interleaved executions), **Independent Thread Scheduling**

## SIMD (Single Instruction Multiple Data) Model
- One instruction is fetched and executed for all threads.
- GPU --> GPCs --> SMs --> Processing Blocks --> Cores.
- Threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time.

## Control Divergence
 - Threads in the same warp executing different paths lead to control divergence. Hardware has to take multiple passes through these paths, one pass for each path.
 - SIMD and divergence lead to extra passes, execution resources that are consumed by the inactive threads in each pass.
 - Decision conditions involving `threadIdx` values can lead to control divergence.
 - Used to handle boundary conditions when mapping threads to data. Larger the size of vectors being processed, lesser the impact of control divergence.

 ## Latency Tolerance & Warp Sceduling
  - **Block** is the unit of resource allocation & ***Warp*** is the unit of scheduling and latency hiding.
  - Mechanism for filling the latency time of operations for some threads (global memory access, pipelined floating-point arithmetic, branch instructions) with work from other threads.
  - ***Zero-Overhead Thread Scheduling*** does not introduce idle/ wasted time on the execution timeline. GPU does that by holding all the execution states for the assigned warps into the hardware registers so no need to save and restore states when swithing between warps.
  - We assign much larger number of threads to a SM than it can simultaneously support with its execution resources so that there always is a thread that is ready to execute at any point in time. (Oversubscription of threads necessary for latency tolerance)

## Occupancy
 - Ratio of the number of warps assigned to an SM to the maximum number it supports is reffered to as **Occupancy**.
 - Actual concurrency limits might vary with the gpu arch., for turing/ampere architectures atleast, the limit is ***64 Warps/2048 Threads*** per SM.
 - 2 blocks with 1024 threads <---> 32 blocks with 64 threads (*Dynamic Thread Partition*).
 - Register resource limitations might lead to wasted threads, **Performance Cliff** in which a slight increase in resource usage can result in significant reduction in parallelism and performance achieved.