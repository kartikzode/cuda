# Notes from Chapter 1 of PMPP book

A ***Thread*** is a simplified view of how a processor executes a sequential program. It consits of the code  of the program, the instruction pointer and the values of its variables and data structures

## Two main trajectories

![CPU/GPU Design Philosophies](./assets/CPUvsGPU.png)

- ### Multi Core Trajectory

  - Focus is on executing multiple instruction sequences simlutaneously on the processor cores.
  - *Latency Oriented Design* to optmize sequential code preformance by using sophisticated algorithms to capture frequently accessed data into cache thereby reducing latency of operations for each individual thread at the cost of increased chip area and power per unit.
  - Low latency arithmatic units, sophisticated operand dilevery logic, large cache memory, control logic consume chip area and power that could have been used to provide more arithmatic execution units and memory access channels.

- ### Many Thread Trajectory

  - *Throughput Oriented Design* focuses on the maximization of total execution throughput of parallel applications by emphasis on performing a massive number of floating point operations and memory accesses.
  - *Memory Bandwidth*, rate of moving data into and out of memory systems is very high.
  - Reducing latency is more expensive as compared to increasing throughput in terms of power and chip area.
  - The speedup acheivable by a parallel computing system over a serial computing system depends on the portion of application that can be parallelized.
  - Getting around memory bandwidth limitations to reduce the number of accesses to the DRAM and limited on-chip memory capacity

## Challenges in parallel programming

- Algorithmic (Computational) Complexity
- Memory access latency and/or throughput (memory bound)
- compute bound (number of instructions performed per byte of data)
- Input data Characteristics
- Synchronization overhead

***MPI***, Message Passing Interface is a programming interface used for data sharing and interaction between computing nodes that do not share memory.

***NCCL***, Nvidia Collective Communications Library supports multi-GPU programming in cuda.

***OpenCL***, Open Compute Language is a programming model similar to CUDA that defines language extensions and runtime APIs for managing parallelism and data delivery in massively parallel processors.

## Note

- Barrier Synchronization, memory consistency and  ??
- regularize and localize memory data accesses to minimize consumption of critical resources and conflicts in updating data structures.