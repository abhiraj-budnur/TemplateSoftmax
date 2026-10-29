CUDA Softmax Implementations

This project explores two CUDA implementations of the Softmax operation using single-precision floating-point (FP32) data:

softmax.cu — A basic Softmax kernel using shared memory and block-wide reductions.
softmaxTemplate.cu — A template-based, optimized Softmax kernel using warp-level shuffle instructions, shared memory, and register caching.

The goal is to demonstrate GPU parallelization, numerically stable Softmax computation, reduction techniques, correctness validation, and performance benchmarking on an NVIDIA GPU.

Table of Contents
Overview
Softmax Mathematics
1. Basic Softmax
2. Template-Based Softmax
CUDA Implementation Details
Correctness Validation
Performance Benchmarking
Requirements
Compilation and Execution
Terminal Output
Performance Analysis
Implementation Status
Conclusion
Overview

Softmax converts a vector of real-valued inputs into a probability distribution. Each output is nonnegative, and the outputs of a row sum to approximately one.

For a row 
𝑥
 of length 
𝑁
, Softmax is defined as:

𝑦
𝑖
=
𝑒
𝑥
𝑖
∑
𝑗
=
0
𝑁
−
1
𝑒
𝑥
𝑗

Directly evaluating the exponential can cause numerical overflow for large inputs. A numerically stable formulation subtracts the maximum input value before computing exponentials:

𝑚
=
max
⁡
𝑗
𝑥
𝑗
𝑦
𝑖
=
𝑒
𝑥
𝑖
−
𝑚
∑
𝑗
=
0
𝑁
−
1
𝑒
𝑥
𝑗
−
𝑚

Subtracting the row maximum preserves the mathematical result while ensuring that the largest exponent is 
𝑒
0
=
1
.

Both implementations assign one CUDA block to process one row independently.

Softmax Mathematics

The computation consists of three main stages:

Maximum reduction: Find the maximum element in each row.
Exponential and sum reduction: Calculate 
𝑒
𝑥
𝑖
−
𝑚
 for each element and sum the exponentials.
Normalization: Divide each exponential by the row's sum.

These stages require synchronization because every output element depends on reductions over the entire row.

1. Basic Softmax

File: softmax.cu

The basic implementation uses a shared-memory reduction to calculate the row maximum and the sum of exponentials.

Kernel configuration
Parameter	Value
Data type	FP32
Default block size	256 threads
Shared memory	256 floats per block
Rows processed	One row per block
Input row length	Runtime parameter N
Execution steps
Each thread scans a subset of the row to find a local maximum.
Threads store local maxima in shared memory.
A tree reduction calculates the row maximum.
Each thread computes the exponential of its assigned input elements after subtracting the row maximum.
A second shared-memory tree reduction calculates the sum of exponentials.
Threads normalize their assigned elements and store the output.

The implementation uses fmaxf() for maximum reduction and expf() for exponential evaluation.

Kernel launch

The host-side launch wrapper is:

cpp
void launch_softmax(
    const float* input,
    float* output,
    int B,
    int N)
{
    constexpr int THREADS = 256;

    softmax_kernel<<<
        B, THREADS, THREADS * sizeof(float)
    >>>(input, output, N);
}


Here, B represents the number of rows. The kernel requires a power-of-two block size for its current reduction implementation.

Current status: The kernel and launch wrapper are implemented, but a complete main() function for standalone testing and benchmarking has not yet been developed.

2. Template-Based Softmax

File: softmaxTemplate.cu

This implementation uses a compile-time row length and configurable block size. It replaces the basic shared-memory reduction across all threads with a hierarchical warp-level and block-level reduction.

Kernel configuration
Parameter	Value
GPU	NVIDIA GeForce MX330
Data type	FP32
Row length (N)	16,384
Threads per block	512
Warps per block	16
Default number of rows	128
Random seed	12345

Each thread processes:

𝑁
THREADS
=
16384
512
=
32

elements per row.

Execution steps
Load input values into registers.
Calculate each thread's local maximum.
Reduce local maxima within each warp using __shfl_down_sync().
Store one partial maximum per warp in shared memory.
Use the first warp to calculate the row maximum.
Compute exponentials and accumulate local sums.
Reduce local sums within warps and across the block.
Normalize the register-cached exponentials and write the output.
Warp-level reduction

The implementation uses warp shuffle instructions:

cpp
__shfl_down_sync(FULL_MASK, v, offset);


This enables efficient reduction within a warp without using shared memory for every reduction step.

Block-level reduction

Each warp produces a partial result, which is stored in shared memory. The first warp then combines those partial results to obtain the row-wide maximum or sum.

The implementation provides a templated reduction function:

cpp
template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(
    float v,
    float *scratch);


The compile-time IS_MAX parameter selects either maximum reduction or sum reduction.

Register caching

The input row is stored in a per-thread register array:

cpp
float values[ITEMS];


After computing the row maximum, the array is reused to hold the exponentials. This avoids reloading the input values for the normalization stage and reduces global-memory traffic.

The kernel is launched with:

cpp
softmax_kernel<THREADS>
    <<<rows, THREADS, 0, stream>>>(
        input, output, rows);

CUDA Implementation Details
Shared memory

Shared memory stores partial reduction results from each warp. The template implementation uses:

cpp
__shared__ float scratch[BLOCK_THREADS / 32];


With 512 threads per block, this array contains 16 floats.

Synchronization

__syncthreads() ensures that partial results are available before the next reduction stage begins. It is used between the warp-level stores, the first-warp reduction, and the subsequent consumption of the final result.

Register pressure

Caching 32 values per thread reduces repeated global-memory loads, but register usage can affect occupancy. The optimal block size depends on the GPU architecture, row length, compiler behavior, and register allocation.

Numerical stability

Both kernels subtract the row maximum before evaluating exponentials. This avoids the overflow associated with directly exponentiating large positive values.

Correctness Validation

The template implementation compares GPU output with a CPU Softmax reference.

It checks three properties:

Finite outputs: Every GPU output must be finite.
Nonnegative outputs: Every output must be greater than or equal to zero.
Row-sum consistency: The sum of each output row should be approximately one.

It also calculates the maximum absolute error relative to the CPU reference.

The configured tolerances are:

cpp
constexpr float ATOL = 1e-6f;
constexpr float RTOL = 1e-5f;


The row-sum tolerance is 1e-5.

The validation result is reported as PASS when all configured conditions are satisfied.

Reporting note: The current terminal output displays YES for the maximum absolute error and maximum row-sum error because the print statements test those values as Boolean conditions. The actual numerical errors are not displayed yet. The README therefore records the reported correctness status without inventing numerical error values.

Performance Benchmarking

The template implementation uses CUDA events to measure average kernel execution time.

Benchmark parameter	Value
Warm-up iterations	20
Timed iterations	100
Rows	128
Elements per row	16,384
Total output elements	2,097,152
Metrics

Kernel latency

The average time taken to execute the Softmax kernel, reported in milliseconds.

Effective bandwidth

The implementation estimates logical memory traffic as one input read and one output write:

𝐵
logical
=
2
×
input tensor bytes

Effective bandwidth is calculated by dividing this estimated traffic by the average kernel latency.

This is a logical traffic estimate rather than a direct measurement of physical DRAM traffic.

Elements per second

The total number of output elements divided by average kernel execution time.

Requirements
NVIDIA GPU with CUDA support.
NVIDIA CUDA Toolkit and nvcc.
Compatible NVIDIA GPU driver.
C++ compiler supported by the installed CUDA Toolkit.

Check the environment:

bash
nvcc --version
nvidia-smi

Compilation and Execution

Compile the template implementation:

bash
nvcc -O3 -std=c++17 softmaxTemplate.cu -o softmax_template


Run with the default configuration:

bash
./softmax_template


Run with a custom row count and random seed:

bash
./softmax_template 128 12345


The executable accepts two optional arguments:

rows: Number of rows to process; default is 128.
seed: Random seed; default is 12345.

The basic softmax.cu implementation currently provides only the kernel and launch wrapper. It needs a host-side test harness before it can be compiled and executed as a standalone benchmark using the same workflow.

Terminal Output

The following output was collected from softmaxTemplate.cu on an NVIDIA GeForce MX330.

text
-----------------
CUDA Softmax
-----------------
GPU: NVIDIA GeForce MX330
Rows: 128
Row lenght: 16384
Data Type: FP32
Threads/Block: 512
Seed: 12345

Correctness
All finite   YES
Non-Negative   YES
Max Abs Error   YES
Max Row Sum Error   YES
Result   PASS
Performance
Kernel Latency: 3.481497e-01 ms
Effective BW: 4.818965e+01 GB/s
Elements/second: 6.023707e+09

Performance Analysis

The measured template Softmax performance on the NVIDIA GeForce MX330 is summarized below.

Metric	Measured result
Row count	128
Row length	16,384
Total elements	2,097,152
Kernel latency	0.348150 ms
Effective bandwidth	48.19 GB/s
Elements per second	6.024 billion
Reported correctness	PASS

The kernel processes approximately 2.10 million elements in 0.348 milliseconds for the specified workload.

The reported effective bandwidth of approximately 48.19 GB/s is based on the implementation's estimate of one input read and one output write per element.

Since the basic implementation has not yet been benchmarked, a direct performance comparison between the two kernels is not currently available.

Implementation Status
Feature	softmax.cu	softmaxTemplate.cu
Stable Softmax	Implemented	Implemented
Row-wise maximum reduction	Shared-memory reduction	Warp and block reduction
Sum reduction	Shared-memory reduction	Warp and block reduction
Register caching	No explicit row cache	Implemented
Host-side main()	Pending	Implemented
CPU reference validation	Pending	Implemented
GPU benchmark	Pending	Implemented
Measured latency	Not available	0.348150 ms
Reported validation	Not available	PASS
Conclusion

This project demonstrates two approaches to implementing numerically stable Softmax in CUDA.

The basic implementation uses shared-memory tree reductions to calculate the row maximum and exponential sum. The template implementation uses warp shuffle operations, shared memory for partial reductions, and register caching to reduce intermediate global-memory traffic.

On an NVIDIA GeForce MX330, the template implementation achieved a measured latency of approximately 0.348 ms, effective logical bandwidth of approximately 48.19 GB/s, and throughput of approximately 6.02 billion elements per second for 128 rows of 16,384 FP32 elements.

The next step is to complete the host-side test harness for softmax.cu, validate both implementations under the same workload, and compare their measured performance.