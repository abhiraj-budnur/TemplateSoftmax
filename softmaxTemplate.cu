#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int N = 16384;
constexpr int THREADS = 512;
constexpr int WARPS = THREADS / 32;
constexpr unsigned int FULL_MASK = 0xffffffffu;

#define CUDA_CHECK(call)                                                                                               \
   do                                                                                                                  \
   {                                                                                                                   \
      cudaError_t err = (call);                                                                                        \
      if (err != cudaSuccess)                                                                                          \
      {                                                                                                                \
         std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
         std::exit(EXIT_FAILURE);                                                                                      \
      }                                                                                                                \
   } while (0)

__device__ __forceinline__ float warp_reduce_max(float v)
{
#pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1)
   {
      float other = __shfl_down_sync(FULL_MASK, v, offset);
      v = fmaxf(v, other);
   }
   return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1)
   {
      v += __shfl_down_sync(FULL_MASK, v, offset);
   }
   return v;
}

template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float *scratch)
{
   const int lane = threadIdx.x & 31;
   const int warp = threadIdx.x >> 5;

   if constexpr (IS_MAX)
   {
      v = warp_reduce_max(v);
   }
   else
   {
      v = warp_reduce_sum(v);
   }

   // One partial result per warp
   if (lane == 0)
   {
      scratch[warp] = v;
   }
   __syncthreads();

   // First warp reduces the WARPS partial results.
   if (warp == 0)
   {
      v = (lane < WARPS) ? scratch[lane] : (IS_MAX ? -CUDART_INF_F : 0.0f);

      if constexpr (IS_MAX)
      {
         v = warp_reduce_max(v);
      }
      else
      {
         v = warp_reduce_sum(v);
      }
      if (lane == 0)
      {
         scratch[0] = v;
      }
   }

   __syncthreads();
   return scratch[0];
}

template <int BLOCK_THREADS>
__global__ void softmax_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    int rows)
{
   constexpr int ITEMS = N / BLOCK_THREADS;

   const int row = blockIdx.x;
   const int tid = threadIdx.x;

   if (row >= rows)
   {
      return;
   }

   const float *x = input + static_cast<size_t>(row) * N;
   float *y = output + static_cast<size_t>(row) * N;

   __shared__ float scratch[BLOCK_THREADS / 32];

   // Cache values in registers
   float values[ITEMS];
   float local_max = -CUDART_INF_F;

   #pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      float value = x[tid + i * BLOCK_THREADS];
      values[i] = value;
      local_max = fmaxf(local_max, value);
   }
   const float row_max = block_reduce<true>(local_max, scratch);

   // Exponentiate and compute local sum
   float local_sum = 0.0f;

   #pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      values[i] = __expf(values[i] - row_max);
      local_sum += values[i];
   }

   const float denominator = block_reduce<false>(local_sum, scratch);

   const float inv_sum = 1.0f / denominator;

   // Store normalized values
   #pragma unroll
   for (int i = 0; i < ITEMS; ++i)
   {
      y[tid + i * BLOCK_THREADS] = values[i] * inv_sum;
   }
}

void launch_softmax(
    const float *input,
    float *output,
    int rows,
    cudaStream_t stream = nullptr)
{
   softmax_kernel<THREADS><<<rows, THREADS, 0, stream>>>(input, output, rows);

   CUDA_CHECK(cudaGetLastError());
}

void cpu_softmax(
   const std::vector<float>& input,
   std::vector<float>& output,
   int rows
){
   for(int r = 0; r< rows; ++r){
      const size_t base = static_cast<size_t>(r)*N;
      float max_value = -INFINITY;

      for(int j = 0; j< N; ++j){
         max_value = std::max(
            max_value,input[base+j]
         );
      }
      double sum = 0.0;
      for(int j = 0; j < N; ++j){
         float e = std::exp(
            input[base+j] - max_value
         );
         output[base+j] = e;
         sum+=e;
      }
      const float inv_sum = static_cast<float>(1.0/sum);
      for(int j = 0; j <N;++j){
         output[base+j] *=inv_sum;
      }
   }
}

void generate_random_input(
    std::vector<float> &input,
    unsigned int seed)
{
   std::mt19937 generator(seed);

   // Standard normal distribution: mean 0, stddev 1.
   std::normal_distribution<float> distribution(0.0f, 1.0f);
   for (float &value : input)
   {
      value = distribution(generator);
   }
}

struct Errorstats
{
   float max_abs_error = 0.0f;
   float max_row_sum_error = 0.0f;
   bool finite = true;
   bool nonnegative = true;
};

Errorstats validate_output(
    const std::vector<float> &actual,
    const std::vector<float> &refernce,
    int rows)
{
   Errorstats stats;

   for (int r = 0; r < rows; ++r)
   {
      const size_t base = static_cast<size_t>(r) * N;
      double row_sum = 0.0;

      for (int j = 0; j < N; ++j)
      {
         const float a = actual[base + j];
         const float b = refernce[base + j];

         if (!std::isfinite(a))
         {
            stats.finite = false;
         }
         if (a < 0.0f)
         {
            stats.nonnegative = false;
         }
         stats.max_abs_error = std::max(
             stats.max_abs_error,
             std::fabs(a - b));
         row_sum += a;
      }
      stats.max_row_sum_error = std::max(
          stats.max_row_sum_error,
          static_cast<float>(std::fabs(row_sum - 1.0)));
   }
   return stats;
}

float benchmark_gpu_ms(
    const float *d_input,
    float *d_output,
    int rows,
    int warmup = 20,
    int iterations = 100)
{
   for (int i = 0; i < warmup; ++i)
   {
      launch_softmax(d_input, d_output, rows);
   }

   CUDA_CHECK(cudaDeviceSynchronize());

   cudaEvent_t start, stop;
   CUDA_CHECK(cudaEventCreate(&start));
   CUDA_CHECK(cudaEventCreate(&stop));

   CUDA_CHECK(cudaEventRecord(start));

   for (int i = 0; i < iterations; ++i)
   {
      launch_softmax(d_input, d_output, rows);
   }

   CUDA_CHECK(cudaEventRecord(stop));
   CUDA_CHECK(cudaEventSynchronize(stop));

   float elapsed_ms = 0.0f;

   CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

   CUDA_CHECK(cudaEventDestroy(start));
   CUDA_CHECK(cudaEventDestroy(stop));

   return elapsed_ms / iterations;
}

int main(
    int argc,
    char **argv)
{
   try
   {
      int rows = 128;
      unsigned int seed = 12345;

      if (argc >= 2)
      {
         rows = std::stoi(argv[1]);
      }

      if (argc >= 3)
      {
         seed = static_cast<unsigned int>(std::stoul(argv[2]));
      }
      if (rows <= 0)
      {
         throw std::invalid_argument(
             "rows must be positive");
      }

      int device = 0;
      CUDA_CHECK(cudaSetDevice(device));

      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(
          &prop, device));

      if (THREADS > prop.maxThreadsPerBlock)
      {
         throw std::runtime_error(
             "THREADS exceeds device block limit");
      }

      std::cout << "-----------------" << std::endl;
      std::cout << "CUDA Softmax" << std::endl;
      std::cout << "-----------------" << std::endl;
      std::cout << "GPU: " << prop.name << "\n";
      std::cout << "Rows: " << rows << "\n";
      std::cout << "Row lenght: " << N << "\n";
      std::cout << "Data Type: FP32\n";
      std::cout << "Threads/Block: " << THREADS << "\n";
      std::cout << "Seed: " << seed << "\n\n";

      const size_t count = static_cast<size_t>(rows) * N;
      const size_t bytes = count * sizeof(float);

      std::vector<float> h_input(count);
      std::vector<float> h_output(count);
      std::vector<float> h_refernce(count);

      generate_random_input(h_input, seed);

      // Allocate GPU Buffers
      float *d_input = nullptr;
      float *d_output = nullptr;

      CUDA_CHECK(cudaMalloc(&d_input, bytes));
      CUDA_CHECK(cudaMalloc(&d_output, bytes));

      CUDA_CHECK(cudaMemcpy(
          d_input,
          h_input.data(),
          bytes,
          cudaMemcpyHostToDevice));
      launch_softmax(d_input, d_output, rows);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(
          h_output.data(),
          d_output,
          bytes,
          cudaMemcpyDeviceToHost));

      
      cpu_softmax(h_input,h_refernce,rows);

      const Errorstats stats = validate_output(
          h_output, h_refernce, rows);

      constexpr float ATOL = 1e-6f;
      constexpr float RTOL = 1e-5f;

      const float error_limit = ATOL + RTOL;

      const bool correct = stats.finite && stats.nonnegative && stats.max_abs_error <= error_limit && stats.max_row_sum_error <= 1e-5;

      std::cout << std::scientific << std::setprecision(6);

      std::cout << "Correctness\n";
      std::cout << "All finite   " << (stats.finite ? "YES" : "NO") << "\n";
      std::cout << "Non-Negative   " << (stats.nonnegative ? "YES" : "NO") << "\n";
      std::cout << "Max Abs Error   " << (stats.max_abs_error ? "YES" : "NO") << "\n";
      std::cout << "Max Row Sum Error   " << (stats.max_row_sum_error ? "YES" : "NO") << "\n";
      std::cout << "Result   " << (correct ? "PASS" : "FAIL") << "\n";

      const float latency_ms = benchmark_gpu_ms(d_input, d_output, rows);

      const double traffic_bytes = 2.0 * static_cast<double>(bytes);

      const double bandwidth_gbs = traffic_bytes / (latency_ms * 1.0e6);

      std::cout << "Performance\n";
      std::cout << "Kernel Latency: " << latency_ms << " ms\n";
      std::cout << "Effective BW: " << bandwidth_gbs << " GB/s\n";
      std::cout << "Elements/second: " << (static_cast<double>(count) / (latency_ms * 1.0e-3)) << "\n";

      CUDA_CHECK(cudaFree(d_input));
      CUDA_CHECK(cudaFree(d_output));

      return correct ? EXIT_SUCCESS : EXIT_FAILURE;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error: " << e.what() << "\n";
      return EXIT_FAILURE;
   }
}