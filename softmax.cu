#include <cuda_runtime.h>
#include <math_constants.h>

#include <iostream>

__global__ void softmax_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    int N)
{
   // One cuda block processes one row
   int row = blockIdx.x;
   int tid = threadIdx.x;

   const float *x = input + (size_t)row * N;
   float *y = output + (size_t)row * N;

   // Assumption: blockDim.x is a power of two
   extern __shared__ float shared[];

   // Step 1: Local maximum
   float local_max = -CUDART_INF_F;

   for (int i = tid; i < N; i += blockDim.x)
   {
      local_max = fmaxf(local_max, x[i]);
   }
   shared[tid] = local_max;
   __syncthreads();

   // Step 2 Block-wise maximum reduction
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
   {
      if (tid < stride)
      {
         shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
      }
      __syncthreads();
   }

   float row_max = shared[0];
   // Step 3: Local sum of exponentials
   float local_sum = 0.0f;

   for (int i = tid; i < N; i += blockDim.x)
   {
      local_sum += expf(x[i] - row_max);
   }

   shared[tid] = local_sum;
   __syncthreads();

   // Step 4: Block wise sum reduction
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
   {
      if (tid < stride)
      {
         shared[tid] += shared[tid + stride];
      }
      __syncthreads();
   }

   float denominator = shared[0];

   // Step 5: Normalize and write output
   for(int i = tid; i < N; i+=blockDim.x){
      y[i] = expf(x[i]-row_max)/denominator;
   }
}

void launch_softmax(
   const float* input,
   float* output,
   int B,
   int N
){
   constexpr int THREADS = 256;

   softmax_kernel<<<B,THREADS,THREADS*sizeof(float)>>>(input,output,N);
}

