#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

template <typename scalar_t>
__global__ void customconv1d_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> threshold,
    int nr_xnor_gates
  )
{

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // y
  const int d = blockIdx.y * blockDim.y + threadIdx.y; // x

  // make sure we don't modify memory regions outside of output
  if ((d < output.size(0)) && (c < output.size(1)))
  {
    // this is (c,d,e), we have as many threads as we have pixels in output out
    // each thread of out calculates a MAC (row of filter times column of input)

    // every thread is responsible for one sum, there are as many threads as mac sums in output
    output[d][c] = 0; // output buffer
    float mult_result = 0;
    float sub_popcnt = 0; // used for sub-popcount computations
    int cycle_counter = 0; // nr of cycles the tlu has executed

    for(int i = 0; i < weight.size(1); i++)
    {
      //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
      sub_popcnt += weight[c][i] * input[d][i];
      cycle_counter += 1;

      if (cycle_counter == nr_xnor_gates)
      {
        mult_result += sub_popcnt;
        sub_popcnt = 0;
        cycle_counter = 0;
      }

      // edge case
      if (i == weight.size(1)-1)
      {
        mult_result += sub_popcnt;
        // TODO: threshold for edge case round(((weight.size(1) % nr_xnor_gates) / nr_xnor_gates)*threshold[c])
      }

      #if 0
        if (d == 0 && c == 0)
        {
          // printf("f01: %.2f, f10: %.2f, seed0: %d, cantor_val: %d\n", f01, f10, seed0, cantor_val);
          printf("threshold: %.2f", threshold[c]);
          //printf("CUDA shape of weight [%d]", weight.size(0));
          //printf("CUDA shape of input [%d,%d]",  input.size(0), input.size(1));
          //printf("CUDA shape of output [%d,%d]\n\n", output.size(0), output.size(1));
        }
      #endif
    }
    output[d][c] = mult_result;
  }
}

torch::Tensor customconv1d_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output,
  torch::Tensor threshold,
  int nr_xnor_gates
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int output_size_x = output.size(1);
  const int output_size_y = output.size(0);
  int threads_x = 16; // per block, 16
  int threads_y = 16; // per block, 16

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
  #endif

  const dim3 threads(threads_x,threads_y);
  const dim3 blocks((output_size_x + threads_x - 1) / threads_x,
                    (output_size_y + threads_y - 1) / threads_y);

  AT_DISPATCH_ALL_TYPES(input.type(), "customconv1d_cuda", ([&] {
    customconv1d_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        threshold.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        nr_xnor_gates
    );
  }));

  return output;
}
