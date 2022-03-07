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
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> threshold,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> popc_acc,
    int nr_xnor_gates,
    int nr_additional_samples,
    int majv_shift,
    int threshold_scaling,
    int popc_acc_activate,
    int threshold_correction
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
    float result = 0;
    float sub_popcnt = 0; // used for sub-popcount computations
    float sub_popcnt_1 = 0; // used for sub-popcount computations with one more sample
    float sub_popcnt_2_1 = 0; // used for sub-popcount computations with two more samples
    float sub_popcnt_2_2 = 0; // used for sub-popcount computations with two more samples
    int global_cycles = 0; // count all cycles
    int cycle_counter = 0; // nr of cycles the tlu has executed at present
    float cycles = weight.size(1) / nr_xnor_gates; // nr of cycles the tlu has to execute
    float threshold_for_sample = round(threshold[c] / cycles);
    if (threshold_scaling == 2)
    {
      threshold_for_sample = 2*floorf(threshold_for_sample/2);
    }
    float last_threshold_for_sample = 0;
    int comparison = 0;
    // #if 1
    //   if (d == 0 && c == 2)
    //   {
    //     printf("a cycles: %.2f, threshold: %.2f, threshold_sample: %2.f\n", cycles, threshold[c], threshold_for_sample);
    //   }
    // #endif
    for(int i = 0; i < weight.size(1); i++)
    {
      //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
      sub_popcnt += weight[c][i] * input[d][i];
      cycle_counter += 1;

      // test for cycles
      // #if 1
      //   if (d == 0 && c == 0)
      //   {
      //     printf("cycles: %d, gcycles: %d\n", cycle_counter, global_cycles);
      //   }
      // #endif

      // one more sample, in the middle of two subsequent samples (overlapping)
      if (nr_additional_samples == 1)
      {
        if (i + nr_xnor_gates <= (weight.size(1) - 1))
        {
          sub_popcnt_1 += weight[c][i + (nr_xnor_gates / 2)] * input[d][i + (nr_xnor_gates / 2)];
        }
      }

      // two more samples
      if (nr_additional_samples == 2)
      {
        if (i + nr_xnor_gates <= (weight.size(1) - 1))
        {
          sub_popcnt_2_1 += weight[c][i + round((nr_xnor_gates)*(1/3))] * input[d][i + round((nr_xnor_gates)*(1/3))];
          sub_popcnt_2_2 += weight[c][i + round((nr_xnor_gates)*(2/3))] * input[d][i + round((nr_xnor_gates)*(2/3))];
        }
      }

      // when "nr_xnor_gates" many operations have been computed
      if (cycle_counter == nr_xnor_gates)
      {

        if (popc_acc_activate == 1)
        {
          //popc_acc[c][d] += sub_popcnt;
          // use atomic add instead
          // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
          atomicAdd((float*)(&popc_acc[c][global_cycles]), (sub_popcnt+nr_xnor_gates));
          // popc_acc[c][d] += sub_popcnt;
          // #if 1
          //   if (d == 0 && c == 0)
          //   {
          //     printf("sub_popcnt: %.2f, pointer1: %p\n", sub_popcnt, &popc_acc[c][d]);
          //   }
          // #endif
        }
        if (threshold_correction == 1)
        {
          comparison = (sub_popcnt >= (popc_acc[c][global_cycles]/cycles));
        }
        else
        {
          comparison = (sub_popcnt >= threshold_for_sample);
        }

        result += comparison;
        sub_popcnt = 0;
        cycle_counter = 0;
        global_cycles += 1;

        if (nr_additional_samples == 1)
        {
          comparison = (sub_popcnt_1 >= threshold_for_sample);
          result += comparison;
          sub_popcnt_1 = 0;
        }

        if (nr_additional_samples == 2)
        {
          comparison = (sub_popcnt_2_1 >= threshold_for_sample);
          result += comparison;
          sub_popcnt_2_1 = 0;

          comparison = (sub_popcnt_2_2 >= threshold_for_sample);
          result += comparison;
          sub_popcnt_2_2 = 0;
        }
      }

      // edge case
      if ((i == weight.size(1)-1)
          && ((weight.size(1) % nr_xnor_gates) != 0))
      {
        if (threshold_correction == 1)
        {
          // comparison = (sub_popcnt >= (popc_acc[c][global_cycles]/cycles));
          last_threshold_for_sample = round(((weight.size(1) % nr_xnor_gates) / nr_xnor_gates) * (popc_acc[c][global_cycles]/cycles));
          comparison = (sub_popcnt >= last_threshold_for_sample);
        }
        else
        {
          last_threshold_for_sample = round(((weight.size(1) % nr_xnor_gates) / nr_xnor_gates) * threshold[c]);
          comparison = (sub_popcnt >= last_threshold_for_sample);
        }
        result += comparison;
        // giving less weight in the majority vote for small samples
        // result += ((weight.size(1) % nr_xnor_gates) / nr_xnor_gates)*comparison;
      }
    }

    if (nr_additional_samples == 2)
    {
      // if (result <= round((cycles/2)*2*((cycles-1)/cycles)))
      if (result <= ((cycles/2) + 2*((cycles-1)/2) + majv_shift))
      {
        output[d][c] = -1;
      }
      else
      {
        output[d][c] = 1;
      }
    }

    if (nr_additional_samples == 1)
    {
      // if (result <= round((cycles/2)*2*((cycles-1)/cycles)))
      if (result <= ((cycles/2) + ((cycles-1)/2) + majv_shift))
      {
        output[d][c] = -1;
      }
      else
      {
        output[d][c] = 1;
      }
    }

    if (nr_additional_samples == 0)
    {
      if (result <= ((cycles/2) + majv_shift))
      {
        output[d][c] = -1;
      }
      else
      {
        output[d][c] = 1;
      }
    }
  }
}

torch::Tensor customconv1d_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output,
  torch::Tensor threshold,
  torch::Tensor popc_acc,
  int nr_xnor_gates,
  int nr_additional_samples,
  int majv_shift,
  int threshold_scaling,
  int popc_acc_activate,
  int threshold_correction
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
        input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        threshold.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        popc_acc.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        nr_xnor_gates,
        nr_additional_samples,
        majv_shift,
        threshold_scaling,
        popc_acc_activate,
        threshold_correction
    );
  }));

  return output;
}
