// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cuda
    {
        namespace math
        {
            const int CUDA_1DBLOCK_DIM = 64;
            const int CUDA_2DBLOCK_DIM_X = 32, CUDA_2DBLOCK_DIM_Y = 16;

            __device__ float correct_rounded_exp(float x)
            {
                return ::exp((double)x);
            }

            __global__ void exp2d_thread(const float *input, float *output, int M, int N)
            {
                int i = blockIdx.y * CUDA_2DBLOCK_DIM_Y + threadIdx.y;
                int j = blockIdx.x * CUDA_2DBLOCK_DIM_X + threadIdx.x;
                if (i < M && j < N)
                {
                    output[i * N + j] = correct_rounded_exp(input[i * N + j]);
                }
            }

            torch::Tensor exp2d(const torch::Tensor &input)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output =
                    torch::empty({M, N}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_2DBLOCK_DIM_X, CUDA_2DBLOCK_DIM_Y);
                dim3 griddim((N - 1) / CUDA_2DBLOCK_DIM_X + 1, (M - 1) / CUDA_2DBLOCK_DIM_Y + 1);
                exp2d_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                    output.data_ptr<float>(), M, N);
                return output;
            }

            __device__ float correct_rounded_log(float x)
            {
                return ::log((double)x);
            }

            __global__ void log1d_thread(const float *input, float *output, int N)
            {
                int i = blockIdx.x * CUDA_1DBLOCK_DIM + threadIdx.x;
                if (i < N)
                {
                    output[i] = correct_rounded_log(input[i]);
                }
            }

            torch::Tensor log1d(const torch::Tensor &input)
            {
                int N = input.size(0);
                torch::Tensor output =
                    torch::empty({N}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_1DBLOCK_DIM);
                dim3 griddim((N - 1) / CUDA_1DBLOCK_DIM + 1);
                log1d_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                    output.data_ptr<float>(), N);
                return output;
            }
        }
    }
}