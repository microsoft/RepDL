// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cuda
    {
        namespace summation
        {
            const int CUDA_1DBLOCK_DIM = 64;

            __global__ void sum1d_thread(const float *__restrict__ input,
                                         float *__restrict__ output, int N, float factor)
            {
                float sum = 0;
                __shared__ float shared[CUDA_1DBLOCK_DIM];
                for (int i0 = 0; i0 < N; i0 += CUDA_1DBLOCK_DIM)
                {
                    int i = i0 + threadIdx.x;
                    shared[threadIdx.x] = i < N ? input[i] : 0;
                    __syncthreads();
                    _Pragma("unroll") for (int di = 0; di < CUDA_1DBLOCK_DIM; di++)
                    {
                        sum += shared[di];
                    }
                }
                if (threadIdx.x == 0)
                    *output = factor * sum;
            }

            torch::Tensor sum1d(const torch::Tensor &input, bool average = false)
            {
                int N = input.size(0);
                torch::Tensor output =
                    torch::tensor(0.0f, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_1DBLOCK_DIM);
                dim3 griddim(1);
                sum1d_thread<<<griddim, blockdim>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), N, average ? 1.0 / N : 1.0);
                return output;
            }

            __global__ void sum2d_dim0_thread(const float *__restrict__ input,
                                              float *__restrict__ output, int M, int N,
                                              float factor)
            {
                int j = blockIdx.x * CUDA_1DBLOCK_DIM + threadIdx.x;
                if (j < N)
                {
                    float sum = 0;
                    for (int i = 0; i < M; i++)
                        sum += input[i * N + j];
                    output[j] = factor * sum;
                }
            }

            torch::Tensor sum2d_dim0(const torch::Tensor &input, bool keepdim = false,
                                     bool average = false)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output =
                    torch::empty({1, N}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_1DBLOCK_DIM);
                dim3 griddim((N - 1) / CUDA_1DBLOCK_DIM + 1);
                sum2d_dim0_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                         output.data_ptr<float>(), M, N,
                                                         average ? 1.0 / M : 1.0);
                return keepdim ? output : output.squeeze();
            }

            __global__ void sum2d_dim1_thread(const float *__restrict__ input,
                                              float *__restrict__ output, int N, float factor)
            {
                int i = blockIdx.x;
                float sum = 0;
                __shared__ float shared[CUDA_1DBLOCK_DIM];
                for (int j0 = 0; j0 < N; j0 += CUDA_1DBLOCK_DIM)
                {
                    int j = j0 + threadIdx.x;
                    shared[threadIdx.x] = j < N ? input[i * N + j] : 0;
                    __syncthreads();
                    _Pragma("unroll") for (int dj = 0; dj < CUDA_1DBLOCK_DIM; dj++)
                    {
                        sum += shared[dj];
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    output[i] = factor * sum;
            }

            torch::Tensor sum2d_dim1(const torch::Tensor &input, bool keepdim = false,
                                     bool average = false)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output =
                    torch::empty({M, 1}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_1DBLOCK_DIM);
                dim3 griddim(M);
                sum2d_dim1_thread<<<griddim, blockdim>>>(
                    input.data_ptr<float>(), output.data_ptr<float>(), N, average ? 1.0 / N : 1.0);
                return keepdim ? output : output.squeeze();
            }

            __global__ void sum4d_dim023_thread(const float *__restrict__ input,
                                                float *__restrict__ output, int N, int C, int HW,
                                                float factor)
            {
                int c = blockIdx.x;
                float sum = 0;
                __shared__ float shared[CUDA_1DBLOCK_DIM];
                for (int i = 0; i < N; i++)
                {
                    for (int j0 = 0; j0 < HW; j0 += CUDA_1DBLOCK_DIM)
                    {
                        int j = j0 + threadIdx.x;
                        shared[threadIdx.x] = j < HW ? input[i * (C * HW) + c * HW + j] : 0;
                        __syncthreads();
                        _Pragma("unroll") for (int dj = 0; dj < CUDA_1DBLOCK_DIM; dj++)
                        {
                            sum += shared[dj];
                        }
                        __syncthreads();
                    }
                }
                if (threadIdx.x == 0)
                    output[c] = factor * sum;
            }

            torch::Tensor sum4d_dim023(const torch::Tensor &input, bool keepdim = false,
                                       bool average = false)
            {
                int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
                torch::Tensor output =
                    torch::empty({1, C, 1, 1}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_1DBLOCK_DIM);
                dim3 griddim(C);
                sum4d_dim023_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                           output.data_ptr<float>(), N, C, H * W,
                                                           average ? 1.0 / (N * H * W) : 1.0);
                return keepdim ? output : output.squeeze();
            }
        }
    }
}

void bind_summation(py::module_ &m)
{
    m.def("sum1d", &repdl::cuda::summation::sum1d);
    m.def("sum2d_dim0", &repdl::cuda::summation::sum2d_dim0);
    m.def("sum2d_dim1", &repdl::cuda::summation::sum2d_dim1);
    m.def("sum4d_dim023", &repdl::cuda::summation::sum4d_dim023);
}
