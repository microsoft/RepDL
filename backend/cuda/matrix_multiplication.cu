// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cuda
    {
        namespace matrix_multiplication
        {
            const unsigned THREAD_M = 8, THREAD_N = 8;
            const unsigned THREAD_STRIDE_M = 4, THREAD_STRIDE_N = 8;
            const unsigned WARP_M = THREAD_M * THREAD_STRIDE_M, WARP_N = THREAD_N * THREAD_STRIDE_N;
            const unsigned BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
            const unsigned CUDA_BLOCK_DIM_X = 32,
                           CUDA_BLOCK_DIM_Y = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N) / 32;

            template <unsigned DIM>
            __device__ float &get(float (*shared)[DIM], unsigned i, unsigned j)
            {
                return shared[i][j];
            }

            template <unsigned DIM>
            __device__ const float &get(const float (*shared)[DIM], unsigned i, unsigned j)
            {
                return shared[i][j];
            }

            __device__ void load_A(float (*__restrict__ A_shared)[BLOCK_M + 1],
                                   const float *__restrict__ A, unsigned i0, unsigned k0,
                                   unsigned M, unsigned K)
            {
                _Pragma("unroll") for (unsigned di = 0; di < BLOCK_M; di += CUDA_BLOCK_DIM_Y)
                {
                    _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_X)
                    {
                        unsigned i = i0 + di + threadIdx.y;
                        unsigned k = k0 + dk + threadIdx.x;
                        get(A_shared, dk + threadIdx.x, di + threadIdx.y) =
                            i < M && k < K ? A[i * K + k] : 0;
                    }
                }
            }

            __device__ void load_A_trans(float (*__restrict__ A_shared)[BLOCK_M],
                                         const float *__restrict__ A, unsigned i0, unsigned k0,
                                         unsigned M, unsigned K)
            {
                _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_Y)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < BLOCK_M; di += CUDA_BLOCK_DIM_X)
                    {
                        unsigned k = k0 + dk + threadIdx.y;
                        unsigned i = i0 + di + threadIdx.x;
                        get(A_shared, dk + threadIdx.y, di + threadIdx.x) =
                            i < M && k < K ? A[k * M + i] : 0;
                    }
                }
            }

            __device__ void load_B(float (*__restrict__ B_shared)[BLOCK_N],
                                   const float *__restrict__ B, unsigned j0, unsigned k0,
                                   unsigned N, unsigned K)
            {
                _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_Y)
                {
                    _Pragma("unroll") for (unsigned dj = 0; dj < BLOCK_N; dj += CUDA_BLOCK_DIM_X)
                    {
                        unsigned k = k0 + dk + threadIdx.y;
                        unsigned j = j0 + dj + threadIdx.x;
                        get(B_shared, dk + threadIdx.y, dj + threadIdx.x) =
                            k < K && j < N ? B[k * N + j] : 0;
                    }
                }
            }

            __device__ void load_B_trans(float (*__restrict__ B_shared)[BLOCK_N + 1],
                                         const float *__restrict__ B, unsigned j0, unsigned k0,
                                         unsigned N, unsigned K)
            {
                _Pragma("unroll") for (unsigned dj = 0; dj < BLOCK_N; dj += CUDA_BLOCK_DIM_Y)
                {
                    _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_X)
                    {
                        unsigned j = j0 + dj + threadIdx.y;
                        unsigned k = k0 + dk + threadIdx.x;
                        get(B_shared, dk + threadIdx.x, dj + threadIdx.y) =
                            k < K && j < N ? B[j * K + k] : 0;
                    }
                }
            }

            __device__ void store_C(float *__restrict__ C,
                                    const float (*__restrict__ sum)[THREAD_N], unsigned i0,
                                    unsigned j0, unsigned wi0, unsigned wj0, unsigned M, unsigned N)
            {
                unsigned thread_id_in_warp = threadIdx.x;
                _Pragma("unroll") for (unsigned di = 0; di < THREAD_M; di++)
                {
                    _Pragma("unroll") for (unsigned dj = 0; dj < THREAD_N; dj++)
                    {
                        unsigned i =
                            i0 + wi0 + di * THREAD_STRIDE_M + thread_id_in_warp / THREAD_STRIDE_N;
                        unsigned j =
                            j0 + wj0 + dj * THREAD_STRIDE_N + thread_id_in_warp % THREAD_STRIDE_N;
                        if (i < M && j < N)
                            C[i * N + j] = sum[di][dj];
                    }
                }
            }

            template <unsigned D_A_SHARED, unsigned D_B_SHARED>
            __device__ void compute_block(const float (*__restrict__ A_shared)[D_A_SHARED],
                                          const float (*__restrict__ B_shared)[D_B_SHARED],
                                          float *__restrict__ a, float *__restrict__ b,
                                          float (*__restrict__ sum)[THREAD_N], unsigned wi0,
                                          unsigned wj0)
            {
                unsigned thread_id_in_warp = threadIdx.x;
                _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk++)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < THREAD_M; di++)
                    {
                        a[di] =
                            get(A_shared, dk,
                                wi0 + di * THREAD_STRIDE_M + thread_id_in_warp / THREAD_STRIDE_N);
                    }
                    _Pragma("unroll") for (unsigned dj = 0; dj < THREAD_N; dj++)
                    {
                        b[dj] =
                            get(B_shared, dk,
                                wj0 + dj * THREAD_STRIDE_N + thread_id_in_warp % THREAD_STRIDE_N);
                    }
                    _Pragma("unroll") for (unsigned di = 0; di < THREAD_M; di++)
                    {
                        _Pragma("unroll") for (unsigned dj = 0; dj < THREAD_N; dj++)
                        {
                            sum[di][dj] += a[di] * b[dj];
                        }
                    }
                }
            }

            __global__ void mm_thread(const float *__restrict__ A, const float *__restrict__ B,
                                      float *__restrict__ C, unsigned M, unsigned K, unsigned N)
            {
                unsigned i0 = blockIdx.y * BLOCK_M;
                unsigned j0 = blockIdx.x * BLOCK_N;
                unsigned warp_id = threadIdx.y;
                unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                __shared__ float A_shared[BLOCK_K][BLOCK_M + 1], B_shared[BLOCK_K][BLOCK_N];
                float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                for (unsigned k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    load_A(A_shared, A, i0, k0, M, K);
                    load_B(B_shared, B, j0, k0, N, K);
                    __syncthreads();
                    compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                    __syncthreads();
                }
                store_C(C, sum, i0, j0, wi0, wj0, M, N);
            }

            __global__ void mm_transA_thread(const float *__restrict__ A,
                                             const float *__restrict__ B, float *__restrict__ C,
                                             unsigned M, unsigned K, unsigned N)
            {
                unsigned i0 = blockIdx.y * BLOCK_M;
                unsigned j0 = blockIdx.x * BLOCK_N;
                unsigned warp_id = threadIdx.y;
                unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                __shared__ float A_shared[BLOCK_K][BLOCK_M], B_shared[BLOCK_K][BLOCK_N];
                float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                for (unsigned k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    load_A_trans(A_shared, A, i0, k0, M, K);
                    load_B(B_shared, B, j0, k0, N, K);
                    __syncthreads();
                    compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                    __syncthreads();
                }
                store_C(C, sum, i0, j0, wi0, wj0, M, N);
            }

            __global__ void mm_transB_thread(const float *__restrict__ A,
                                             const float *__restrict__ B, float *__restrict__ C,
                                             unsigned M, unsigned K, unsigned N)
            {
                unsigned i0 = blockIdx.y * BLOCK_M;
                unsigned j0 = blockIdx.x * BLOCK_N;
                unsigned warp_id = threadIdx.y;
                unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                __shared__ float A_shared[BLOCK_K][BLOCK_M + 1], B_shared[BLOCK_K][BLOCK_N + 1];
                float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                for (unsigned k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    load_A(A_shared, A, i0, k0, M, K);
                    load_B_trans(B_shared, B, j0, k0, N, K);
                    __syncthreads();
                    compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                    __syncthreads();
                }
                store_C(C, sum, i0, j0, wi0, wj0, M, N);
            }

            __global__ void mm_transA_transB_thread(const float *__restrict__ A,
                                                    const float *__restrict__ B,
                                                    float *__restrict__ C, unsigned M, unsigned K,
                                                    unsigned N)
            {
                unsigned i0 = blockIdx.y * BLOCK_M;
                unsigned j0 = blockIdx.x * BLOCK_N;
                unsigned warp_id = threadIdx.y;
                unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                __shared__ float A_shared[BLOCK_K][BLOCK_M], B_shared[BLOCK_K][BLOCK_N + 1];
                float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                for (unsigned k0 = 0; k0 < K; k0 += BLOCK_K)
                {
                    load_A_trans(A_shared, A, i0, k0, M, K);
                    load_B_trans(B_shared, B, j0, k0, N, K);
                    __syncthreads();
                    compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                    __syncthreads();
                }
                store_C(C, sum, i0, j0, wi0, wj0, M, N);
            }

            torch::Tensor mm(const torch::Tensor &input, const torch::Tensor &mat2, bool transA,
                             bool transB)
            {
                unsigned M = !transA ? input.size(0) : input.size(1);
                unsigned K = !transA ? input.size(1) : input.size(0);
                unsigned N = !transB ? mat2.size(1) : mat2.size(0);
                auto output = torch::empty({M, N}, torch::TensorOptions().device(torch::kCUDA));
                dim3 blockdim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
                dim3 griddim((N - 1) / BLOCK_N + 1, (M - 1) / BLOCK_M + 1);
                if (!transA && !transB)
                    mm_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                     mat2.data_ptr<float>(),
                                                     output.data_ptr<float>(), M, K, N);
                else if (!transA && transB)
                    mm_transB_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                            mat2.data_ptr<float>(),
                                                            output.data_ptr<float>(), M, K, N);
                else if (transA && !transB)
                    mm_transA_thread<<<griddim, blockdim>>>(input.data_ptr<float>(),
                                                            mat2.data_ptr<float>(),
                                                            output.data_ptr<float>(), M, K, N);
                else
                    mm_transA_transB_thread<<<griddim, blockdim>>>(
                        input.data_ptr<float>(), mat2.data_ptr<float>(), output.data_ptr<float>(),
                        M, K, N);
                return output;
            }
        }
    }
}

void bind_matrix_multiplication(py::module_ &m)
{
    m.def("mm", &repdl::cuda::matrix_multiplication::mm);
}
