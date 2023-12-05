// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <cstdio>
#include <torch/extension.h>
namespace repdl
{
    namespace cuda
    {
        namespace convolution
        {
            const unsigned THREAD_M = 4, THREAD_N = 4;
            const unsigned THREAD_STRIDE_M = 8, THREAD_STRIDE_N = 4;
            const unsigned WARP_M = THREAD_M * THREAD_STRIDE_M, WARP_N = THREAD_N * THREAD_STRIDE_N;
            const unsigned BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32;
            const unsigned CUDA_BLOCK_DIM_X = 32,
                           CUDA_BLOCK_DIM_Y = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N) / 32;

            __device__ void compute_block(const float (*__restrict__ A_shared)[BLOCK_M + 1],
                                          const float (*__restrict__ B_shared)[BLOCK_N],
                                          float *__restrict__ a, float *__restrict__ b,
                                          float (*__restrict__ sum)[THREAD_N], unsigned wi0,
                                          unsigned wj0)
            {
                unsigned thread_id_in_warp = threadIdx.x;
                _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk++)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < THREAD_M; di++)
                    {
                        a[di] = A_shared[dk][wi0 + di * THREAD_STRIDE_M +
                                             thread_id_in_warp / THREAD_STRIDE_N];
                    }
                    _Pragma("unroll") for (unsigned dj = 0; dj < THREAD_N; dj++)
                    {
                        b[dj] = B_shared[dk][wj0 + dj * THREAD_STRIDE_N +
                                             thread_id_in_warp % THREAD_STRIDE_N];
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

            __device__ void store_C(float *__restrict__ output,
                                    const float (*__restrict__ sum)[THREAD_N], unsigned i0,
                                    unsigned j0, unsigned wi0, unsigned wj0, unsigned dimM,
                                    unsigned dimN)
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
                        if (i < dimM && j < dimN)
                            output[i * dimN + j] = sum[di][dj];
                    }
                }
            }

            namespace conv2d
            {
                __device__ void load_A(float (*__restrict__ A_shared)[BLOCK_M + 1],
                                       const float *__restrict__ kernel, unsigned i0, unsigned k0,
                                       unsigned dimM, unsigned dimK, unsigned group_id,
                                       unsigned OC_per_group)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < BLOCK_M; di += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K;
                                               dk += CUDA_BLOCK_DIM_X)
                        {
                            unsigned i = i0 + di + threadIdx.y;
                            unsigned k = k0 + dk + threadIdx.x;
                            // [i] = [oc_in_group], [k] = [ic_in_group][kh][kw]
                            if (i < dimM && k < dimK)
                                A_shared[dk + threadIdx.x][di + threadIdx.y] =
                                    kernel[(group_id * OC_per_group + i) * dimK + k];
                            else
                                A_shared[dk + threadIdx.x][di + threadIdx.y] = 0;
                        }
                    }
                }

                __device__ void load_B(float (*__restrict__ B_shared)[BLOCK_N],
                                       const float *__restrict__ input, unsigned j0, unsigned k0,
                                       unsigned dimK, unsigned dimN, unsigned OW, unsigned KH,
                                       unsigned KW, unsigned H, unsigned W, unsigned strideH,
                                       unsigned strideW, unsigned paddingH, unsigned paddingW,
                                       unsigned group_id, unsigned IC_per_group)
                {
                    _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dj = 0; dj < BLOCK_N;
                                               dj += CUDA_BLOCK_DIM_X)
                        {
                            unsigned k = k0 + dk + threadIdx.y;
                            unsigned j = j0 + dj + threadIdx.x;
                            // [k] = [ic_in_group][kh][kw], [j] = [oh][ow]
                            unsigned oh = j / OW, ow = j % OW;
                            unsigned ic_in_group = k / (KH * KW), kh = k % (KH * KW) / KW,
                                     kw = k % KW;
                            int h = oh * strideH + kh - (int)paddingH,
                                w = ow * strideW + kw - (int)paddingW;
                            if (k < dimK && j < dimN && h >= 0 && h < H && w >= 0 && w < W)
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] =
                                    input[ic_in_group * (H * W) + h * W + w];
                            else
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] = 0;
                        }
                    }
                }

                __global__ void conv2d_thread(const float *__restrict__ input,
                                              const float *__restrict__ kernel,
                                              float *__restrict__ output, unsigned strideH,
                                              unsigned strideW, unsigned paddingH,
                                              unsigned paddingW, unsigned IC, unsigned H,
                                              unsigned W, unsigned OC, unsigned OH, unsigned OW,
                                              unsigned KH, unsigned KW, unsigned groups,
                                              unsigned IC_per_group, unsigned OC_per_group)
                {
                    unsigned dimM = OC_per_group, dimN = OH * OW, dimK = IC_per_group * KH * KW;
                    unsigned batch_id = blockIdx.z / groups, group_id = blockIdx.z % groups;
                    unsigned i0 = blockIdx.y * BLOCK_M;
                    unsigned j0 = blockIdx.x * BLOCK_N;
                    unsigned warp_id = threadIdx.y;
                    unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                    unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                    __shared__ float A_shared[BLOCK_K][BLOCK_M + 1], B_shared[BLOCK_K][BLOCK_N];
                    float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                    for (unsigned k0 = 0; k0 < dimK; k0 += BLOCK_K)
                    {
                        load_A(A_shared, kernel, i0, k0, dimM, dimK, group_id, OC_per_group);
                        load_B(B_shared, input + blockIdx.z * (IC_per_group * H * W), j0, k0, dimK,
                               dimN, OW, KH, KW, H, W, strideH, strideW, paddingH, paddingW,
                               group_id, IC_per_group);
                        __syncthreads();
                        compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                        __syncthreads();
                    }
                    store_C(output + blockIdx.z * (OC_per_group * OH * OW), sum, i0, j0, wi0, wj0,
                            dimM, dimN);
                }

                torch::Tensor conv2d(const torch::Tensor &input, const torch::Tensor &kernel,
                                     unsigned strideH, unsigned strideW, unsigned paddingH,
                                     unsigned paddingW, unsigned groups)
                {
                    unsigned N = input.size(0), IC = input.size(1), H = input.size(2),
                             W = input.size(3);
                    unsigned OC = kernel.size(0), KH = kernel.size(2), KW = kernel.size(3);
                    unsigned OH = (H + 2 * paddingH - KH) / strideH + 1,
                             OW = (W + 2 * paddingW - KW) / strideW + 1;
                    unsigned IC_per_group = IC / groups, OC_per_group = OC / groups;
                    torch::Tensor output =
                        torch::empty({N, OC, OH, OW}, torch::TensorOptions().device(torch::kCUDA));
                    dim3 blockdim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
                    dim3 griddim((OH * OW - 1) / BLOCK_N + 1, (OC_per_group - 1) / BLOCK_M + 1,
                                 N * groups);
                    conv2d_thread<<<griddim, blockdim>>>(
                        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
                        strideH, strideW, paddingH, paddingW, IC, H, W, OC, OH, OW, KH, KW, groups,
                        IC_per_group, OC_per_group);
                    return output;
                }
            }

            namespace conv2d_grad_input
            {
                __device__ void load_A(float (*__restrict__ A_shared)[BLOCK_M + 1],
                                       const float *__restrict__ kernel, unsigned i0, unsigned k0,
                                       unsigned dimM, unsigned dimK, unsigned KHKW, unsigned IC)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < BLOCK_M; di += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K;
                                               dk += CUDA_BLOCK_DIM_X)
                        {
                            unsigned i = i0 + di + threadIdx.y;
                            unsigned k = k0 + dk + threadIdx.x;
                            // [i] = [ic], [k] = [oc][kh][kw]
                            unsigned oc = k / KHKW, khkw = k % KHKW;
                            if (i < dimM && k < dimK)
                                A_shared[dk + threadIdx.x][di + threadIdx.y] =
                                    kernel[oc * IC * KHKW + i * KHKW + khkw];
                            else
                                A_shared[dk + threadIdx.x][di + threadIdx.y] = 0;
                        }
                    }
                }

                __device__ void load_B(float (*__restrict__ B_shared)[BLOCK_N],
                                       const float *__restrict__ grad_output, unsigned j0,
                                       unsigned k0, unsigned dimK, unsigned dimN, unsigned OH,
                                       unsigned OW, unsigned KH, unsigned KW, unsigned W,
                                       unsigned strideH, unsigned strideW, unsigned paddingH,
                                       unsigned paddingW)
                {
                    _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dj = 0; dj < BLOCK_N;
                                               dj += CUDA_BLOCK_DIM_X)
                        {
                            unsigned k = k0 + dk + threadIdx.y;
                            unsigned j = j0 + dj + threadIdx.x;
                            // [k] = [oc][kh][kw], [j] = [h][w]
                            unsigned oc = k / (KH * KW), kh = k % (KH * KW) / KW, kw = k % KW;
                            unsigned h = j / W, w = j % W;
                            int oh = (h - (int)kh + paddingH) / strideH,
                                ohr = (h - (int)kh + paddingH) % strideH;
                            int ow = (w - (int)kw + paddingW) / strideW,
                                owr = (w - (int)kw + paddingW) % strideW;
                            if (k < dimK && j < dimN && oh >= 0 && oh < OH && ow >= 0 && ow < OW &&
                                !ohr && !owr)
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] =
                                    grad_output[oc * (OH * OW) + oh * OW + ow];
                            else
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] = 0;
                        }
                    }
                }

                __global__ void conv2d_grad_input_thread(
                    const float *__restrict__ grad_output, const float *__restrict__ kernel,
                    float *__restrict__ grad_input, unsigned strideH, unsigned strideW,
                    unsigned paddingH, unsigned paddingW, unsigned IC, unsigned H, unsigned W,
                    unsigned OC, unsigned OH, unsigned OW, unsigned KH, unsigned KW)
                {
                    unsigned dimM = IC, dimN = H * W, dimK = OC * KH * KW;
                    unsigned batch_id = blockIdx.z;
                    unsigned i0 = blockIdx.y * BLOCK_M;
                    unsigned j0 = blockIdx.x * BLOCK_N;
                    unsigned warp_id = threadIdx.y;
                    unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                    unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                    __shared__ float A_shared[BLOCK_K][BLOCK_M + 1], B_shared[BLOCK_K][BLOCK_N];
                    float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                    for (unsigned k0 = 0; k0 < dimK; k0 += BLOCK_K)
                    {
                        load_A(A_shared, kernel, i0, k0, dimM, dimK, KH * KW, IC);
                        load_B(B_shared, grad_output + batch_id * (OC * OH * OW), j0, k0, dimK,
                               dimN, OH, OW, KH, KW, W, strideH, strideW, paddingH, paddingW);
                        __syncthreads();
                        compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                        __syncthreads();
                    }
                    store_C(grad_input + batch_id * (IC * H * W), sum, i0, j0, wi0, wj0, dimM,
                            dimN);
                }

                torch::Tensor conv2d_grad_input(const torch::Tensor &grad_output,
                                                const torch::Tensor &kernel, unsigned H, unsigned W,
                                                unsigned strideH, unsigned strideW,
                                                unsigned paddingH, unsigned paddingW)
                {
                    unsigned OC = kernel.size(0), IC = kernel.size(1), KH = kernel.size(2),
                             KW = kernel.size(3);
                    unsigned N = grad_output.size(0), OH = grad_output.size(2),
                             OW = grad_output.size(3);
                    torch::Tensor grad_input =
                        torch::empty({N, IC, H, W}, torch::TensorOptions().device(torch::kCUDA));
                    dim3 blockdim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
                    dim3 griddim((H * W - 1) / BLOCK_N + 1, (IC - 1) / BLOCK_M + 1, N);
                    conv2d_grad_input_thread<<<griddim, blockdim>>>(
                        grad_output.data_ptr<float>(), kernel.data_ptr<float>(),
                        grad_input.data_ptr<float>(), strideH, strideW, paddingH, paddingW, IC, H,
                        W, OC, OH, OW, KH, KW);
                    return grad_input;
                }
            }

            namespace conv2d_grad_kernel
            {
                __device__ void load_A(float (*__restrict__ A_shared)[BLOCK_M + 1],
                                       const float *__restrict__ grad_output, unsigned i0,
                                       unsigned k0, unsigned dimM, unsigned dimK, unsigned OHOW,
                                       unsigned OC)
                {
                    _Pragma("unroll") for (unsigned di = 0; di < BLOCK_M; di += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K;
                                               dk += CUDA_BLOCK_DIM_X)
                        {
                            unsigned i = i0 + di + threadIdx.y;
                            unsigned k = k0 + dk + threadIdx.x;
                            // [i] = [oc], [k] = [n][oh][ow]
                            unsigned n = k / OHOW, ohow = k % OHOW;
                            if (i < dimM && k < dimK)
                                A_shared[dk + threadIdx.x][di + threadIdx.y] =
                                    grad_output[n * OC * OHOW + i * OHOW + ohow];
                            else
                                A_shared[dk + threadIdx.x][di + threadIdx.y] = 0;
                        }
                    }
                }

                __device__ void load_B(float (*__restrict__ B_shared)[BLOCK_N],
                                       const float *__restrict__ input, unsigned j0, unsigned k0,
                                       unsigned dimK, unsigned dimN, unsigned OH, unsigned OW,
                                       unsigned KH, unsigned KW, unsigned IC, unsigned H,
                                       unsigned W, unsigned strideH, unsigned strideW,
                                       unsigned paddingH, unsigned paddingW)
                {
                    _Pragma("unroll") for (unsigned dk = 0; dk < BLOCK_K; dk += CUDA_BLOCK_DIM_Y)
                    {
                        _Pragma("unroll") for (unsigned dj = 0; dj < BLOCK_N;
                                               dj += CUDA_BLOCK_DIM_X)
                        {
                            unsigned k = k0 + dk + threadIdx.y;
                            unsigned j = j0 + dj + threadIdx.x;
                            // [k] = [n][oh][ow], [j] = [ic][kh][kw]
                            unsigned n = k / (OH * OW), oh = k % (OH * OW) / OW, ow = k % OW;
                            unsigned ic = j / (KH * KW), kh = j % (KH * KW) / KW, kw = j % KW;
                            int h = oh * strideH + kh - (int)paddingH;
                            int w = ow * strideW + kw - (int)paddingW;
                            if (k < dimK && j < dimN && h >= 0 && h < H && w >= 0 && w < W)
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] =
                                    input[n * (IC * H * W) + ic * H * W + h * W + w];
                            else
                                B_shared[dk + threadIdx.y][dj + threadIdx.x] = 0;
                        }
                    }
                }

                __global__ void conv2d_grad_kernel_thread(
                    const float *__restrict__ grad_output, const float *__restrict__ input,
                    float *__restrict__ grad_kernel, unsigned strideH, unsigned strideW,
                    unsigned paddingH, unsigned paddingW, unsigned BatchN, unsigned IC, unsigned H,
                    unsigned W, unsigned OC, unsigned OH, unsigned OW, unsigned KH, unsigned KW)
                {
                    unsigned dimM = OC, dimN = IC * KH * KW, dimK = BatchN * OH * OW;
                    unsigned i0 = blockIdx.y * BLOCK_M;
                    unsigned j0 = blockIdx.x * BLOCK_N;
                    unsigned warp_id = threadIdx.y;
                    unsigned wi0 = warp_id / (BLOCK_N / WARP_N) * WARP_M;
                    unsigned wj0 = warp_id % (BLOCK_N / WARP_N) * WARP_N;
                    __shared__ float A_shared[BLOCK_K][BLOCK_M + 1], B_shared[BLOCK_K][BLOCK_N];
                    float sum[THREAD_M][THREAD_N] = {0}, a[THREAD_M], b[THREAD_N];
                    for (unsigned k0 = 0; k0 < dimK; k0 += BLOCK_K)
                    {
                        load_A(A_shared, grad_output, i0, k0, dimM, dimK, OH * OW, OC);
                        load_B(B_shared, input, j0, k0, dimK, dimN, OH, OW, KH, KW, IC, H, W,
                               strideH, strideW, paddingH, paddingW);
                        __syncthreads();
                        compute_block(A_shared, B_shared, a, b, sum, wi0, wj0);
                        __syncthreads();
                    }
                    store_C(grad_kernel, sum, i0, j0, wi0, wj0, dimM, dimN);
                }

                torch::Tensor conv2d_grad_kernel(const torch::Tensor &grad_output,
                                                 const torch::Tensor &input, unsigned KH,
                                                 unsigned KW, unsigned strideH, unsigned strideW,
                                                 unsigned paddingH, unsigned paddingW)
                {
                    unsigned BatchN = input.size(0), IC = input.size(1), H = input.size(2),
                             W = input.size(3);
                    unsigned OC = grad_output.size(1), OH = grad_output.size(2),
                             OW = grad_output.size(3);
                    torch::Tensor grad_kernel =
                        torch::empty({OC, IC, KH, KW}, torch::TensorOptions().device(torch::kCUDA));
                    dim3 blockdim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
                    dim3 griddim((IC * KH * KW - 1) / BLOCK_N + 1, (OC - 1) / BLOCK_M + 1);
                    conv2d_grad_kernel_thread<<<griddim, blockdim>>>(
                        grad_output.data_ptr<float>(), input.data_ptr<float>(),
                        grad_kernel.data_ptr<float>(), strideH, strideW, paddingH, paddingW, BatchN,
                        IC, H, W, OC, OH, OW, KH, KW);
                    return grad_kernel;
                }
            }
        }
    }
}

void bind_convolution(py::module_ &m)
{
    m.def("conv2d", &repdl::cuda::convolution::conv2d::conv2d);
    m.def("conv2d_grad_input", &repdl::cuda::convolution::conv2d_grad_input::conv2d_grad_input);
    m.def("conv2d_grad_kernel", &repdl::cuda::convolution::conv2d_grad_kernel::conv2d_grad_kernel);
}
