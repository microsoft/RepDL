// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace convolution
        {
            torch::Tensor conv2d(const torch::Tensor &input, const torch::Tensor &kernel,
                                 int strideH, int strideW, int paddingH, int paddingW, int groups)
            {
                int N = input.size(0), IC = input.size(1), H = input.size(2), W = input.size(3);
                int OC = kernel.size(0), KH = kernel.size(2), KW = kernel.size(3);
                int OH = (H + 2 * paddingH - KH) / strideH + 1,
                    OW = (W + 2 * paddingW - KW) / strideW + 1;
                int IC_per_group = IC / groups, OC_per_group = OC / groups;
                torch::Tensor output = torch::zeros({N, OC, OH, OW});
                auto I = input.accessor<float, 4>();
                auto K = kernel.accessor<float, 4>();
                auto O = output.accessor<float, 4>();
#pragma omp parallel for collapse(4)
                for (int n = 0; n < N; n++)
                    for (int oc = 0; oc < OC; oc++)
                        for (int oh = 0; oh < OH; oh++)
                            for (int ow = 0; ow < OW; ow++)
                            {
                                float sum = 0;
                                int group = oc / OC_per_group;
                                for (int i = 0; i < IC_per_group; i++)
                                {
                                    int ic = group * IC_per_group + i;
                                    for (int kh = 0; kh < KH; kh++)
                                    {
                                        int h = oh * strideH - paddingH + kh;
                                        if (h < 0 || h >= H)
                                            continue;
                                        for (int kw = 0; kw < KW; kw++)
                                        {
                                            int w = ow * strideW - paddingW + kw;
                                            if (w < 0 || w >= W)
                                                continue;
                                            sum = std::fmaf(I[n][ic][h][w], K[oc][i][kh][kw], sum);
                                        }
                                    }
                                }
                                O[n][oc][oh][ow] = sum;
                            }
                return output;
            }

            torch::Tensor conv2d_grad_input(const torch::Tensor &grad_output,
                                            const torch::Tensor &kernel, int H, int W, int strideH,
                                            int strideW, int paddingH, int paddingW)
            {
                int OC = kernel.size(0), IC = kernel.size(1), KH = kernel.size(2),
                    KW = kernel.size(3);
                int N = grad_output.size(0), OH = grad_output.size(2), OW = grad_output.size(3);
                torch::Tensor grad_input = torch::zeros({N, IC, H, W});
                auto GO = grad_output.accessor<float, 4>();
                auto K = kernel.accessor<float, 4>();
                auto GI = grad_input.accessor<float, 4>();
#pragma omp parallel for collapse(4)
                for (int n = 0; n < N; n++)
                    for (int ic = 0; ic < IC; ic++)
                        for (int h = 0; h < H; h++)
                            for (int w = 0; w < W; w++)
                            {
                                float sum = 0;
                                for (int oc = 0; oc < OC; oc++)
                                    for (int kh = 0; kh < KH; kh++)
                                    {
                                        int oh = (h - kh + paddingH) / strideH;
                                        if (oh < 0 || oh >= OH || (h - kh + paddingH) % strideH)
                                            continue;
                                        for (int kw = 0; kw < KW; kw++)
                                        {
                                            int ow = (w - kw + paddingW) / strideW;
                                            if (ow < 0 || ow >= OW || (w - kw + paddingW) % strideW)
                                                continue;
                                            sum = std::fmaf(GO[n][oc][oh][ow], K[oc][ic][kh][kw],
                                                            sum);
                                        }
                                    }
                                GI[n][ic][h][w] = sum;
                            }
                return grad_input;
            }

            torch::Tensor conv2d_grad_kernel(const torch::Tensor &grad_output,
                                             const torch::Tensor &input, int KH, int KW,
                                             int strideH, int strideW, int paddingH, int paddingW)
            {
                int N = input.size(0), IC = input.size(1), H = input.size(2), W = input.size(3);
                int OC = grad_output.size(1), OH = grad_output.size(2), OW = grad_output.size(3);
                torch::Tensor grad_kernel = torch::zeros({OC, IC, KH, KW});
                auto GO = grad_output.accessor<float, 4>();
                auto I = input.accessor<float, 4>();
                auto GK = grad_kernel.accessor<float, 4>();
#pragma omp parallel for collapse(4)
                for (int oc = 0; oc < OC; oc++)
                    for (int ic = 0; ic < IC; ic++)
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                            {
                                float sum = 0;
                                for (int n = 0; n < N; n++)
                                {
                                    for (int oh = 0; oh < OH; oh++)
                                    {
                                        int h = oh * strideH - paddingH + kh;
                                        if (h < 0 || h >= H)
                                            continue;
                                        for (int ow = 0; ow < OW; ow++)
                                        {
                                            int w = ow * strideW - paddingW + kw;
                                            if (w < 0 || w >= W)
                                                continue;
                                            sum = std::fmaf(GO[n][oc][oh][ow], I[n][ic][h][w], sum);
                                        }
                                    }
                                }
                                GK[oc][ic][kh][kw] = sum;
                            }
                return grad_kernel;
            }

        }
    }
}

void bind_convolution(py::module_ &m)
{
    m.def("conv2d", &repdl::cpu::convolution::conv2d);
    m.def("conv2d_grad_input", &repdl::cpu::convolution::conv2d_grad_input);
    m.def("conv2d_grad_kernel", &repdl::cpu::convolution::conv2d_grad_kernel);
}
