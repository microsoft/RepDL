// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace summation
        {
            torch::Tensor sum1d(const torch::Tensor &input, bool average = false)
            {
                int N = input.size(0);
                float sum = 0, factor = average ? 1.0 / N : 1.0;
                auto I = input.accessor<float, 1>();
                for (int i = 0; i < N; i++)
                    sum += I[i];
                return torch::tensor(factor * sum);
            }

            torch::Tensor sum2d_dim0(const torch::Tensor &input, bool keepdim = false,
                                     bool average = false)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output = torch::empty({1, N});
                float factor = average ? 1.0 / M : 1.0;
                auto I = input.accessor<float, 2>();
                auto O = output.accessor<float, 2>();
#pragma omp parallel for
                for (int j = 0; j < N; j++)
                {
                    float sum = 0;
                    for (int i = 0; i < M; i++)
                        sum += I[i][j];
                    O[0][j] = factor * sum;
                }
                return keepdim ? output : output.squeeze();
            }

            torch::Tensor sum2d_dim1(const torch::Tensor &input, bool keepdim = false,
                                     bool average = false)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output = torch::empty({M, 1});
                float factor = average ? 1.0 / N : 1.0;
                auto I = input.accessor<float, 2>();
                auto O = output.accessor<float, 2>();
#pragma omp parallel for
                for (int i = 0; i < M; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < N; j++)
                        sum += I[i][j];
                    O[i][0] = factor * sum;
                }
                return keepdim ? output : output.squeeze();
            }
            torch::Tensor sum4d_dim023(const torch::Tensor &input, bool keepdim = false,
                                       bool average = false)
            {
                int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
                torch::Tensor output = torch::zeros({1, C, 1, 1});
                float factor = average ? 1.0 / (N * H * W) : 1.0;
                auto I = input.accessor<float, 4>();
                auto O = output.accessor<float, 4>();
#pragma omp parallel for
                for (int c = 0; c < C; c++)
                {
                    float sum = 0;
                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < H; j++)
                            for (int k = 0; k < W; k++)
                                sum += I[i][c][j][k];
                    O[0][c][0][0] = factor * sum;
                }
                return keepdim ? output : output.squeeze();
            }
        }
    }
}

void bind_summation(py::module_ &m)
{
    m.def("sum1d", &repdl::cpu::summation::sum1d);
    m.def("sum2d_dim0", &repdl::cpu::summation::sum2d_dim0);
    m.def("sum2d_dim1", &repdl::cpu::summation::sum2d_dim1);
    m.def("sum4d_dim023", &repdl::cpu::summation::sum4d_dim023);
}
