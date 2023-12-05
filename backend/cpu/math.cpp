// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <cmath>

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace math
        {
            inline float correct_rounded_exp(float x)
            {
                return std::exp((double)x);
            }

            torch::Tensor exp2d(const torch::Tensor &input)
            {
                int M = input.size(0), N = input.size(1);
                torch::Tensor output = torch::empty({M, N});
                auto I = input.accessor<float, 2>();
                auto O = output.accessor<float, 2>();
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        O[i][j] = correct_rounded_exp(I[i][j]);
                return output;
            }

            inline float correct_rounded_log(float x)
            {
                return std::log((double)x);
            }

            torch::Tensor log1d(const torch::Tensor &input)
            {
                int N = input.size(0);
                torch::Tensor output = torch::empty({N});
                auto I = input.accessor<float, 1>();
                auto O = output.accessor<float, 1>();
#pragma omp parallel for
                for (int i = 0; i < N; i++)
                    O[i] = correct_rounded_log(I[i]);
                return output;
            }
        }
    }
}