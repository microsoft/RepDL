// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace matrix_multiplication
        {
            torch::Tensor mm(const torch::Tensor &input, const torch::Tensor &mat2, bool transA,
                             bool transB)
            {
                int M = !transA ? input.size(0) : input.size(1);
                int K = !transA ? input.size(1) : input.size(0);
                int N = !transB ? mat2.size(1) : mat2.size(0);
                torch::Tensor output = torch::zeros({M, N});
                auto A = input.accessor<float, 2>();
                auto B = mat2.accessor<float, 2>();
                auto C = output.accessor<float, 2>();
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < K; k++)
                        {
                            float a = !transA ? A[i][k] : A[k][i];
                            float b = !transB ? B[k][j] : B[j][k];
                            sum = std::fmaf(a, b, sum);
                        }
                        C[i][j] = sum;
                    }
                return output;
            }
        }
    }
}

void bind_matrix_multiplication(py::module_ &m)
{
    m.def("mm", &repdl::cpu::matrix_multiplication::mm);
}
