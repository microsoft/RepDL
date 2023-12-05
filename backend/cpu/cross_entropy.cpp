// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

#include "math.h"
#include "summation.h"

namespace repdl
{
    namespace cpu
    {
        namespace cross_entropy
        {
            torch::Tensor cross_entropy(const torch::Tensor &input, const torch::Tensor &target)
            {
                torch::Tensor input_shifted = input - std::get<0>(input.max(1, true));
                torch::Tensor loss = repdl::cpu::math::exp2d(input_shifted);
                loss = repdl::cpu::summation::sum2d_dim1(loss);
                loss = repdl::cpu::math::log1d(loss);
                loss -= input_shifted.gather(1, target.unsqueeze(-1)).squeeze();
                loss = repdl::cpu::summation::sum1d(loss, true /*average*/);
                return loss;
            }
        }
    }
}

void bind_cross_entropy(py::module_ &m)
{
    m.def("cross_entropy", &repdl::cpu::cross_entropy::cross_entropy);
}
