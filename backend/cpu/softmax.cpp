// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

#include "math.h"
#include "summation.h"

namespace repdl
{
    namespace cpu
    {
        namespace softmax
        {
            torch::Tensor softmax(const torch::Tensor &input)
            {
                torch::Tensor output = input - std::get<0>(input.max(1)).view({-1, 1});
                output = repdl::cpu::math::exp2d(output);
                output /= repdl::cpu::summation::sum2d_dim1(output, true);
                return output;
            }
        }
    }
}

void bind_softmax(py::module_ &m)
{
    m.def("softmax", &repdl::cpu::softmax::softmax);
}
