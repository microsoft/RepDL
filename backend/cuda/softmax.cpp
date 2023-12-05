// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

#include "math.h"
#include "summation.h"

namespace repdl
{
    namespace cuda
    {
        namespace softmax
        {
            torch::Tensor softmax(const torch::Tensor &input)
            {
                torch::Tensor output = input - std::get<0>(input.max(1)).view({-1, 1});
                output = repdl::cuda::math::exp2d(output);
                output /= repdl::cuda::summation::sum2d_dim1(output, true);
                return output;
            }
        }
    }
}

void bind_softmax(py::module_ &m)
{
    m.def("softmax", &repdl::cuda::softmax::softmax);
}
