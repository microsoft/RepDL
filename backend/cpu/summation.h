// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#pragma once

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace summation
        {
            torch::Tensor sum1d(const torch::Tensor &input, bool average = false);
            torch::Tensor sum2d_dim1(const torch::Tensor &input, bool keepdim = false,
                                     bool average = false);
        }
    }
}