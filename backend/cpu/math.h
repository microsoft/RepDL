// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#pragma once

#include <torch/extension.h>

namespace repdl
{
    namespace cpu
    {
        namespace math
        {
            torch::Tensor exp2d(const torch::Tensor &input);
            torch::Tensor log1d(const torch::Tensor &input);
        }
    }
}