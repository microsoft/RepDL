// Copyright (c) Microsoft Corporation. 
// Licensed under the MIT license.

#include <torch/extension.h>

void bind_convolution(py::module_ &m);
void bind_cross_entropy(py::module_ &m);
void bind_matrix_multiplication(py::module_ &m);
void bind_softmax(py::module_ &m);
void bind_summation(py::module_ &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    bind_convolution(m);
    bind_cross_entropy(m);
    bind_matrix_multiplication(m);
    bind_softmax(m);
    bind_summation(m);
}