# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch

from repdl.backend import cpu
from repdl.backend import cuda


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.conv2d(
            input, weight, stride[0], stride[1], padding[0], padding[1], groups
        )
    else:
        return cpu.conv2d(
            input, weight, stride[0], stride[1], padding[0], padding[1], groups
        )


def conv2d_grad_input(
    input_shape: tuple[int, ...],
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    if grad_output.is_cuda:
        return cuda.conv2d_grad_input(
            grad_output,
            weight,
            input_shape[2],
            input_shape[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
        )
    else:
        return cpu.conv2d_grad_input(
            grad_output,
            weight,
            input_shape[2],
            input_shape[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
        )


def conv2d_grad_kernel(
    input: torch.Tensor,
    weight_shape: tuple[int, ...],
    grad_output: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.conv2d_grad_kernel(
            grad_output,
            input,
            weight_shape[2],
            weight_shape[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
        )
    else:
        return cpu.conv2d_grad_kernel(
            grad_output,
            input,
            weight_shape[2],
            weight_shape[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
        )


def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if input.is_cuda:
        return cuda.cross_entropy(input, target)
    else:
        return cpu.cross_entropy(input, target)


def div(input: torch.Tensor, other: Union[torch.Tensor, float]) -> torch.Tensor:
    # See pytorch/aten/src/ATen/native/cuda/BinaryMulDivKernel.cu
    if isinstance(other, torch.Tensor) or input.device == "cpu":
        return input.div(other)
    else:
        return input.div(torch.tensor(other, device=input.device))


def mm(
    input: torch.Tensor, mat2: torch.Tensor, transA: bool = False, transB: bool = False
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.mm(input, mat2, transA, transB)
    else:
        return cpu.mm(input, mat2, transA, transB)


def softmax(input: torch.Tensor) -> torch.Tensor:
    if input.is_cuda:
        return cuda.softmax(input)
    else:
        return cpu.softmax(input)


def sqrt(input: torch.Tensor) -> torch.Tensor:
    return input.to(dtype=torch.float64).sqrt().to(dtype=torch.float32)


def sum1d(input: torch.Tensor, average: bool = False) -> torch.Tensor:
    if input.is_cuda:
        return cuda.sum1d(input, average)
    else:
        return cpu.sum1d(input, average)


def sum2d_dim0(
    input: torch.Tensor, keepdim: bool = False, average: bool = False
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.sum2d_dim0(input, keepdim, average)
    else:
        return cpu.sum2d_dim0(input, keepdim, average)


def sum2d_dim1(
    input: torch.Tensor, keepdim: bool = False, average: bool = False
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.sum2d_dim1(input, keepdim, average)
    else:
        return cpu.sum2d_dim1(input, keepdim, average)


def sum4d_dim023(
    input: torch.Tensor, keepdim: bool = False, average: bool = False
) -> torch.Tensor:
    if input.is_cuda:
        return cuda.sum4d_dim023(input, keepdim, average)
    else:
        return cpu.sum4d_dim023(input, keepdim, average)
