# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from . import ops


class ExpandFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        return input.expand(shape)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        grad_input = None
        if ctx.needs_input_grad[0]:
            if grad_output.dim() == 4:
                grad_input = ops.sum4d_dim023(grad_output, keepdim=True).squeeze(0)
            else:
                grad_input = ops.sum2d_dim0(grad_output)
        return grad_input, None


def expand_as(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # (n) -> (?, n) or (n, 1, 1) -> (?, n, ?, ?)
    if not (
        input.dim() == 1 and other.dim() == 2 and input.shape[-1] == other.shape[-1]
    ) and not (
        input.dim() == 3
        and other.dim() == 4
        and input.shape[-3] == other.shape[-3]
        and input.shape[-2] == input.shape[-1] == 1
    ):
        raise NotImplementedError
    return ExpandFunction.apply(input, other.shape)


class Mean1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input: torch.Tensor
    ) -> torch.Tensor:
        ctx.input_shape = input.shape
        return ops.sum1d(input, average=True)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> torch.Tensor:
        grad_input = None
        input_shape: torch.Size = ctx.input_shape
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (1 / input_shape.numel())
            grad_input = grad_input.expand(input_shape).contiguous()
        return grad_input


def mean1d(input: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 1
    return Mean1dFunction.apply(input)


class Mean2dDim0Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input: torch.Tensor
    ) -> torch.Tensor:
        ctx.input_shape = input.shape
        return ops.sum2d_dim0(input, average=True)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> torch.Tensor:
        grad_input = None
        input_shape: torch.Size = ctx.input_shape
        if ctx.needs_input_grad[0]:
            factor = 1 / (input_shape.numel() // grad_output.numel())
            grad_input = grad_output * factor
            grad_input = grad_input.unsqueeze(-1).expand(input_shape)
        return grad_input


def mean2d_dim0(input: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 2
    return Mean2dDim0Function.apply(input)


class Mean4dDim023Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input: torch.Tensor
    ) -> torch.Tensor:
        ctx.input_shape = input.shape
        return ops.sum4d_dim023(input, average=True)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> torch.Tensor:
        grad_input = None
        input_shape: torch.Size = ctx.input_shape
        if ctx.needs_input_grad[0]:
            factor = 1 / (input_shape.numel() // grad_output.numel())
            grad_input = grad_output * factor
            grad_input = grad_input.unsqueeze(-1).unsqueeze(-1).expand(input_shape)
        return grad_input


def mean4d_dim023(input: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 4
    return Mean4dDim023Function.apply(input)
