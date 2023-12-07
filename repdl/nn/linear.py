# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from repdl import ops


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        res = ops.mm(input, weight, transB=True)
        if bias is not None:
            res += bias
        return res

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        input, weight = saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = ops.mm(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = ops.mm(grad_output, input, transA=True)
        if ctx.needs_input_grad[2]:
            grad_bias = ops.sum2d_dim0(grad_output)
        return grad_input, grad_weight, grad_bias


def linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if input.dim() != 2:
        raise NotImplementedError
    assert weight.dim() == 2
    if bias is not None:
        assert bias.dim() == 1
    return LinearFunction.apply(input, weight, bias)


class Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.weight, self.bias)
