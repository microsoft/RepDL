# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union

import torch

from repdl import ops


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        res = ops.conv2d(input, weight, stride=stride, padding=padding, groups=groups)
        if bias is not None:
            res += bias.unsqueeze(-1).unsqueeze(-1)
        return res

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        if ctx.groups != 1:
            raise NotImplementedError
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        input, weight = saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = ops.conv2d_grad_input(
                input.shape, weight, grad_output, ctx.stride, ctx.padding
            )
        if ctx.needs_input_grad[1]:
            grad_weight = ops.conv2d_grad_kernel(
                input, weight.shape, grad_output, ctx.stride, ctx.padding
            )
        if ctx.needs_input_grad[2]:
            grad_bias = ops.sum4d_dim023(grad_output)
        return grad_input, grad_weight, grad_bias, None, None, None


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[str, int, tuple[int, int]] = 0,
    dilation: Union[int, tuple[int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    if (
        isinstance(padding, str)
        or (dilation != 1 and dilation != (1, 1))
        # or groups != 1
        or input.dim() != 4
    ):
        raise NotImplementedError
    assert weight.dim() == 4
    if bias is not None:
        assert bias.dim() == 1
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    return Conv2dFunction.apply(input, weight, bias, stride, padding, groups)


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, target)
        return ops.cross_entropy(input, target)

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        input, target = saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ops.softmax(input) - torch.nn.functional.one_hot(
                target, input.shape[1]
            )
            grad_input *= 1.0 / input.shape[0]
            grad_input *= grad_output
        return grad_input, None


class Conv2d(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
