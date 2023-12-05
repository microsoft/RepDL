# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from .. import ops
from .. import func


class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        input: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        std = ops.sqrt(var + eps)
        ctx.save_for_backward(input, mean, std, weight)
        if input.dim() == 4:
            output = input - mean.unsqueeze(-1).unsqueeze(-1)
            output /= std.unsqueeze(-1).unsqueeze(-1)
            output *= weight.unsqueeze(-1).unsqueeze(-1)
            output += bias.unsqueeze(-1).unsqueeze(-1)
        else:
            output = (input - mean) / std
            output = weight * output + bias
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        grad_input = grad_mean = grad_var = grad_weight = grad_bias = None
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        input, mean, std, weight = saved_tensors
        if grad_output.dim() == 4:
            mean = mean.unsqueeze(-1).unsqueeze(-1)
            std = std.unsqueeze(-1).unsqueeze(-1)
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        def sum_func(x: torch.Tensor) -> torch.Tensor:
            if grad_output.dim() == 4:
                return ops.sum4d_dim023(x)
            else:
                return ops.sum2d_dim0(x)

        if ctx.needs_input_grad[0]:
            grad_input = weight / std * grad_output
        if ctx.needs_input_grad[1]:
            grad_mean = (-weight / std).squeeze() * sum_func(grad_output)
        if ctx.needs_input_grad[2]:
            grad_var = weight / (2 * std * std * std)
            grad_var = sum_func(grad_output * (mean - input) * grad_var)
        if ctx.needs_input_grad[3]:
            grad_weight = sum_func((input - mean) / std * grad_output)
        if ctx.needs_input_grad[4]:
            grad_bias = sum_func(grad_output)
        return grad_input, grad_mean, grad_var, grad_weight, grad_bias, None


def batch_norm(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    batch_statistics: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    if weight is None or bias is None or (input.dim() != 4 and input.dim() != 2):
        raise NotImplementedError
    assert weight.dim() == bias.dim() == 1
    if batch_statistics:
        if input.dim() == 4:
            mean = func.mean4d_dim023(input)
            var = func.expand_as(mean.unsqueeze(-1).unsqueeze(-1), input)
            var = input - var
            var = func.mean4d_dim023(var * var)
        else:
            mean = func.mean2d_dim0(input)
            var = func.expand_as(mean, input)
            var = input - mean
            var = func.mean2d_dim0(var * var)
        if running_mean is not None and running_var is not None:
            running_mean = momentum * mean.data + (1.0 - momentum) * running_mean
            running_var = momentum * var.data + (1.0 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var
    return BatchNormFunction.apply(input, mean, var, weight, bias, eps)


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        batch_statistics = self.training or not self.track_running_stats
        return batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            batch_statistics,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    _check_input_dim = torch.nn.BatchNorm1d._check_input_dim


class BatchNorm2d(_BatchNorm):
    _check_input_dim = torch.nn.BatchNorm2d._check_input_dim
