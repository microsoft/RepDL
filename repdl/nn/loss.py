# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from repdl import ops


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


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if (
        weight is not None
        or size_average is not None
        or ignore_index != -100
        or reduce is not None
        or reduction != "mean"
        or label_smoothing != 0.0
        or input.dim() != 2
        or target.dim() != 1
    ):
        raise NotImplementedError
    return CrossEntropyFunction.apply(input, target)


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
