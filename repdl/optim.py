# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch

from . import ops


class Adam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state["step"] += 1
                    state_steps.append(state["step"])

            # Changed:

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(torch.mul(grad, grad).mul(1 - beta2))

                if group["amsgrad"]:
                    torch.maximum(
                        max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
                    )
                    denom = ops.div(
                        ops.sqrt(max_exp_avg_sqs[i]), math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = ops.div(
                        ops.sqrt(exp_avg_sq), math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                param.add_(torch.div(exp_avg, denom).mul(-step_size))

        return loss
