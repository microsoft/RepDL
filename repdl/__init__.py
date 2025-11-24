# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import copy

import torch

from . import func
from . import nn
from . import ops
from . import optim
from . import utils


def from_torch_module(m: torch.nn.Module) -> torch.nn.Module:
    if not isinstance(m, torch.nn.Module):
        raise TypeError
    if type(m) is torch.nn.BatchNorm1d:
        n = nn.BatchNorm1d(
            m.num_features,
            m.eps,
            m.momentum,
            m.weight is not None,
            m.track_running_stats,
        )
        n.load_state_dict(m.state_dict())
    elif type(m) is torch.nn.BatchNorm2d:
        n = nn.BatchNorm2d(
            m.num_features,
            m.eps,
            m.momentum,
            m.weight is not None,
            m.track_running_stats,
        )
        n.load_state_dict(m.state_dict())
    elif type(m) is torch.nn.Conv2d:
        n = nn.Conv2d(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            m.stride,
            m.padding,
            m.dilation,
            m.groups,
            m.bias is not None,
            m.padding_mode,
        )
        n.load_state_dict(m.state_dict())
    elif type(m) is torch.nn.CrossEntropyLoss:
        n = nn.CrossEntropyLoss(
            m.weight, None, m.ignore_index, None, m.reduction, m.label_smoothing
        )
    elif type(m) is torch.nn.Linear:
        n = nn.Linear(m.in_features, m.out_features, m.bias is not None)
        n.load_state_dict(m.state_dict())
    elif type(m) is torch.nn.Sequential:
        n = []
        for sub_m in m:
            n.append(from_torch_module(sub_m))
        n = torch.nn.Sequential(*n)
    else:
        n = copy(m)
        for i, sub_m in m.named_children():
            setattr(n, i, from_torch_module(sub_m))
    return n
