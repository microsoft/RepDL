# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from hashlib import sha256
from typing import Union

import torch


def get_hash(x: Union[torch.Tensor, torch.nn.Module]) -> str:
    if isinstance(x, torch.Tensor):
        return sha256(x.data.cpu().numpy().tobytes()).hexdigest()
    elif isinstance(x, torch.nn.Module):
        hash = sha256()
        for para in x.parameters():
            hash.update(para.data.cpu().numpy().tobytes())
        return hash.hexdigest()
    else:
        raise TypeError


def print_hash(x: Union[torch.Tensor, torch.nn.Module]) -> None:
    print(get_hash(x))
