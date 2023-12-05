# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import repdl

torch.manual_seed(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

x = torch.rand(1000, 1000)

print("PyTorch")
y_cpu = torch.mm(x, x)
y_cuda = torch.mm(x.cuda(), x.cuda())
print(torch.equal(y_cpu, y_cuda.cpu()))  # False
y_cpu = torch.div(x, 10)
y_cuda = torch.div(x.cuda(), 10)
print(torch.equal(y_cpu, y_cuda.cpu()))  # False
y_cpu = torch.sqrt(x)
y_cuda = torch.sqrt(x.cuda())
print(torch.equal(y_cpu, y_cuda.cpu()))  # False

print("RepDL")
y_cpu = repdl.ops.mm(x, x)
y_cuda = repdl.ops.mm(x.cuda(), x.cuda())
print(torch.equal(y_cpu, y_cuda.cpu()))  # True
y_cpu = repdl.ops.div(x, 10)
y_cuda = repdl.ops.div(x.cuda(), 10)
print(torch.equal(y_cpu, y_cuda.cpu()))  # True
y_cpu = repdl.ops.sqrt(x)
y_cuda = repdl.ops.sqrt(x.cuda())
print(torch.equal(y_cpu, y_cuda.cpu()))  # True
