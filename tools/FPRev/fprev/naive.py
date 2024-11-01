import numpy
import torch
from graphviz import Digraph
from .ops import *


def Verify(op: NumpySum | TorchSum | JaxSum, tree: Digraph, n_trials: int=1) -> bool:
    n = op.n_summands
    op = op.__class__(n)
    for _ in range(n_trials):
        A = numpy.random.randn(n).astype(numpy.float32)
        op.data = A.copy()
        if type(op) is TorchSum:
            op.data = torch.tensor(op.data)
        order = tree.source.split("\n")
        for line in order:
            if "->" not in line:
                continue
            line = line.split("->")
            i = int(line[0]) % n
            j = int(line[1]) % n
            if i != j:
                A[j] += A[i]
        if A[0] != op.get_sum():
            return False
    return True


def BruteForce(op: NumpySum | TorchSum | JaxSum) -> Digraph:
    tree = Digraph()
    n = op.n_summands
    V = list(range(n))
    for i in range(n):
        tree.node(str(i))

    def search(d: int, T: Digraph) -> Digraph:
        if d == n - 1:
            return T if Verify(op, T, 500) else None
        for i in range(n):
            for j in range(i + 1, n):
                if V[i] >= 0 and V[j] >= 0:
                    new_T = T.copy()
                    new_T.node(str(V[i] + n), "+")
                    new_T.edge(str(V[i]), str(V[i] + n))
                    new_T.edge(str(V[j]), str(V[i] + n))
                    V[i] += n
                    V[j] = -V[j]
                    new_T = search(d + 1, new_T)
                    if new_T is not None:
                        return new_T
                    V[i] -= n
                    V[j] = -V[j]
        return None

    return search(0, tree)
