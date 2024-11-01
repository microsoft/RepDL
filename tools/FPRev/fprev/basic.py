from graphviz import Digraph
from .ops import OpTemplate


def FPRevBasic(op: OpTemplate) -> Digraph:
    tree = Digraph()
    n = op.n_summands
    for i in range(n):
        tree.node(str(i))
    L = []
    for i in range(n):
        op.set_mask(i, negative=False)
        for j in range(i + 1, n):
            op.set_mask(j, negative=True)
            L.append((n - int(op.get_sum()), i, j))
            op.reset(j)
        op.reset(i)

    class DisjointSet:
        def __init__(self, n: int):
            self.ancestor = list(range(n))

        def find_root(self, k: int) -> int:
            f = self.ancestor[k % n]
            if f % n == k % n:
                return f
            f = self.find_root(f)
            self.ancestor[k % n] = f
            return f

        def merge(self, i: int, j: int, root: int):
            self.ancestor[i % n] = root
            self.ancestor[j % n] = root

    S = DisjointSet(n)
    L = sorted(L)
    for _, i, j in L:
        i = S.find_root(i)
        j = S.find_root(j)
        if i != j:
            k = i + n
            tree.node(str(k), "+")
            tree.edge(str(i), str(k))
            tree.edge(str(j), str(k))
            S.merge(i, j, k)
    return tree
