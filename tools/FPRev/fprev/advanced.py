from graphviz import Digraph
from .ops import OpTemplate


def FPRevAdvanced(op: OpTemplate, debug: bool = False) -> Digraph:
    tree = Digraph()

    def build(indexes: list[int]) -> tuple[int, int]:
        def calculate_l(i: int, others: list[int]) -> list[int]:
            res = []
            op.set_mask(i, negative=False)
            for j in others:
                op.set_mask(j, negative=True)
                res.append(op.n_summands - int(op.get_sum()))
                op.reset(j)
            op.reset(i)
            return res

        nleaves_lca = [1] + calculate_l(indexes[0], indexes[1:])
        F = sorted(zip(nleaves_lca, indexes))
        indexes = [i for _, i in F]
        nleaves_lca = [x for x, _ in F]

        tree.node(str(indexes[0]))
        current_root = indexes[0]
        k = 1
        while k < len(indexes):
            cnt = 1
            while k + cnt < len(indexes) and nleaves_lca[k + cnt] == nleaves_lca[k]:
                cnt += 1
            other_root, other_nleaves = build(indexes[k : k + cnt])
            if debug and indexes[0] == 0:
                print(
                    f"s[i] += s[i + {indexes[k]}] (sum of {k} elements += sum of {other_nleaves} elements)"
                )

            if other_nleaves > cnt:
                tree.edge(str(current_root), str(other_root))
                current_root = other_root
            else:
                next_index = current_root + op.n_summands
                tree.node(str(next_index), "+")
                tree.edge(str(current_root), str(next_index))
                tree.edge(str(other_root), str(next_index))
                current_root = next_index
                next_index += 1
            k += cnt
        return current_root, nleaves_lca[-1]

    build(list(range(op.n_summands)))
    return tree
