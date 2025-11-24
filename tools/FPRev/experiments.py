from timeit import default_timer

import pandas

from fprev.advanced import FPRevAdvanced
from fprev.basic import FPRevBasic
from fprev.naive import BruteForce, Verify
from fprev.ops import *


def run(algo: callable, op: OpTemplate) -> list[float]:
    print(algo.__name__, op.__class__, op.n_summands)
    algo(op)
    res = []
    for t in range(10):
        print(f"Run {t}: ", end="")
        time = default_timer()
        algo(op)
        time = default_timer() - time
        print(f"{time} sec")
        res.append(time)
    return res


def rq1():
    algos = [BruteForce, FPRevBasic, FPRevAdvanced]
    ops = [NumpySum, TorchSum, JaxSum]
    data = []
    sizes = []
    for op in ops:
        for algo in algos:
            n = 4
            tested_n = []
            res = []
            while True:
                ret = run(algo, op(n))
                time = sum(ret) / 10
                print("mean:", time)
                tested_n.append(n)
                res.append(time)
                if time > 1 or (algo is BruteForce and n == 8):
                    break
                if n < 8:
                    n += 1
                else:
                    n *= 2
            data.append(res)
            if len(tested_n) > len(sizes):
                sizes = tested_n
    df = pandas.DataFrame(data).transpose()
    df.index = sizes
    df.columns = pandas.MultiIndex.from_product(
        [[op.__name__ for op in ops], [algo.__name__ for algo in algos]]
    )
    print(df)
    df.to_csv("rq1.csv")


def rq2():
    algos = [FPRevBasic, FPRevAdvanced]
    ops = [
        NumpyDot,
        TorchDot,
        JaxDot,
        NumpyGEMV,
        TorchGEMV,
        JaxGEMV,
        NumpyGEMM,
        TorchGEMM,
        JaxGEMM,
    ]
    data = []
    sizes = []
    for op in ops:
        for algo in algos:
            n = 4
            tested_n = []
            res = []
            while True:
                ret = run(algo, op(n))
                time = sum(ret) / 10
                print("mean:", time)
                tested_n.append(n)
                res.append(time)
                if time > 1:
                    break
                n *= 2
            data.append(res)
            if len(tested_n) > len(sizes):
                sizes = tested_n
    df = pandas.DataFrame(data).transpose()
    df.index = sizes
    df.columns = pandas.MultiIndex.from_product(
        [[op.__name__ for op in ops], [algo.__name__ for algo in algos]]
    )
    print(df)
    df.to_csv("rq2.csv")


def rq3():
    algos = [FPRevBasic, FPRevAdvanced]
    data = []
    sizes = []
    for use_gpu in [False, True]:
        for algo in algos:
            n = 4
            tested_n = []
            res = []
            while True:
                ret = run(algo, TorchGEMM(n, use_gpu))
                time = sum(ret) / 10
                print("mean:", time)
                tested_n.append(n)
                res.append(time)
                if time > 1:
                    break
                n *= 2
            data.append(res)
            if len(tested_n) > len(sizes):
                sizes = tested_n
    df = pandas.DataFrame(data).transpose()
    df.index = sizes
    df.columns = pandas.MultiIndex.from_product(
        [["CPU", "GPU"], [algo.__name__ for algo in algos]]
    )
    print(df)
    df.to_csv("rq3.csv")


def rq4():
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        for op in [NumpySum, TorchSum, JaxSum]:
            res = FPRevAdvanced(op(n, use_gpu=False))
            res.render(f"{op.__name__}{n}", format="pdf")


def rq5():
    for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        res = FPRevAdvanced(TorchF16GEMM(n, use_gpu=True))
        res.render(f"gemm{n}", format="pdf")


rq1()
rq2()
rq3()
rq4()
rq5()
