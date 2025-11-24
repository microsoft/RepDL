import math
from warnings import warn
from typing import Iterator

import torch
from tqdm import tqdm


def generate_test_case(
    m: int, n: int, k: int, dtype: torch.dtype, compute_type: torch.dtype
) -> Iterator[tuple[torch.tensor, torch.tensor, torch.tensor]]:
    """returns a generator that can generate a tuple of test input and the correct output"""
    # number of bits in the mantissa
    nmant = int(-math.log2(torch.finfo(dtype).eps))
    precision = int(-math.log2(torch.finfo(compute_type).eps))
    # number of bits in the exponent
    nexp = torch.finfo(dtype).bits - 1 - nmant
    log2n = math.ceil(math.log2(k - 1))
    significand_range = 2 ** min(nmant + 1, precision + 1 - log2n)
    if significand_range < 2:
        warn(
            f"The reduction size (K) {k} is too large for compute precison {compute_type} so that the sum of {k} 1.0s can be not equal to {k}. Please use a higher-persicion compute type."
        )
        significand_range = 2
    minexp = -(2 ** (nexp - 1)) + 2
    maxexp = 2 ** (nexp - 1) - 1 - log2n
    assert (
        minexp <= maxexp
    ), f"The reduction size {k} is too large for data type {dtype} so that the result can overflow. Please use a wider-range data type."
    while True:
        sign = 1 - 2 * torch.randint(0, 2, [m, k])
        exp = torch.randint(minexp - nmant, maxexp - nmant, [1]).item()
        significand = torch.randint(-significand_range + 1, significand_range, [m, k])
        input = sign * significand * 2.0**exp
        input2 = torch.ones([k, n])
        output = input @ input2
        yield (input.to(dtype), input2.to(dtype), output.to(dtype))

        sign = 1 - 2 * torch.randint(0, 2, [k, n])
        exp = torch.randint(minexp - nmant, maxexp - nmant, [1]).item()
        significand = torch.randint(-significand_range + 1, significand_range, [k, n])
        input2 = sign * significand * 2.0**exp
        input = torch.ones([m, k])
        output = input @ input2
        yield (input.to(dtype), input2.to(dtype), output.to(dtype))


def test_gemm(
    num_trials: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    compute_type: torch.dtype,
    device="cuda",
):
    """test the correctness of GEMM on the CUDA backend of PyTorch"""
    g = generate_test_case(m, n, k, dtype, compute_type)
    for _ in tqdm(range(num_trials)):
        input, input2, correct_output = next(g)
        output = input.to(device=device).mm(input2.to(device=device)).cpu()
        assert output.equal(correct_output)


if __name__ == "__main__":
    test_gemm(
        num_trials=1000,
        m=100,
        n=100,
        k=100,
        dtype=torch.float16,
        compute_type=torch.float16,
        device="cuda",
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    test_gemm(
        num_trials=1000,
        m=100,
        n=100,
        k=100,
        dtype=torch.float16,
        compute_type=torch.float32,
        device="cuda",
    )
