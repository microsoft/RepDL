import math
from warnings import warn
from typing import Iterator

import torch
from tqdm import tqdm


def generate_test_case(
    num_summands: int, dtype: torch.dtype, compute_type: torch.dtype
) -> Iterator[tuple[torch.tensor, torch.tensor]]:
    """returns a generator that can generate a tuple of test input and the correct output"""
    # number of bits in the mantissa
    nmant = int(-math.log2(torch.finfo(dtype).eps))
    precision = int(-math.log2(torch.finfo(compute_type).eps))
    # number of bits in the exponent
    nexp = torch.finfo(dtype).bits - 1 - nmant
    log2n = math.ceil(math.log2(num_summands - 1))
    significand_range = 2 ** min(nmant + 1, precision + 1 - log2n)
    if significand_range < 2:
        warn(
            f"The number of summands {num_summands} is too large for compute precison {compute_type} so that the sum of {num_summands} 1.0s can be not equal to {num_summands}. Please use a higher-persicion compute type."
        )
        significand_range = 2
    minexp = -(2 ** (nexp - 1)) + 2
    maxexp = 2 ** (nexp - 1) - 1 - log2n
    assert (
        minexp <= maxexp
    ), f"The number of summands {num_summands} is too large for data type {dtype} so that the result can overflow. Please use a wider-range data type."
    while True:
        sign = 1 - 2 * torch.randint(0, 2, [num_summands])
        exp = torch.randint(minexp - nmant, maxexp - nmant, [1]).item()
        significand = torch.randint(
            -significand_range + 1, significand_range, [num_summands]
        )
        input = sign * significand * 2.0**exp
        output = input.sum()
        yield (input.to(dtype), output.to(dtype))


def test_sum(
    num_trials: int,
    num_summands: int,
    dtype: torch.dtype,
    compute_type: torch.dtype,
    device="cuda",
):
    """test the correctness of SUM on the CUDA backend of PyTorch"""
    g = generate_test_case(num_summands, dtype, compute_type)
    for _ in tqdm(range(num_trials)):
        input, correct_output = next(g)
        output = input.to(device=device).sum().cpu()
        assert output.equal(correct_output)


if __name__ == "__main__":
    test_sum(
        num_trials=10000,
        num_summands=1000,
        dtype=torch.float32,
        compute_type=torch.float32,
        device="cuda",
    )
