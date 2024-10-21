## A tool for testing correctness of linear operations

### Introduction

Linear operations (such as REDUCE, ALLREDUCE, DOT, GEMV, GEMM, CONV, etc.) can be inconsistent due to different floating-point computation orders. However, this variety can obscure bugs.

Traditional unit testing sets an empirical error bound (e.g., 0.01) to test correctness, which has no theoretic basis. This cannot detect correctness bugs whose error is within the error bound.

This tool generates test cases with unique correct output. In other words, the output is consistent in arbitrary computation order, so we can check the correctness by simple equality test.

### Method

As the sum of fixed-numbers is consistent in any computation order, we leverage this idea for floating-point summation. We truncate random floating-point input with a fixed range, making it like fixed-point numbers.

For example, suppose we have five 6-bit precision floating-point numbers, and truncate them to the range of the largest one with 5-bit precision.

```
1010.10
  11.0111
 100.110
   1.00011
   0.0101010
------||||||||
      truncate
```

Then, we obtain

```
1010.1
  11.0
 100.1
   1.0
   0.0
```

The sum of these truncated numbers equals to 10011.0 in any summation order. Therefore, we use these numbers as test input, and compare the result with the unique answer 10011.0 to check the correctness of the summation operation.

### Example

In `test_sum.py` and `test_gemm.py`, we demonstrate how to test the correctness of SUM and GEMM on the CUDA backend of PyTorch.
