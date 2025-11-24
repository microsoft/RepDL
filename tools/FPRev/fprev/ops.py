import numpy
import torch
import jax


class OpTemplate:
    n_summands: int

    def __init__(self, n: int, use_gpu: bool = False): ...
    def set_mask(self, k: int, negative: bool): ...
    def reset(self, k: int): ...
    def get_sum(self) -> float: ...


class NumpySum(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        if use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return self.data.sum()


class NumpyDot(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        if use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return self.data @ self.ones


class NumpyGEMV(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        if use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = numpy.ones([n, n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return (self.data @ self.ones)[0]


class NumpyGEMM(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        if use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = numpy.ones([n, n], dtype=numpy.float32)
        self.ones = numpy.ones([n, n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[0, k] = 1

    def get_sum(self) -> float:
        return (self.data @ self.ones)[0, 0]


class TorchSum(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return self.data.sum().item()


class TorchDot(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return (self.data @ self.ones).item()


class TorchGEMV(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n, n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self) -> float:
        return (self.data @ self.ones)[0].item()


class TorchGEMM(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        device = "cuda" if use_gpu else "cpu"
        self.data = torch.ones([n, n], dtype=torch.float32, device=device)
        self.ones = torch.ones([n, n], dtype=torch.float32, device=device)

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[0, k] = 1

    def get_sum(self) -> float:
        return (self.data @ self.ones)[0, 0].item()


class TorchF16GEMM(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        if not use_gpu:
            raise ValueError
        self.n_summands = n
        self.data = torch.full([n, n], 2.0**-24, dtype=torch.float16, device="cuda")
        self.ones = torch.ones([n, n], dtype=torch.float16, device="cuda")

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**15) if negative else 2.0**15

    def reset(self, k: int):
        self.data[0, k] = 2.0**-24

    def get_sum(self) -> float:
        return (self.data @ self.ones)[0, 0].item() * 2.0**24


class JaxSum(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self):
        with jax.default_device(self.device):
            return jax.numpy.sum(self.data).item()


class JaxDot(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self):
        with jax.default_device(self.device):
            return jax.numpy.dot(self.data, self.ones).item()


class JaxGEMV(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n, n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[k] = 1

    def get_sum(self):
        with jax.default_device(self.device):
            return jax.numpy.dot(self.data, self.ones)[0].item()


class JaxGEMM(OpTemplate):
    def __init__(self, n: int, use_gpu: bool = False):
        self.n_summands = n
        self.device = jax.devices("gpu")[0] if use_gpu else jax.devices("cpu")[0]
        self.data = numpy.ones([n, n], dtype=numpy.float32)
        self.ones = jax.device_put(
            jax.numpy.ones([n, n], dtype=jax.numpy.float32), device=self.device
        )

    def set_mask(self, k: int, negative: bool):
        self.data[0, k] = -(2.0**127) if negative else 2.0**127

    def reset(self, k: int):
        self.data[0, k] = 1

    def get_sum(self):
        with jax.default_device(self.device):
            return jax.numpy.dot(self.data, self.ones)[0, 0].item()
