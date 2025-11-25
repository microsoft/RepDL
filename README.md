# RepDL: Reproducible Deep Learning

> This research project is for academic and non-production purposes. Your suggestions and contributions are warmly welcomed.

RepDL is a specialized library designed to facilitate reproducible deep learning by guaranteeing **bitwise identical outcomes across various hardware platforms** for identical training or inference tasks.

Citation:

```
@misc{xie_repdl_2025,
	title = {{RepDL}: {Bit}-level {Reproducible} {Deep} {Learning} {Training} and {Inference}},
	url = {https://arxiv.org/abs/2510.09180},
	author = {Xie, Peichen and Zhang, Xian and Chen, Shuo},
	year = {2025},
	note = {arXiv: 2510.09180},
}
```

## Get Started

Before setting up RepDL, ensure that PyTorch and the corresponding CUDA version are installed on your system.  

To build and install RepDL, execute the following commands in your terminal:  

    git clone https://github.com/microsoft/RepDL.git  
    cd RepDL  
    pip install .  

The easiest way to enable reproducible inference for an existing PyTorch model is by using the following code

    import repdl
    model = repdl.from_torch_module(model)

For reproducible training, refer to the example script located at [examples/mnist_training.py](examples/mnist_trainig.py). The output of this script is consistent across different devices, as shown below:

    Hash of the initial model: 2a2d133895b1684e55d0f152ead2914b55adc551d9790a4b4585309b79c60362
    Hash of the trained model: 31ee86a7f75dbd76bac22f209617eb7349f93b0ede116dba924f328cac3013f1
    Test accuracy of the model on the 10000 test images: 0.9804
    Hash of the logits: 0ca34dcd37105b4af690c46a62407b4b9b097d1063864c6d99f4309aeb4bca3e

## Reproducible Operations, Functions, and Modules

Many operations in PyTorch are non-reproducible, even if `torch.use_deterministic_algorithms(True)` is set. For example, the following operations on a tensor `x`: `torch.mm(x, x)`, `torch.div(x, 10)`, and `torch.sqrt(x)`, can lead to various results on different devices. However, with RepDL, the equivalent operations: `repdl.ops.mm(x, x)`, `repdl.ops.div(x, 10)`, and `repdl.ops.sqrt(x)` will produce identical results regardless of the device used. The script of this example is located at [examples/reproducible_ops.py](examples/reproducible_ops.py).

RepDL defines reproducible operations in `repdl.ops`, and implements them in `repdl.backend`. Building on these operations, RepDL provides PyTorch-compatible functions and modules in `repdl.nn`. Currently, only a subset of functions and modules is available.

To expand RepDL, you can write your own reproducible operations, functions, or modules following the guidelines below:

- Add custom operations to [repdl/ops.py](repdl/ops.py), put the C++ and CUDA implementations of the operations in the backend directory, and register them in [backend/cpu/bind.cpp](backend/cpu/bind.cpp) and [backend/cuda/bind.cpp](backend/cuda/bind.cpp).
- Add custom differentiable functions, i.e., functions compatible with PyTorch's `.backward()` automatic differentiation, to [repdl/func.py](repdl/func.py).
- Add PyTorch-compatible differentiable functions and modules to the corresponding locations, such as [repdl/nn/functionals.py](repdl/nn/functionals.py) and [repdl/optim.py](repdl/optim.py).
- Ensure that your implementations are reproducible by fixing the order of floating-point operations, avoiding instructions that do not comply with the IEEE-754 standard, and using correctly rounded mathematical functions rather than inaccurate ones.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
