# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from glob import glob
from setuptools import setup, find_packages

from torch.utils import cpp_extension

setup(
    name="repdl",
    packages=find_packages(),
    ext_modules=[
        cpp_extension.CppExtension(
            name="repdl.backend.cpu",
            sources=glob("backend/cpu/*.cpp"),
            extra_compile_args=["-fopenmp"],
        ),
        cpp_extension.CUDAExtension(
            name="repdl.backend.cuda",
            sources=glob("backend/cuda/*.cu") + glob("backend/cuda/*.cpp"),
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    install_requires=["torch>=1.10.0"],
    version="0.1.1",
)
