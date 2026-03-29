from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-Xcompiler", "-O3",
        "-gencode", "arch=compute_80,code=sm_80",   # A100
        "-gencode", "arch=compute_86,code=sm_86",   # RTX 3090 / A10
        "-gencode", "arch=compute_89,code=sm_89",   # RTX 4090
        "-gencode", "arch=compute_87,code=sm_87",   # Jetson AGX Orin
    ],
}

setup(
    name="cuda_matmul",
    version="0.1.0",
    author="Jayden Vasquez",
    description="Custom CUDA matrix multiplication kernels with PyTorch bindings",
    ext_modules=[
        CUDAExtension(
            name="cuda_matmul",
            sources=[
                "python/bindings.cpp",
                "src/matmul.cu",
            ],
            include_dirs=[
                os.path.join(os.path.dirname(__file__), "include"),
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
