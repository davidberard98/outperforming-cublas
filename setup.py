from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemm",
    ext_modules=[
        CUDAExtension(
            "gemm",
            [
                "src/op.cu",
            ],
            extra_compile_args=["-lineinfo"],
            extra_link_args=["-lcuda"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
