from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tluconv1d',
    ext_modules=[
        CUDAExtension('tluconv1d', [
            'tluconv1d.cpp',
            'tluconv1d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
