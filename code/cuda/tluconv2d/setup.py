from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tluconv2d',
    ext_modules=[
        CUDAExtension('tluconv2d', [
            'tluconv2d.cpp',
            'tluconv2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
