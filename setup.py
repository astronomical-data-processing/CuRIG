
import os
import ctypes
from setuptools import setup, Extension

try:
    lib = ctypes.cdll.LoadLibrary('libcufinufft.so')
except Exception as e:
    print('CUDA shared libraries not found in library path.')
    raise(e)

setup(
    name='curagridder',  
    version='0.1',
    author="LIU Honghao",
    author_email="stein.h.liu@gmail.com",
    description="GPU version of NUFFT and Radio astronomy gridder package",
    packages=['curagridder'],
    package_dir={'': 'python'},
    url="https://github.com/HLSUD/NUFFT",
    install_requires=['numpy', 'pycuda', 'six'],
    python_requires='>=3.6',
    ext_modules=[
        Extension(name='NUFFT',
                  sources=[],
                  libraries=['NUFFT'],
                  library_dirs=['lib'])
        ]
)