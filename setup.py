
import os
import ctypes
from setuptools import setup, Extension


lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/libcurafft.so")
try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
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
    url="https://github.com/HLSUD/CURIG",
    install_requires=['numpy', 'pycuda', 'six'],
    python_requires='>=3.6',
    zip_safe=False,
    ext_modules=[
        Extension(name='CURIG',
                  sources=[],
                  libraries=['curafft'],
                  library_dirs=['lib'])
        ]
)