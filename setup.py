
import os
import subprocess
from setuptools import setup, Extension 
from distutils.command.install import install as _install

class _deferred_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)        

include_dirs = ['./', _deferred_pybind11_include(True),
                _deferred_pybind11_include()]
extra_compile_args = ['-Wall', '-Wextra', '-Wfatal-errors', '-Wstrict-aliasing=2', '-Wwrite-strings', '-Wredundant-decls', '-Woverloaded-virtual', '-Wcast-qual', '-Wcast-align', '-Wpointer-arith', '-Wfloat-conversion']
#, '-Wsign-conversion', '-Wconversion'
python_module_link_args = []


class install(_install):
    def run(self):
        subprocess.call(['make', 'clean'])
        subprocess.call(['make', 'python'])
        _install.run(self)

setup(
    name='curig',  
    version='0.1',
    author="LIU Honghao",
    author_email="stein.h.liu@gmail.com",
    description="GPU version of NUFFT For Radio astronomy gridder package",
    packages=['curig'],
    package_dir={'curig': 'python/curagridder'},
    package_data={'curig': ['libcurafft.so']},
    setup_requires=['numpy>=1.15.0', 'pybind11>=2.2.4'],
    url="https://github.com/astronomical-data-processing/cuRIG",
    install_requires=['numpy', 'pycuda', 'six'],
    python_requires='>=3.6',
    zip_safe=False,
    include_package_data=True,
    cmdclass={'install': install},
)