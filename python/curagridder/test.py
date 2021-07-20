# import ctypes
# ctypes.cdll.LoadLibrary("libcurafft.so")

from curagridder import ms2dirty
import numpy as np
import ctypes
import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

arr = np.arange(10,dtype=np.double)
c_arr = np.ctypeslib.as_ctypes(arr)

fk_gpu = gpuarray.GPUArray([10,], dtype=np.double)

ms2dirty(c_arr,fk_gpu.ptr,10)

fk = fk_gpu.get()

print(arr)
print(fk)

