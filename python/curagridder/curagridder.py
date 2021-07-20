import ctypes
import os
import warnings

import numpy as np

from ctypes import c_double
from ctypes import c_int
from ctypes import c_float
from ctypes import c_void_p

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

# TODO: See if there is a way to improve this so it is less hacky.
lib = None
# Try to load a local library directly.
try:
    lib = ctypes.cdll.LoadLibrary("/home/liuhao/Project/NUFFT/lib/libcurafft.so")
except Exception:
    raise RuntimeError('Failed to find curagridder library')


def _get_ctypes(dtype):
    """
    Checks dtype is float32 or float64.
    Returns floating point and floating point pointer.
    Y. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett (2021). cuFINUFFT
    """

    if dtype == np.float64:
        REAL_t = c_double
    elif dtype == np.float32:
        REAL_t = c_float
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    REAL_ptr = ctypes.POINTER(REAL_t)

    return REAL_t, REAL_ptr


ms2dirty = lib.ms2dirty
# the last two parameters have default value
ms2dirty.argtypes = [c_int, c_int, c_int, c_double, c_double, c_double_p, c_double_p, c_double_p,
                     np.ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'), c_void_p, c_double, c_double] 
ms2dirty.restype = c_int


def imaging_ms2dirty():
    print("xiqi")
