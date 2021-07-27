from curagridder import imaging_ms2dirty
import numpy as np
import time

def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize):
    speedoflight = 299792458.

    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
        indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((nxdirty, nydirty))

    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    n = nm1+1
    for row in range(ms.shape[0]):
        for chan in range(ms.shape[1]):
            phase = (freq[chan]/speedoflight *
                     (x*uvw[row, 0] + y*uvw[row, 1] + uvw[row, 2]*nm1))
            res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
    return res/n


def test_against_wdft(nrow, nchan, nxdirty, nydirty, fov, epsilon):
    print("\n\nTesting imaging with {} rows and {} "
          "frequency channels".format(nrow, nchan))
    print("Dirty image has {}x{} pixels, "
          "FOV={} degrees".format(nxdirty, nydirty, fov))
    print("Requested accuracy: {}".format(epsilon))

    xpixsize = fov*np.pi/180/nxdirty
    ypixsize = fov*np.pi/180/nydirty

    speedoflight = 299792458.
    np.random.seed(40)
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(f0/speedoflight)
    
    ms = np.random.rand(nrow, nchan)-0.5 + 1j*(np.random.rand(nrow, nchan)-0.5)
    dirty = np.zeros((nxdirty,nydirty),dtype=np.complex128)
    print("begin")
    start = time.time()
    dirty = imaging_ms2dirty(uvw,freq, ms, None, dirty, fov, epsilon,2)
    end = time.time()
    print("The elapsed time {} (sec)".format(end-start))
    print("Execution finished")
    dirty = np.reshape(dirty,[nxdirty,nydirty])
    if nrow<1e4:
        print("Vertification begin")
        truth = explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize)
        print("L2 error between explicit transform and gridder:",
              _l2error(truth, dirty.real))

test_against_wdft(10000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(10000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(10000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(10000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(10000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(10000, 1, 512, 512, 2, 1e-12)


print("new 1024")
test_against_wdft(500000000, 1, 1024, 1024, 2, 1e-12)

#test_against_wdft(700000000, 1, 1024, 1024, 2, 1e-12)



# time.sleep(5)
# test_against_wdft(10000, 1, 2048, 2048, 2, 1e-12)


