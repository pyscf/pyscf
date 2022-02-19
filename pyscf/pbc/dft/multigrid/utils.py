import numpy
from pyscf import lib

def _take_4d(a, indices):
    a_shape = a.shape
    ranges = []
    for i, s in enumerate(indices):
        if s is None:
            idx = numpy.arange(a_shape[i], dtype=numpy.int32)
        else:
            idx = numpy.asarray(s, dtype=numpy.int32)
            idx[idx < 0] += a_shape[i]
        ranges.append(idx)
    idx = ranges[0][:,None] * a_shape[1] + ranges[1]
    idy = ranges[2][:,None] * a_shape[3] + ranges[3]
    a = a.reshape(a_shape[0]*a_shape[1], a_shape[2]*a_shape[3])
    out = lib.take_2d(a, idx.ravel(), idy.ravel())
    return out.reshape([len(s) for s in ranges])

def _takebak_4d(out, a, indices):
    out_shape = out.shape
    a_shape = a.shape
    ranges = []
    for i, s in enumerate(indices):
        if s is None:
            idx = numpy.arange(a_shape[i], dtype=numpy.int32)
        else:
            idx = numpy.asarray(s, dtype=numpy.int32)
            idx[idx < 0] += out_shape[i]
        assert(len(idx) == a_shape[i])
        ranges.append(idx)
    idx = ranges[0][:,None] * out_shape[1] + ranges[1]
    idy = ranges[2][:,None] * out_shape[3] + ranges[3]
    nx = idx.size
    ny = idy.size
    out = out.reshape(out_shape[0]*out_shape[1], out_shape[2]*out_shape[3])
    lib.takebak_2d(out, a.reshape(nx,ny), idx.ravel(), idy.ravel())
    return out

def _take_5d(a, indices):
    a_shape = a.shape
    a = a.reshape((a_shape[0]*a_shape[1],) + a_shape[2:])
    indices = (None,) + indices[2:]
    return _take_4d(a, indices)

def _takebak_5d(out, a, indices):
    a_shape = a.shape
    out_shape = out.shape
    a = a.reshape((a_shape[0]*a_shape[1],) + a_shape[2:])
    out = out.reshape((out_shape[0]*out_shape[1],) + out_shape[2:])
    indices = (None,) + indices[2:]
    return _takebak_4d(out, a, indices)

