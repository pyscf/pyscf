'''
A Python interface to mimic numpy.einsum
'''

import sys
import re
import ctypes
import numpy
from pyscf.lib import misc

libtblis = misc.load_library('libtblis')

libtblis.as_einsum.restype = None
libtblis.as_einsum.argtypes = (
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(), numpy.ctypeslib.ndpointer()
)

tblis_dtype = {
    numpy.dtype(numpy.float32)    : 0,
    numpy.dtype(numpy.double)     : 1,
    numpy.dtype(numpy.complex64)  : 2,
    numpy.dtype(numpy.complex128) : 3,
}

numpy_einsum = numpy.einsum

def _contract(subscripts, *tensors, **kwargs):
    '''
    c = alpha * contract(a, b) + beta * c

    Args:
        tensors (list of ndarray) : Tensors for the operation.

    Kwargs:
        out (ndarray) : If provided, the calculation is done into this array.
        dtype (ndarray) : If provided, forces the calculation to use the data
            type specified.
        alpha (number) : Default is 1
        beta (number) :  Default is 0
    '''
    sub_idx = re.split(',|->', subscripts)
    indices  = ''.join(sub_idx)
    c_dtype = kwargs.get('dtype', numpy.result_type(*tensors))
    if ('...' in subscripts or
        tensors[0].size == 0 or tensors[1].size == 0 or
        not (numpy.issubdtype(c_dtype, numpy.float) or
             numpy.issubdtype(c_dtype, numpy.complex))):
        return numpy_einsum(subscripts, *tensors)

    if '->' not in subscripts:
        # Find chararacters which appear only once in the subscripts for c_descr
        for x in set(indices):
            if indices.count(x) > 1:
                indices = indices.replace(x, '')
        sub_idx += [indices]

    alpha = kwargs.get('alpha', 1)
    beta  = kwargs.get('beta', 0)
    c_dtype = numpy.result_type(c_dtype, alpha, beta)
    alpha = numpy.asarray(alpha, dtype=c_dtype)
    beta  = numpy.asarray(beta , dtype=c_dtype)
    a = numpy.asarray(tensors[0], dtype=c_dtype)
    b = numpy.asarray(tensors[1], dtype=c_dtype)

    a_shape = a.shape
    b_shape = b.shape
    a_descr, b_descr, c_descr = sub_idx
    a_shape_dic = dict(zip(a_descr, a_shape))
    b_shape_dic = dict(zip(b_descr, b_shape))
    if any(a_shape_dic[x] != b_shape_dic[x]
           for x in set(a_descr).intersection(b_descr)):
        raise ValueError('operands dimension error for "%s" : %s %s'
                         % (subscripts, a_shape, b_shape))

    ab_shape_dic = a_shape_dic
    ab_shape_dic.update(b_shape_dic)
    c_shape = tuple([ab_shape_dic[x] for x in c_descr])

    out = kwargs.get('out', None)
    if out is None:
        order = kwargs.get('order', 'C')
        c = numpy.empty(c_shape, dtype=c_dtype, order=order)
    else:
        assert(out.dtype == c_dtype)
        assert(out.shape == c_shape)
        c = out

    a_shape = (ctypes.c_size_t*a.ndim)(*a_shape)
    b_shape = (ctypes.c_size_t*b.ndim)(*b_shape)
    c_shape = (ctypes.c_size_t*c.ndim)(*c_shape)

    nbytes = c_dtype.itemsize
    a_strides = (ctypes.c_size_t*a.ndim)(*[x//nbytes for x in a.strides])
    b_strides = (ctypes.c_size_t*b.ndim)(*[x//nbytes for x in b.strides])
    c_strides = (ctypes.c_size_t*c.ndim)(*[x//nbytes for x in c.strides])

    libtblis.as_einsum(a, a.ndim, a_shape, a_strides, a_descr.encode('ascii'),
                       b, b.ndim, b_shape, b_strides, b_descr.encode('ascii'),
                       c, c.ndim, c_shape, c_strides, c_descr.encode('ascii'),
                       tblis_dtype[c_dtype], alpha, beta)
    return c

def einsum(subscripts, *tensors, **kwargs):
    subscripts = subscripts.replace(' ','')
    if len(tensors) <= 1:
        out = numpy_einsum(subscripts, *tensors, **kwargs)
    elif len(tensors) <= 2:
        out = _contract(subscripts, *tensors, **kwargs)
    else:
        sub_idx = subscripts.split(',', 2)
        res_idx = ''.join(set(sub_idx[0]+sub_idx[1]).intersection(sub_idx[2]))
        res_idx = res_idx.replace(',','')
        script0 = sub_idx[0] + ',' + sub_idx[1] + '->' + res_idx
        subscripts = res_idx + ',' + sub_idx[2]
        tensors = [_contract(script0, *tensors[:2])] + list(tensors[2:])
        out = einsum(subscripts, *tensors, **kwargs)
    return out

