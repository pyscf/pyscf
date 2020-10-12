#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
A Python interface to mimic numpy.einsum
'''

import re
import ctypes
import numpy
from pyscf.lib import misc

libtblis = misc.load_library('libtblis_einsum')

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

EINSUM_MAX_SIZE = getattr(misc.__config__, 'lib_einsum_max_size', 2000)

_numpy_einsum = numpy.einsum
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
    a = numpy.asarray(tensors[0])
    b = numpy.asarray(tensors[1])
    if not kwargs and (a.size < EINSUM_MAX_SIZE or b.size < EINSUM_MAX_SIZE):
        return _numpy_einsum(subscripts, a, b)

    c_dtype = kwargs.get('dtype', numpy.result_type(a, b))
    if (not (numpy.issubdtype(c_dtype, numpy.floating) or
             numpy.issubdtype(c_dtype, numpy.complexfloating))):
        return _numpy_einsum(subscripts, a, b)

    sub_idx = re.split(',|->', subscripts)
    indices  = ''.join(sub_idx)
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
    a = numpy.asarray(a, dtype=c_dtype)
    b = numpy.asarray(b, dtype=c_dtype)

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

