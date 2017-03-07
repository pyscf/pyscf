#!/usr/bin/env python

import os
import ctypes
import numpy
from pyscf import lib

from ctypes import CDLL, POINTER, c_int64, c_double, byref, c_char
from numpy import empty

dll = CDLL('../siesta.so')

if __name__ == '__main__':
    print('A test')
    fname = numpy.array("file_name\x00", dtype="c")
    dll.c_get_string(fname.ctypes.data_as(POINTER(c_char)))
    
    
    
    npy = 8000
    a = c_int64(33)
    n = c_int64(npy)
    d = c_double(2.0)
    dat = empty(npy, dtype="double")
    dll.c_hello_world(byref(a), byref(d), byref(n), dat.ctypes.data_as(POINTER(c_double)))
    
    

    print(dat[0])
    print("after c_hello_world")
    #print(libsiesta)
    #print(libsiesta.hello_world)


