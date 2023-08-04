#!/usr/bin/env python

import unittest
import ctypes
import itertools
import numpy
from pyscf.dft.numint import libdft

class KnownValues(unittest.TestCase):
    def test_empty_blocks(self):
        ao_loc = numpy.array([0,51,60,100,112,165,172], dtype=numpy.int32)

        def get_empty_mask(non0tab_mask):
            non0tab_mask = numpy.asarray(non0tab_mask, dtype=numpy.uint8)
            shls_slice = (0, non0tab_mask.size)
            empty_mask = numpy.empty(4, dtype=numpy.int8)
            empty_mask[:] = -9
            libdft.VXCao_empty_blocks(
                empty_mask.ctypes.data_as(ctypes.c_void_p),
                non0tab_mask.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*2)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p))
            return empty_mask.tolist()

        def naive_emtpy_mask(non0tab_mask):
            blksize = 56
            ao_mask = numpy.zeros(ao_loc[-1], dtype=bool)
            for k, (i0, i1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
                ao_mask[i0:i1] = non0tab_mask[k] == 1
            valued = [m.any() for m in numpy.split(ao_mask, [56, 112, 168])]
            empty_mask = ~numpy.array(valued)
            return empty_mask.astype(int).tolist()

        def check(non0tab_mask):
            if get_empty_mask(non0tab_mask) != naive_emtpy_mask(non0tab_mask):
                raise ValueError(non0tab_mask)

        for mask in list(itertools.product([0, 1], repeat=6)):
            check(mask)

if __name__ == "__main__":
    print("Test libdft")
    unittest.main()
