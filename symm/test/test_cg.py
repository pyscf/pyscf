#!/usr/bin/env python

from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import symm

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": 'cc-pVDZ',
             "O": 'cc-pVDZ',}
h2o.build()


class KnowValues(unittest.TestCase):
    def test_real2spinor(self):
        s0 = h2o.intor('cint1e_ovlp_sph')
        s1 = h2o.intor('cint1e_ovlp')

        ua, ub = symm.cg.real2spinor_whole(h2o)

        s2 = reduce(numpy.dot, (ua.T.conj(), s0, ua)) \
           + reduce(numpy.dot, (ub.T.conj(), s0, ub))
        self.assertTrue(numpy.allclose(s2,s1))


if __name__ == "__main__":
    print("Full Tests geom")
    unittest.main()
