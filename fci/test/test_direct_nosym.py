#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import fci_slow

nelec = (3,4)
norb = 8
h2e = numpy.random.random((norb,norb,norb,norb))
h2e += h2e.transpose(2,3,0,1)
na = fci.cistring.num_strings(norb, nelec[0])
nb = fci.cistring.num_strings(norb, nelec[1])
ci0 = numpy.random.random((na,nb))

class KnowValues(unittest.TestCase):
    def test_contract(self):
        ci1ref = fci_slow.contract_2e(h2e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_2e(h2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))


if __name__ == "__main__":
    print("Full Tests for spin1")
    unittest.main()

