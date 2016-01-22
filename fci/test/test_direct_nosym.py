#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

nelec = (3,4)
norb = 8
h2e = numpy.random.random((norb,norb,norb,norb))
h2e += h2e.transpose(2,3,0,1)
na = fci.cistring.num_strings(norb, nelec[0])
nb = fci.cistring.num_strings(norb, nelec[1])
ci0 = numpy.random.random((na,nb))

def contract_2e_o0(g2e, fcivec, norb, nelec, opt=None):
    neleca, nelecb = nelec
    link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
    na = fci.cistring.num_strings(norb, neleca)
    nb = fci.cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * fcivec[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * fcivec[:,str0]
    t1 = numpy.dot(g2e.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na,nb)
    fcinew = numpy.zeros_like(fcivec)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * t1[a,i,:,str0]
    return fcinew

class KnowValues(unittest.TestCase):
    def test_contract(self):
        ci1ref = contract_2e_o0(h2e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_2e(h2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))


if __name__ == "__main__":
    print("Full Tests for spin1")
    unittest.main()

