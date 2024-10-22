#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
from functools import reduce
import numpy
from pyscf import lib
from pyscf.fci import direct_ep

nsite = 2
nelec = 2
nphonon = 3

t = numpy.zeros((nsite,nsite))
idx = numpy.arange(nsite-1)
t[idx+1,idx] = t[idx,idx+1] = -1
u = 1.5
g = 0.5
hpp = numpy.eye(nsite) * 1.1
hpp[idx+1,idx] = hpp[idx,idx+1] = .1

class KnownValues(unittest.TestCase):
    def test_kernel(self):
        es = []
        nelecs = [(ia,ib) for ia in range(nsite+1) for ib in range(ia+1)]
        for nelec in nelecs:
            e,c = direct_ep.kernel(t, u, g, hpp, nsite, nelec, nphonon,
                                   tol=1e-10, verbose=0, nroots=1)
            #print('nelec =', nelec, 'E =', e)
            es.append(e)
        es = numpy.hstack(es)
        idx = numpy.argsort(es)
        #print(es[idx])
        ref = [-1.43147218e+00, -1.04287040e+00, 0, 0, 4.57129605e-01, 3.00000000e+00]
        self.assertAlmostEqual(abs(es[idx] - ref).max(), 0, 8)

    def test_make_rdm(self):
        nelec = (1,1)
        e,c = direct_ep.kernel(t, u, g, hpp, nsite, nelec, nphonon,
                     tol=1e-10, verbose=0, nroots=1)
        dm1 = direct_ep.make_rdm1e(c, nsite, nelec)

        dm1a, dm2 = direct_ep.make_rdm12e(c, nsite, nelec)
        print('check 1e DM')
        self.assertTrue(numpy.allclose(dm1, dm1a))
        print('check 2e DM')
        self.assertTrue(numpy.allclose(dm1, numpy.einsum('ijkk->ij', dm2)/(sum(nelec)-1.)))
        print('check 2e DM')
        self.assertTrue(numpy.allclose(dm1, numpy.einsum('kkij->ij', dm2)/(sum(nelec)-1.)))

        dm1 = direct_ep.make_rdm1p(c, nsite, nelec, nphonon)
        dm1a = numpy.empty_like(dm1)
        for i in range(nsite):
            for j in range(nsite):
                c1 = direct_ep.des_phonon(c, nsite, nelec, nphonon, j)
                c1 = direct_ep.cre_phonon(c1, nsite, nelec, nphonon, i)
                dm1a[i,j] = numpy.dot(c.ravel(), c1.ravel())
        print('check phonon DM')
        self.assertTrue(numpy.allclose(dm1, dm1a))

    def test_contract_2e_hubbard(self):
        cishape = direct_ep.make_shape(nsite, nelec, nphonon)
        eri = numpy.zeros((nsite,nsite,nsite,nsite))
        for i in range(nsite):
            eri[i,i,i,i] = u
        numpy.random.seed(3)
        ci0 = numpy.random.random(cishape)
        ci1 = direct_ep.contract_2e([eri*0,eri*.5,eri*0], ci0, nsite, nelec, nphonon)
        ci2 = direct_ep.contract_2e_hubbard(u, ci0, nsite, nelec, nphonon)
        self.assertAlmostEqual(abs(ci1-ci2).sum(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for direct_ep (electron-phonon coupled system)")
    unittest.main()
