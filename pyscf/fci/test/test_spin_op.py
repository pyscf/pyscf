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

import os
import unittest
from functools import reduce
import numpy
import h5py
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import lib

def setUpModule():
    global h1, h2, c0, ci0, norb, nelec, e0
    hfile = os.path.realpath(os.path.join(__file__, '..', 'spin_op_hamiltonian.h5'))
    with h5py.File(hfile, 'r') as f:
        h1 = lib.unpack_tril(f['h1'][:])
        h2 = f['h2'][:]

    norb = 10
    nelec = (5,5)
    na = fci.cistring.num_strings(norb, nelec[0])
    c0 = numpy.zeros((na,na))
    c0[0,0] = 1
    c0[-1,-1] = 1e-4
    e0, ci0 = fci.direct_spin0.kernel(h1, h2, norb, nelec, ci0=c0)


def tearDownModule():
    global h1, h2, c0, ci0
    del h1, h2, c0, ci0

class KnownValues(unittest.TestCase):
    def test_spin_squre(self):
        ss = fci.spin_op.spin_square(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 6, 9)
        ss = fci.spin_op.spin_square0(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 6, 9)

        numpy.random.seed(1)
        u,w,v = numpy.linalg.svd(numpy.random.random((norb,6)))
        u = u[:,:6]
        h1a = h1[:6,:6]
        h1b = reduce(numpy.dot, (v.T, h1a, v))
        h2aa = ao2mo.restore(1, h2, norb)[:6,:6,:6,:6]
        h2ab = lib.einsum('klpq,pi,qj->klij', h2aa, v, v)
        h2bb = lib.einsum('pqkl,pi,qj->ijkl', h2ab, v, v)
        e1, ci1 = fci.direct_uhf.kernel((h1a,h1b), (h2aa,h2ab,h2bb), 6, (3,2))
        ss = fci.spin_op.spin_square(ci1, 6, (3,2), mo_coeff=(numpy.eye(6),v))[0]
        self.assertAlmostEqual(ss, 3.75, 8)

        numpy.random.seed(1)
        n = fci.cistring.num_strings(6,3)
        ci1 = numpy.random.random((n,n))
        ss1 = numpy.einsum('ij,ij->', ci1, fci.spin_op.contract_ss(ci1, 6, 6))
        self.assertAlmostEqual(ss1, fci.spin_op.spin_square(ci1, 6, 6)[0], 12)

        na = fci.cistring.num_strings(6,4)
        nb = fci.cistring.num_strings(6,2)
        ci1 = numpy.random.random((na,nb))
        ss1 = numpy.einsum('ij,ij->', ci1, fci.spin_op.contract_ss(ci1, 6, (4,2)))
        self.assertAlmostEqual(ss1, fci.spin_op.spin_square(ci1, 6, (4,2))[0], 12)

        numpy.random.seed(1)
        n = fci.cistring.num_strings(10,5)
        ci1 = numpy.random.random((n,n))
        ss1 = numpy.einsum('ij,ij->', ci1, fci.spin_op.contract_ss(ci1, 10, 10))
        self.assertAlmostEqual(ss1, fci.spin_op.spin_square(ci1, 10, 10)[0], 8)

    def test_contract_ss(self):
        self.assertAlmostEqual(e0, -25.4538751043, 9)
        nelec = (6,4)
        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])
        c0 = numpy.zeros((na,nb))
        c0[0,0] = 1
        solver0 = fci.addons.fix_spin(fci.direct_spin0.FCI(), shift=0.02)
        solver1 = fci.addons.fix_spin(fci.direct_spin1.FCI())
        e, ci0 = solver1.kernel(h1, h2, norb, nelec, ci0=c0)
        self.assertAlmostEqual(e, -25.4437866823, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(ci0, norb, nelec)[0], 2, 9)

        # Note: the symmetry required by direct_spin0 (c = c.T) may cause
        # numerical issue for difficult spin systems. In this test, we have to
        # decrease the convergence tolerance. Otherwise the davidson solver
        # may produce vectors that break the symmetry required by direct_spin0.
        nelec = (5,5)
        na = fci.cistring.num_strings(norb, nelec[0])
        c0 = numpy.zeros((na,na))
        c0[0,0] = 1
        e, ci0 = solver0.kernel(h1, h2, norb, nelec, ci0=c0)
        self.assertAlmostEqual(e, -25.4095560762, 7)
        self.assertAlmostEqual(fci.spin_op.spin_square0(ci0, norb, nelec)[0], 0, 5)

    def test_rdm2_baab(self):
        numpy.random.seed(9)
        nelec = 5, 4
        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])
        ci0 = numpy.random.random((na,nb))
        ci0 /= numpy.linalg.norm(ci0)
        dm2baab = fci.spin_op.make_rdm2_baab(ci0, norb, nelec)
        dm2abba = fci.spin_op.make_rdm2_abba(ci0, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm2baab), -0.04113790921902272, 12)
        self.assertAlmostEqual(lib.fp(dm2abba), -0.10910630874863614, 12)

        dm2ab = fci.direct_spin1.make_rdm12s(ci0, norb, nelec)[1][1]
        self.assertAlmostEqual(abs(dm2baab - -dm2ab.transpose(2,1,0,3)).max(), 0, 12)
        self.assertAlmostEqual(abs(dm2abba - -dm2ab.transpose(0,3,2,1)).max(), 0, 12)

    def test_local_spin(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 0 ,  0    , 0.   )],
            ['H', ( 0 ,  0    , 8.   )],
        ]

        mol.basis = {'H': 'cc-pvdz'}
        mol.spin = 0
        mol.build()

        m = scf.RHF(mol)
        ehf = m.scf()

        cis = fci.direct_spin0.FCISolver(mol)
        #cis.verbose = 5
        norb = m.mo_coeff.shape[1]
        nelec = (mol.nelectron, 0)
        nelec = mol.nelectron
        h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
        eri = ao2mo.incore.full(m._eri, m.mo_coeff)
        e, ci0 = cis.kernel(h1e, eri, norb, nelec)
        ss = fci.spin_op.spin_square(ci0, norb, nelec, m.mo_coeff, m.get_ovlp())
        print('local spin for H1+H2 = 0')
        self.assertAlmostEqual(ss[0], 0, 9)
        ss = fci.spin_op.local_spin(ci0, norb, nelec, m.mo_coeff, m.get_ovlp(), range(5))
        print('local spin for H1 = 0.75')
        self.assertAlmostEqual(ss[0], .75, 9)
        ss = fci.spin_op.local_spin(ci0, norb, nelec, m.mo_coeff, m.get_ovlp(), range(5,10))
        print('local spin for H2 = 0.75')
        self.assertAlmostEqual(ss[0], .75, 9)

        ss = fci.spin_op.spin_square0(ci0, norb, nelec)
        print('tot spin for HH = 0')
        self.assertAlmostEqual(ss[0], 0, 9)


if __name__ == "__main__":
    print("Full Tests for fci.spin_op")
    unittest.main()
