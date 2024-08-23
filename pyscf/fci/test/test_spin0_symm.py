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

from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring, direct_spin1, direct_spin1_symm
import pyscf.symm
from pyscf.fci import fci_slow

def setUpModule():
    global mol, m, h1e, g2e, ci0, cis
    global norb, nelec, orbsym
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
        O    0.  0.      0.
        H    0.  -0.757  0.587
        H    0.  0.757   0.587'''
    mol.basis = 'sto-3g'
    mol.symmetry = 'c2v'
    mol.build()
    m = scf.RHF(mol)
    m.conv_tol_grad = 1e-8
    ehf = m.scf()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
    g2e = ao2mo.incore.full(m._eri, m.mo_coeff)
    orbsym = m.orbsym
    cis = fci.direct_spin0_symm.FCISolver(mol)
    cis.orbsym = orbsym

    numpy.random.seed(15)
    na = fci.cistring.num_strings(norb, nelec//2)
    ci0 = numpy.random.random((na,na))
    ci0 = (ci0 + ci0.T) * .5

def tearDownModule():
    global mol, m, h1e, g2e, ci0, cis
    del mol, m, h1e, g2e, ci0, cis

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=0)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=0)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 83.221199436109003, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=1)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=1)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.571087072474697, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=3)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=3)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.257163492625622, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=2)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=2)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 81.010497935954916, 9)

    def test_kernel(self):
        e, c = fci.direct_spin0_symm.kernel(h1e, g2e, norb, nelec, orbsym=orbsym)
        self.assertAlmostEqual(e, -84.200905534209554, 8)
        e = fci.direct_spin0_symm.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -84.200905534209554, 8)

        eref = fci_slow.kernel(h1e, g2e, norb, nelec)
        self.assertAlmostEqual(e, eref, 9)

    def test_linearmole(self):
        mol = gto.M(
            atom = 'Li 0 0 0; Li 0 0 2.913',
            basis = '''
#BASIS SET: (9s,4p,1d) -> [3s,2p,1d]
Li    S
   1469.0000000              0.0007660             -0.0001200        
    220.5000000              0.0058920             -0.0009230        
     50.2600000              0.0296710             -0.0046890        
     14.2400000              0.1091800             -0.0176820        
      4.5810000              0.2827890             -0.0489020        
      1.5800000              0.4531230             -0.0960090        
      0.5640000              0.2747740             -0.1363800        
      0.0734500              0.0097510              0.5751020        
Li    P
      1.5340000              0.0227840        
      0.2749000              0.1391070        
      0.0736200              0.5003750        
Li    P
      0.0240300              1.0000000        
''',
            symmetry = True,
        )
        mf = mol.RHF().run()
        mci = fci.FCI(mol, mf.mo_coeff, singlet=True)
        ex, ci_x = mci.kernel(wfnsym='E1ux')
        ey, ci_y = mci.kernel(wfnsym='E1uy')
        self.assertAlmostEqual(ex - ey, 0, 7)
        self.assertAlmostEqual(ex - -14.79681308052051, 0, 7)
        ss, sz = mci.spin_square(ci_x, mf.mo_energy.size, mol.nelec)
        self.assertAlmostEqual(ss, 0, 6)

        swap_xy = numpy.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        mo_swapxy = mol.ao_rotation_matrix(swap_xy).dot(mf.mo_coeff)
        u = mf.mo_coeff.T.dot(mf.get_ovlp()).dot(mo_swapxy)
        ci1 = fci.addons.transform_ci(ci_x, (3,3), u.T)
        self.assertAlmostEqual(ci_x.ravel().dot(ci_y.ravel()), 0, 9)
        self.assertAlmostEqual(abs(ci1.ravel().dot(ci_y.ravel())), 1, 9)


if __name__ == "__main__":
    print("Full Tests for spin0 symm")
    unittest.main()
