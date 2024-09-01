#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
#

import unittest
import numpy as np
from pyscf import lib, gto, scf, tdscf

def setUpModule():
    global mol, mf, nstates
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.symmetry = True
    mol.build()
    mf = scf.RHF(mol).run()
    nstates = 5 # make sure first 3 states are converged

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_tda_singlet(self):
        td = mf.TDA().set(nstates=nstates)
        td.max_memory = 1e-3
        e = td.kernel()[0]
        ref = [11.90276464, 11.90276464, 16.86036434]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.65616659, 5)

    def test_tda_triplet(self):
        td = mf.TDA().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = [11.01747918, 11.01747918, 13.16955056]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_tdhf_singlet(self):
        td = mf.TDHF().set(nstates=nstates)
        e = td.kernel()[0]
        ref = [11.83487199, 11.83487199, 16.66309285]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.64009191, 5)

    def test_tdhf_triplet(self):
        td = mf.TDHF().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = [10.8919234, 10.8919234, 12.63440705]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_symmetry_init_guess(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry='D2h')
        mf = mol.RHF.run()
        td = mf.TDA().run(nstates=1)
        self.assertAlmostEqual(td.e[0], 0.22349707455528, 7)
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orbsym = tdscf.rhf.hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = tdscf.rhf.symm.direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
        wfnsym = tdscf.rhf._analyze_wfnsym(td, x_sym, td.xy[0][0])
        self.assertEqual(wfnsym, 'Au')

        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        mf = mol.RHF.run()
        td = mf.TDA().run(nstates=1, singlet=False)
        self.assertAlmostEqual(td.e[0], 0.14147328219131602, 7)
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orbsym = tdscf.rhf.hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = tdscf.rhf.symm.direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
        wfnsym = tdscf.rhf._analyze_wfnsym(td, x_sym, td.xy[0][0])
        self.assertEqual(wfnsym, 'A1u')


if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
