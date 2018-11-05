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
import numpy
from pyscf import gto, lib
from pyscf import scf
from pyscf.prop import nmr
from pyscf.data import nist
from pyscf.prop.nsr import rhf


class KnowValues(unittest.TestCase):
    def test_giao_vs_nmr(self):
        mol = gto.Mole()
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = '''h  ,  0.   0.   0.917
                      f  ,  0.   0.   0.
                      '''
        mol.basis = 'dzp'
        mol.build()
        mf = scf.RHF(mol).run()
        m = rhf.NSR(mf).kernel()

        im, mass_center = rhf.inertia_tensor(mol)
        unit_ppm = nist.ALPHA**2 * 1e6
        unit = rhf._atom_gyro_list(mol) * nist.ALPHA**2
        m_nuc = rhf.nuc(mol, [0, 1])

        atm_id = 0
        atom_coords = mol.atom_coords()
        m1 = nmr.RHF(mf).kernel()[atm_id]/unit_ppm
        m2 = nmr.RHF(mf).dia(gauge_orig=atom_coords[atm_id])[atm_id]
        m3 = rhf._safe_solve(im, m1 - m2) * 2 * unit[atm_id]
        m_ref = (m3 + m_nuc[atm_id]) * rhf.AU2KHZ
        self.assertAlmostEqual(abs(m[atm_id] - m_ref).max(), 0, 9)

        atm_id = 1
        atom_coords = mol.atom_coords()
        m1 = nmr.RHF(mf).kernel()[atm_id]/unit_ppm
        m2 = nmr.RHF(mf).dia(gauge_orig=atom_coords[atm_id])[atm_id]
        m3 = rhf._safe_solve(im, m1 - m2) * 2 * unit[atm_id]
        m_ref = (m3 + m_nuc[atm_id]) * rhf.AU2KHZ
        self.assertAlmostEqual(abs(m[atm_id] - m_ref).max(), 0, 9)

    #test against J. Chem. Phys. 138, 024111
    #def test_moles(self):

if __name__ == "__main__":
    print("Full Tests of RHF-NSR")
    unittest.main()
