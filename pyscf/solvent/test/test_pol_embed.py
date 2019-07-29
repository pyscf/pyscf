#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto, scf
from pyscf.solvent import pol_embed
import cppe
import pyscf.solvent as solvent
import os
from numpy.testing import assert_allclose


dname = os.path.dirname(__file__)


class TestPolEmbed(unittest.TestCase):
    def test_pol_embed_scf(self):
        mol = gto.Mole()
        mol.atom = '''
        C          8.64800        1.07500       -1.71100
        C          9.48200        0.43000       -0.80800
        C          9.39600        0.75000        0.53800
        C          8.48200        1.71200        0.99500
        C          7.65300        2.34500        0.05500
        C          7.73200        2.03100       -1.29200
        H         10.18300       -0.30900       -1.16400
        H         10.04400        0.25200        1.24700
        H          6.94200        3.08900        0.38900
        H          7.09700        2.51500       -2.01800
        N          8.40100        2.02500        2.32500
        N          8.73400        0.74100       -3.12900
        O          7.98000        1.33100       -3.90100
        O          9.55600       -0.11000       -3.46600
        H          7.74900        2.71100        2.65200
        H          8.99100        1.57500        2.99500
        '''
        mol.basis = "STO-3G"
        mol.build()
        pe_options = cppe.PeOptions()
        pe_options.potfile = os.path.join(dname, "pna_6w.potential")
        print(pe_options.potfile)
        pe = pol_embed.PolEmbed(mol, pe_options)
        mf = solvent.PE(scf.RHF(mol), pe)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        ref_pe_energy = -0.03424830892844
        ref_scf_energy = -482.9411084900
        assert_allclose(ref_pe_energy, mf._pe_energy, atol=1e-6)
        assert_allclose(ref_scf_energy, mf.e_tot, atol=1e-6)


if __name__ == "__main__":
    print("Full Tests for pol_embed")
    unittest.main()
