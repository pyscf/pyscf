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
import os
import tempfile
from numpy.testing import assert_allclose
from pyscf import gto, scf

have_pe = False
try:
    import cppe
    import pyscf.solvent as solvent
    from pyscf.solvent import pol_embed
    have_pe = True
except (ImportError, ModuleNotFoundError):
    pass


dname = os.path.dirname(__file__)

potf = tempfile.NamedTemporaryFile()
potf.write(b'''!
@COORDINATES
3
AA
O     3.53300000    2.99600000    0.88700000      1
H     4.11100000    3.13200000    1.63800000      2
H     4.10500000    2.64200000    0.20600000      3
@MULTIPOLES
ORDER 0
3
1     -0.67444000
2      0.33722000
3      0.33722000
@POLARIZABILITIES
ORDER 1 1
3
1      5.73935000     0.00000000     0.00000000     5.73935000     0.00000000     5.73935000
2      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
3      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
EXCLISTS
3 3
1   2  3
2   1  3
3   1  2''')
potf.flush()

mol = gto.M(atom='''
       6        0.000000    0.000000   -0.542500
       8        0.000000    0.000000    0.677500
       1        0.000000    0.935307   -1.082500
       1        0.000000   -0.935307   -1.082500
            ''', basis='sto3g', verbose=7,
            output='/dev/null')

def tearDownModule():
    global potf, mol
    mol.stdout.close()
    del potf, mol


@unittest.skipIf(not have_pe, "CPPE library not found.")
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
        mf.kernel()
        ref_pe_energy = -0.03424830892844
        ref_scf_energy = -482.9411084900
        assert_allclose(ref_pe_energy, mf.with_solvent.e, atol=1e-6)
        assert_allclose(ref_scf_energy, mf.e_tot, atol=1e-6)

    def test_pe_scf(self):
        pe = solvent.PE(mol, potf.name)
        mf = solvent.PE(mol.RHF(), pe).run(conv_tol=1e-10)
        self.assertAlmostEqual(mf.e_tot, -112.35232445743728, 9)
        self.assertAlmostEqual(mf.with_solvent.e, 0.00020182314249546455, 9)

    def test_as_scanner(self):
        mf_scanner = solvent.PE(scf.RHF(mol), potf.name).as_scanner()
        mf_scanner(mol)
        self.assertAlmostEqual(mf_scanner.with_solvent.e, 0.00020182314249546455, 9)
        mf_scanner('H  0. 0. 0.; H  0. 0. .9')
        self.assertAlmostEqual(mf_scanner.with_solvent.e, 5.2407234004672825e-05, 9)

    def test_newton_rohf(self):
        mf = solvent.PE(mol.ROHF(max_memory=0), potf.name)
        mf = mf.newton()
        e = mf.kernel()
        self.assertAlmostEqual(e, -112.35232445745123, 9)

        mf = solvent.PE(mol.ROHF(max_memory=0), potf.name)
        e = mf.kernel()
        self.assertAlmostEqual(e, -112.35232445745123, 9)


if __name__ == "__main__":
    print("Full Tests for pol_embed")
    unittest.main()
