# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Ivan Chernyshov <ivan.chernyshov@gmail.com>
#

import unittest
import io, re
from pyscf import gto, dft
from pyscf.solvent import pcm, cosmors
from pyscf.data.nist import BOHR as _BOHR

def setUpModule():
    global mol, cm0, cm1, mf0
    mol = gto.M(atom='''
           6        0.000000    0.000000   -0.542500
           8        0.000000    0.000000    0.677500
           1        0.000000    0.935307   -1.082500
           1        0.000000   -0.935307   -1.082500
                ''', basis='sto3g', verbose=0,
                output='/dev/null')
    # ideal conductor
    cm0 = pcm.PCM(mol)
    cm0.eps = float('inf')
    cm0.method = 'C-PCM'
    cm0.lebedev_order = 29
    cm0.verbose = 0
    # computation
    mf0 = dft.RKS(mol, xc='b3lyp').PCM(cm0)
    mf0.kernel()
    # water
    cm1 = pcm.PCM(mol)
    cm1.eps = 78.4
    cm1.method = 'C-PCM'
    cm1.lebedev_order = 29
    cm1.verbose = 0

def tearDownModule():
    global mol, cm0, cm1, mf0
    mol.stdout.close()
    del mol, cm0, cm1, mf0

class TestCosmoRS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_finite_epsilon(self):
        mf1 = dft.RKS(mol, xc='b3lyp').PCM(cm1)
        mf1.kernel()
        def save_cosmo_file(mf):
            with io.StringIO() as outp:
                cosmors.write_cosmo_file(outp, mf)
        self.assertRaises(ValueError, save_cosmo_file, mf1)

    def test_cosmo_file(self):
        with io.StringIO() as outp: 
            cosmors.write_cosmo_file(outp, mf0)
            text = outp.getvalue()
        E_diel = float(re.search('Dielectric energy \[a.u.\] += +(-*\d+\.\d+)', text).group(1))
        self.assertAlmostEqual(E_diel, -0.0023256022, 5)

    def test_pcm_parameters(self):
        ps = cosmors.get_pcm_parameters(mf0)
        self.assertAlmostEqual(ps['energies']['e_tot'], -112.953044138, 5)
        self.assertAlmostEqual(ps['energies']['e_diel'], -0.0023256022, 5)
        self.assertAlmostEqual(ps['pcm_data']['area'] * _BOHR**2, 64.848604, 2)

    def test_sas_volume(self):
        V1 = cosmors.get_sas_volume(mf0.with_solvent.surface, step = 0.2) * _BOHR**3
        self.assertAlmostEqual(V1, 46.391962, 3)
        V2 = cosmors.get_sas_volume(mf0.with_solvent.surface, step = 0.05) * _BOHR**3
        self.assertAlmostEqual(V2, 46.497054, 3)


if __name__ == "__main__":
    print("Full Tests for COSMO-RS")
    unittest.main()
