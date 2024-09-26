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

import unittest
import numpy
from pyscf import scf, gto, solvent, mcscf, cc, dft
from pyscf.solvent import pcm

def setUpModule():
    global mol, mol0, epsilon, lebedev_order
    mol0 = gto.Mole()
    mol0.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol0.basis = 'sto3g'
    mol0.output = '/dev/null'
    mol0.build(verbose=0)
    mol = mol0.copy()
    mol.nelectron = mol.nao * 2
    epsilon = 35.9
    lebedev_order = 3

def tearDownModule():
    global mol, mol0
    mol.stdout.close()
    mol0.stdout.close()
    del mol, mol0

def _energy_with_solvent(mf, method):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = 29
    cm.method = method
    mf = mf.PCM(cm)
    e_tot = mf.kernel()
    return e_tot

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_CPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'C-PCM')
        print(f"Energy error in RHF with C-PCM: {numpy.abs(e_tot - -71.19244927767662)}")
        assert numpy.abs(e_tot - -71.19244927767662) < 1e-5

    def test_COSMO(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'COSMO')
        print(f"Energy error in RHF with COSMO: {numpy.abs(e_tot - -71.16259314943571)}")
        assert numpy.abs(e_tot - -71.16259314943571) < 1e-5

    def test_IEFPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'IEF-PCM')
        print(f"Energy error in RHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024647)}")
        assert numpy.abs(e_tot - -71.19244457024647) < 1e-5

    def test_SSVPE(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'SS(V)PE')
        print(f"Energy error in RHF with SS(V)PE: {numpy.abs(e_tot - -71.13576912425178)}")
        assert numpy.abs(e_tot - -71.13576912425178) < 1e-5

    def test_uhf(self):
        e_tot = _energy_with_solvent(scf.UHF(mol), 'IEF-PCM')
        print(f"Energy error in UHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024645)}")
        assert numpy.abs(e_tot - -71.19244457024645) < 1e-5

    def test_rks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in RKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_uks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in UKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_dfrks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFRKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643568)}")
        assert numpy.abs(e_tot - -71.67135250643568) < 1e-5

    def test_dfuks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFUKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643567)}")
        assert numpy.abs(e_tot - -71.67135250643567) < 1e-5
    def test_reset(self):
        mol1 = gto.M(atom='H 0 0 0; H 0 0 .9', basis='cc-pvdz')
        mf = scf.RHF(mol).density_fit().PCM().newton()
        mf = mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertTrue(mf.with_solvent.mol is mol1)
        self.assertTrue(mf._scf.with_df.mol is mol1)
        self.assertTrue(mf._scf.with_solvent.mol is mol1)

    def test_casci(self):
        mf = scf.RHF(mol0).PCM().run()
        mc = solvent.PCM(mcscf.CASCI(mf,2,2)).run()
        assert numpy.abs(mc.e_tot - -74.97040819700919) < 1e-8

    def test_casscf(self):
        mf = scf.RHF(mol0).run()
        mc1 = solvent.PCM(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e1 = mc1.e_tot
        assert numpy.abs(e1 - -74.9709884530835) < 1e-8

    def test_ccsd(self):
        mf = scf.RHF(mol0).PCM()
        mf.conv_tol = 1e-12
        mf.kernel()
        mycc = cc.CCSD(mf).PCM()
        mycc.kernel()
        assert numpy.abs(mycc.e_tot - -75.0172322934944) < 1e-8

if __name__ == "__main__":
    print("Full Tests for PCMs")
    unittest.main()
