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
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 35.9
    lebedev_order = 3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

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
    def test_CPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'C-PCM')
        print(f"Energy error in RHF with C-PCM: {numpy.abs(e_tot - -74.9690902442)}")
        assert numpy.abs(e_tot - -74.9690902442) < 1e-9

    def test_COSMO(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'COSMO')
        print(f"Energy error in RHF with COSMO: {numpy.abs(e_tot - -74.96900351922464)}")
        assert numpy.abs(e_tot - -74.96900351922464) < 1e-9

    def test_IEFPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'IEF-PCM')
        print(f"Energy error in RHF with IEF-PCM: {numpy.abs(e_tot - -74.9690111344)}")
        assert numpy.abs(e_tot - -74.9690111344) < 1e-9

    def test_SSVPE(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'SS(V)PE')
        print(f"Energy error in RHF with SS(V)PE: {numpy.abs(e_tot - -74.9689577454)}")
        assert numpy.abs(e_tot - -74.9689577454) < 1e-9

    def test_uhf(self):
        e_tot = _energy_with_solvent(scf.UHF(mol), 'IEF-PCM')
        print(f"Energy error in UHF with IEF-PCM: {numpy.abs(e_tot - -74.96901113434953)}")
        assert numpy.abs(e_tot - -74.96901113434953) < 1e-9

    def test_rks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in RKS with IEF-PCM: {numpy.abs(e_tot - -75.3182692148)}")
        assert numpy.abs(e_tot - -75.3182692148) < 1e-6

    def test_uks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in UKS with IEF-PCM: {numpy.abs(e_tot - -75.3182692148)}")
        assert numpy.abs(e_tot - -75.3182692148) < 1e-6

    def test_dfrks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFRKS with IEF-PCM: {numpy.abs(e_tot - -75.31863727142068)}")
        assert numpy.abs(e_tot -  -75.31863727142068) < 1e-9

    def test_dfuks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFUKS with IEF-PCM: {numpy.abs(e_tot - -75.31863727142068)}")
        assert numpy.abs(e_tot - -75.31863727142068) < 1e-9

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
        mf = scf.RHF(mol).PCM().run()
        mc = solvent.PCM(mcscf.CASCI(mf,2,2)).run()
        assert numpy.abs(mc.e_tot - -74.97040819700919) < 1e-8

    def test_casscf(self):
        mf = scf.RHF(mol).run()
        mc1 = solvent.PCM(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e1 = mc1.e_tot
        assert numpy.abs(e1 - -74.9709884530835) < 1e-8

    def test_ccsd(self):
        mf = scf.RHF(mol).PCM()
        mf.conv_tol = 1e-12
        mf.kernel()
        mycc = cc.CCSD(mf).PCM()
        mycc.kernel()
        assert numpy.abs(mycc.e_tot - -75.0172322934944) < 1e-8

if __name__ == "__main__":
    print("Full Tests for PCMs")
    unittest.main()
