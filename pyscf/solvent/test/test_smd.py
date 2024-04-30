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
from pyscf import gto
from pyscf import scf, dft
from pyscf.solvent import smd

def setUpModule():
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'def2-tzvpp'
    mol.output = '/dev/null'
    mol.build()
    lebedev_order = 29

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_smd(atom, e_ref, solvent='water'):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'def2-tzvpp'
    mol.output = '/dev/null'
    mol.build()
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    smdobj.sasa_ng = 590
    smdobj.lebedev_order = 29
    e_cds = smdobj.get_cds() * 627.509 # in kcal/mol
    mol.stdout.close()
    assert numpy.abs(e_cds - e_ref) < 1e-3

class KnownValues(unittest.TestCase):
    def setUp(self):
        if smd.libsolvent is None:
            raise self.skipTest('SMD Fortran library not compiled')

    def test_cds_solvent(self):
        smdobj = smd.SMD(mol)
        smdobj.sasa_ng = 590
        smdobj.solvent = 'toluene'
        e_cds = smdobj.get_cds()
        assert numpy.abs(e_cds - -0.0013479524949097355) < 1e-8

    def test_cds_water(self):
        smdobj = smd.SMD(mol)
        smdobj.sasa_ng = 590
        smdobj.solvent = 'water'
        e_cds = smdobj.get_cds()
        assert numpy.abs(e_cds - 0.002298448590009083) < 1e-8

    def test_smd_solvent(self):
        mf = scf.RHF(mol)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'ethanol'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.075066568) < 2e-4

    def test_smd_water(self):
        mf = scf.RHF(mol)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.0756052903) < 2e-4

    def test_uhf(self):
        mf = scf.UHF(mol)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.07550951172617) < 2e-4

    def test_rks(self):
        mf = dft.RKS(mol, xc='b3lyp')
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.478626548) < 2e-4

    def test_uks(self):
        mf = dft.UKS(mol, xc='b3lyp')
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.478626548) < 2e-4

    def test_dfrks(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit()
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.47848839552529) < 2e-4

    def test_dfuks(self):
        mf = dft.UKS(mol, xc='b3lyp').density_fit()
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.47848839552529) < 2e-4

    def test_CN(self):
        atom = '''
C       0.000000     0.000000     0.000000
N       0.000000     0.000000     1.500000
H       0.000000     1.000000    -0.500000
H       0.866025    -0.500000    -0.500000
H      -0.866025    -0.500000    -0.500000
    '''
        _check_smd(atom,-2.9126, solvent='water')
        _check_smd(atom, 1.351, solvent='toluene')

    def test_CC(self):
        atom = '''
C 0.000 0.000 0.000
C 1.339 0.000 0.000
H -0.507 0.927 0.000
H -0.507 -0.927 0.000
H 1.846 0.927 0.000
H 1.846 -0.927 0.000
    '''
        _check_smd(atom, 3.2504, solvent='water')
        _check_smd(atom, 0.085, solvent='toluene')

    def test_OO(self):
        atom = '''
O 0.000 0.000 0.000
O 1.207 0.000 0.000
    '''
        _check_smd(atom, 0.0000, solvent='water')
        _check_smd(atom, -2.0842, solvent='toluene')

    def test_ON(self):
        atom = '''
N 0.000 0.000 0.000
O 1.159 0.000 0.000
H -0.360 0.000 0.000
    '''
        _check_smd(atom, 2.2838, solvent='water')
        _check_smd(atom, 1.4294, solvent='toluene')

    def test_OP(self):
        atom = '''
P 0.000 0.000 0.000
O 1.480 0.000 0.000
H -0.932 0.932 0.000
H -0.932 -0.932 0.000
H 0.368 0.000 0.933
    '''
        _check_smd(atom, 3.0384, solvent='water')
        _check_smd(atom,-0.0337, solvent='toluene')

    def test_OC(self):
        atom = '''
C 0.000 0.000 0.000
O 1.208 0.000 0.000
H -0.603 0.928 0.000
H -0.603 -0.928 0.000
    '''
        _check_smd(atom, 4.0974, solvent='water')
        _check_smd(atom, 0.4919, solvent='toluene')

    def test_F(self):
        atom = '''
C 0.000 0.000 0.000
F 1.380 0.000 0.000
H -0.520 0.920 -0.400
H -0.520 -0.920 -0.400
H -0.520 0.000 1.000
    '''
        _check_smd(atom, 3.0212, solvent='water')
        _check_smd(atom, 0.6014, solvent='toluene')

    def test_Si(self):
        atom = '''
Si 0.000 0.000 0.000
H 0.875 0.875 0.875
H -0.875 -0.875 0.875
H 0.875 -0.875 -0.875
H -0.875 0.875 -0.875
    '''
        _check_smd(atom, 2.2328, solvent='water')
        _check_smd(atom, -0.2248, solvent='toluene')

    def test_S(self):
        atom = '''
S 0.000 0.000 0.000
H 0.962 0.280 0.000
H -0.962 0.280 0.000
    '''
        _check_smd(atom, 0.5306, solvent='water')
        _check_smd(atom, -1.5374, solvent='toluene')

    def test_Cl(self):
        atom = '''
C 0.000 0.000 0.000
Cl 1.784 0.000 0.000
H -0.595 0.952 0.000
H -0.595 -0.476 0.824
H -0.595 -0.476 -0.824
    '''
        _check_smd(atom, 2.1279, solvent='water')
        _check_smd(atom, -0.9778, solvent='toluene')

if __name__ == "__main__":
    print("Full Tests for SMDs")
    unittest.main()
