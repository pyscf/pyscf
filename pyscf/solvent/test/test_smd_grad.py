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
from pyscf import gto, dft
from pyscf.solvent.grad import smd as smd_grad
from pyscf.solvent import smd

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_grad(atom, solvent='water'):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    natm = mol.natm
    fd_cds = numpy.zeros([natm,3])
    eps = 1e-4
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e0_cds = smdobj.get_cds()

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e1_cds = smdobj.get_cds()

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            fd_cds[ia,j] = (e0_cds - e1_cds) / (2.0 * eps)

    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    grad_cds = smd.get_cds_legacy(smdobj)[1]
    mol.stdout.close()
    assert numpy.linalg.norm(fd_cds - grad_cds) < 1e-8

class KnownValues(unittest.TestCase):
    def setUp(self):
        if smd.libsolvent is None:
            raise self.skipTest('SMD Fortran library not compiled')

    def test_grad_water(self):
        mf = dft.rks.RKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        g_ref = numpy.array(
            [[0.000000,       0.000000,      -0.101523],
            [0.043933,      -0.000000,       0.050761],
            [-0.043933,      -0.000000,       0.050761]]
            )
        assert numpy.linalg.norm(g - g_ref) < 1e-4

        mf = dft.uks.UKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        assert numpy.linalg.norm(g - g_ref) < 1e-4

    def test_grad_solvent(self):
        mf = dft.rks.RKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        g_ref = numpy.array(
            [[-0.000000,       0.000000,      -0.106849],
            [0.047191,      -0.000000,       0.053424],
            [-0.047191,       0.000000,       0.053424]]
            )
        assert numpy.linalg.norm(g - g_ref) < 1e-4

        mf = dft.uks.UKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        assert numpy.linalg.norm(g - g_ref) < 1e-4

    def test_CN(self):
        atom = '''
C       0.000000     0.000000     0.000000
N       0.000000     0.000000     1.500000
H       0.000000     1.000000    -0.500000
H       0.866025    -0.500000    -0.500000
H      -0.866025    -0.500000    -0.500000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_CC(self):
        atom = '''
C 0.000 0.000 0.000
C 1.339 0.000 0.000
H -0.507 0.927 0.000
H -0.507 -0.927 0.000
H 1.846 0.927 0.000
H 1.846 -0.927 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OO(self):
        atom = '''
O 0.000 0.000 0.000
O 1.207 0.000 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_ON(self):
        atom = '''
N 0.000 0.000 0.000
O 1.159 0.000 0.000
H -0.360 0.000 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OP(self):
        atom = '''
P 0.000 0.000 0.000
O 1.480 0.000 0.000
H -0.932 0.932 0.000
H -0.932 -0.932 0.000
H 0.368 0.000 0.933
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OC(self):
        atom = '''
C 0.000 0.000 0.000
O 1.208 0.000 0.000
H -0.603 0.928 0.000
H -0.603 -0.928 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_F(self):
        atom = '''
C 0.000 0.000 0.000
F 1.380 0.000 0.000
H -0.520 0.920 -0.400
H -0.520 -0.920 -0.400
H -0.520 0.000 1.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Si(self):
        atom = '''
Si 0.000 0.000 0.000
H 0.875 0.875 0.875
H -0.875 -0.875 0.875
H 0.875 -0.875 -0.875
H -0.875 0.875 -0.875
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_S(self):
        atom = '''
S 0.000 0.000 0.000
H 0.962 0.280 0.000
H -0.962 0.280 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Cl(self):
        atom = '''
C 0.000 0.000 0.000
Cl 1.784 0.000 0.000
H -0.595 0.952 0.000
H -0.595 -0.476 0.824
H -0.595 -0.476 -0.824
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Br(self):
        atom = '''
C 0.000 0.000 0.000
Br 1.939 0.000 0.000
H -0.646 0.929 0.000
H -0.646 -0.464 0.804
H -0.646 -0.464 -0.804
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

if __name__ == "__main__":
    print("Full Tests for Gradient of SMD")
    unittest.main()
