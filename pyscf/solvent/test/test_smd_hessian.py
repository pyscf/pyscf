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
from pyscf.solvent.hessian import smd as smd_hess
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

def _check_hess(atom, solvent='water'):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto-3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    hess_cds = smd_hess.get_cds(smdobj)

    eps = 1e-4
    coords = mol.atom_coords()
    v = numpy.zeros_like(coords)
    v[0,0] = eps
    mol.set_geom_(coords + v, unit='Bohr')
    mol.build()
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    g0 = smd_grad.get_cds(smdobj)

    mol.set_geom_(coords - v, unit='Bohr')
    mol.build()
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    g1 = smd_grad.get_cds(smdobj)
    h_fd = (g0 - g1)/2.0/eps
    mol.stdout.close()
    assert(numpy.linalg.norm(hess_cds[0,:,0,:] - h_fd) < 1e-3)

class KnownValues(unittest.TestCase):
    def setUp(self):
        if smd.libsolvent is None:
            raise self.skipTest('SMD Fortran library not compiled')

    def test_h2o(self):
        h2o = gto.Mole()
        h2o.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        h2o.basis = 'sto3g'
        h2o.output = '/dev/null'
        h2o.build()

        mf = dft.RKS(h2o, xc='b3lyp').density_fit()
        mf.grids.atom_grid = (99,590)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.with_solvent.lebedev_order = 29

        mf.kernel()
        h = mf.Hessian().kernel()

        assert abs(h[0,0,0,0] - 0.9199776)  < 1e-3
        assert abs(h[0,0,1,1] - -0.0963789) < 1e-3
        assert abs(h[0,0,2,2] - 0.5852264)  < 1e-3
        assert abs(h[1,0,0,0] - -0.4599888) < 1e-3
        h2o.stdout.close()

    def test_CN(self):
        atom = '''
C  0.0  0.0  0.0
H  1.09  0.0  0.0
H  -0.545  0.944  0.0
H  -0.545  -0.944  0.0
N  0.0  0.0  1.16
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_CC(self):
        atom = '''
C 0.000 0.000 0.000
C 1.339 0.000 0.000
H -0.507 0.927 0.000
H -0.507 -0.927 0.000
H 1.846 0.927 0.000
H 1.846 -0.927 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OO(self):
        atom = '''
O 0.000 0.000 0.000
O 1.207 0.000 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_ON(self):
        atom = '''
N 0.000 0.000 0.000
O 1.159 0.000 0.000
H -0.360 0.000 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OP(self):
        atom = '''
P 0.000 0.000 0.000
O 1.480 0.000 0.000
H -0.932 0.932 0.000
H -0.932 -0.932 0.000
H 0.368 0.000 0.933
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OC(self):
        atom = '''
C 0.000 0.000 0.000
O 1.208 0.000 0.000
H -0.603 0.928 0.000
H -0.603 -0.928 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_F(self):
        atom = '''
C 0.000 0.000 0.000
F 1.380 0.000 0.000
H -0.520 0.920 -0.400
H -0.520 -0.920 -0.400
H -0.520 0.000 1.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Si(self):
        atom = '''
Si 0.000 0.000 0.000
H 0.875 0.875 0.875
H -0.875 -0.875 0.875
H 0.875 -0.875 -0.875
H -0.875 0.875 -0.875
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_S(self):
        atom = '''
S 0.000 0.000 0.000
H 0.962 0.280 0.000
H -0.962 0.280 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Cl(self):
        atom = '''
C 0.000 0.000 0.000
Cl 1.784 0.000 0.000
H -0.595 0.952 0.000
H -0.595 -0.476 0.824
H -0.595 -0.476 -0.824
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Br(self):
        atom = '''
C 0.000 0.000 0.000
Br 1.939 0.000 0.000
H -0.646 0.929 0.000
H -0.646 -0.464 0.804
H -0.646 -0.464 -0.804
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

if __name__ == "__main__":
    print("Full Tests for Hessian of SMD")
    unittest.main()
