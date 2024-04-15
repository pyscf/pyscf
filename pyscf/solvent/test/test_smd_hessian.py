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
from pyscf.solvent.hessian import smd as smd_hess
from pyscf.solvent import smd

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = '''P 0.000 0.000 0.000
O 1.500 0.000 0.000
O -1.500 0.000 0.000
O 0.000 1.500 0.000
O 0.000 -1.500 0.000
H 1.000 1.000 0.000
H -1.000 -1.000 0.000
H 0.000 -2.000 0.000
'''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_hess(mol, solvent='water'):
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    smd_hess.get_cds(smdobj)

class KnownValues(unittest.TestCase):
    def test_hess_water(self):
        _check_hess(mol, solvent='water')

    def test_hess_solvent(self):
        _check_hess(mol, solvent='ethanol')

if __name__ == "__main__":
    print("Full Tests for Hessian of SMD")
    unittest.main()