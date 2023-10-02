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
from pyscf import scf, gto, df
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

class KnownValues(unittest.TestCase):
    def test_CPCM(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 29
        cm.method = 'C-PCM'
        mf = scf.RHF(mol).PCM(cm)
        e_tot = mf.kernel()
        print(f"Energy error in C-PCM: {numpy.abs(e_tot - -74.9690902442)}")
        assert numpy.abs(e_tot - -74.9690902442) < 1e-9

    def test_COSMO(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 29
        cm.method = 'COSMO'
        mf = scf.RHF(mol).PCM(cm)
        e_tot = mf.kernel()
        print(f"Energy error in COSMO: {numpy.abs(e_tot - -74.96900351922464)}")
        assert numpy.abs(e_tot - -74.96900351922464) < 1e-9
    
    def test_IEFPCM(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 29
        cm.method = 'IEF-PCM'
        mf = scf.RHF(mol).PCM(cm)
        e_tot = mf.kernel()
        print(f"Energy error in IEF-PCM: {numpy.abs(e_tot - -74.9690111344)}")
        assert numpy.abs(e_tot - -74.9690111344) < 1e-9
    
    def test_SSVPE(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 29
        cm.method = 'SS(V)PE'
        mf = scf.RHF(mol).PCM(cm)
        e_tot = mf.kernel()
        print(f"Energy error in SS(V)PE: {numpy.abs(e_tot - -74.9689577454)}")
        assert numpy.abs(e_tot - -74.9689577454) < 1e-9
    
if __name__ == "__main__":
    print("Full Tests for PCMs")
    unittest.main()
