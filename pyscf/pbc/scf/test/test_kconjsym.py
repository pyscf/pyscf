#!/usr/bin/env python

import unittest
import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.gto import Cell
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.scf.addons import kconj_symmetry_


cell = Cell()
cell.atom = 'O 0 0 0 ; O 2 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[4,10,10]
cell.unit = 'A'
cell.verbose = 7
cell.output = '/dev/null'
cell.build()
kpts = cell.make_kpts([4,1,1])

kmf = scf.KRHF(cell, kpts).density_fit()
kmf.conv_tol = 1e-10
kmf = kconj_symmetry_(kmf)
kmf.kernel()

def break_spinsym(cell, dm1, delta=0.05):
    """Break spin symmetry of density-matrix, to converge AFM order"""
    start, stop = cell.aoslice_by_atom()[:,[2,3]]
    atm1 = np.s_[start[0]:stop[0]]
    atm2 = np.s_[start[1]:stop[1]]
    dm1a, dm1b = dm1.copy()
    # Atom 1: Majority spin = alpha
    dm1a[:,atm1,atm1] += delta*dm1b[:,atm1,atm1]
    dm1b[:,atm1,atm1] -= delta*dm1b[:,atm1,atm1]
    # Atom 2: Majority spin = beta
    dm1a[:,atm2,atm2] -= delta*dm1a[:,atm2,atm2]
    dm1b[:,atm2,atm2] += delta*dm1a[:,atm2,atm2]
    return np.stack((dm1a, dm1b), axis=0)

kumf = scf.KUHF(cell, kpts).density_fit()
kumf.conv_tol = 1e-10
kumf = kconj_symmetry_(kumf)
dm0 = break_spinsym(cell, kumf.get_init_guess())
kumf.kernel(dm0=dm0)

def get_symmetry_error(mf):
    dm1 = mf.make_rdm1()
    conj_indices = kpts_helper.conj_mapping(mf.mol, mf.kpts)
    ddm1 = (dm1 - dm1[...,conj_indices,:,:].conj())
    err = np.linalg.norm(ddm1)
    return err

class KnownValues(unittest.TestCase):

    # KRHF

    def test_krhf_converged(self):
        self.assertTrue(kmf.converged)

    def test_krhf_energy(self):
        self.assertAlmostEqual(kmf.e_tot, -149.08951818965966, 8)

    def test_krhf_symmetry(self):
        self.assertAlmostEqual(get_symmetry_error(kmf), 0.0, 12)

    # KUHF

    def test_kuhf_converged(self):
        self.assertTrue(kumf.converged)

    def test_kuhf_energy(self):
        self.assertAlmostEqual(kumf.e_tot, -149.53931291972197, 8)

    def test_kuhf_symmetry(self):
        self.assertAlmostEqual(get_symmetry_error(kumf), 0.0, 12)

def tearDownModule():
    global cell, kmf, kumf
    cell.stdout.close()
    del cell, kmf, kumf

if __name__ == '__main__':
    print("Full Tests for pbc.scf.kconjsym")
    unittest.main()
