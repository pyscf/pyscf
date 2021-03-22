#!/usr/bin/env python

import unittest
import numpy
from pyscf import lib, gto, scf, dft, tdscf
from pyscf.gw import gw_ac
from pyscf.gw import gw_cd
from pyscf.gw import gw as gw_exact

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.7571 , 0.5861)],
    ['H' , (0. , 0.7571 , 0.5861)]]
mol.basis = 'def2-svp'
mol.build()

mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_gwac_pade(self):
        nocc = mol.nelectron//2
        #gw_obj = gw.GW(mf, freq_int='ac', frozen=0)
        gw_obj = gw_ac.GWAC(mf, frozen=0)
        gw_obj.linearized = False
        gw_obj.ac = 'pade'
        gw_obj.kernel(orbs=range(nocc-3, nocc+3))
        self.assertAlmostEqual(gw_obj.mo_energy[nocc-1], -0.412849230989, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[nocc], 0.165745160102, 5)

    def test_gwcd(self):
        nocc = mol.nelectron//2
        gw_obj = gw_cd.GWCD(mf, frozen=0)
        gw_obj.linearized = False
        gw_obj.kernel(orbs=range(0, nocc+3))
        self.assertAlmostEqual(gw_obj.mo_energy[nocc-1], -0.41284735, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[nocc], 0.16574524, 5)
        self.assertAlmostEqual(gw_obj.mo_energy[0], -19.53387986, 5)

    def test_gw_exact(self):
        mol = gto.Mole()
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            ['H' , (0. , -0.757 , 0.587)],
            ['H' , (0. , 0.757 , 0.587)]]
        mol.basis = 'cc-pvdz'
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = 'hf'
        mf.kernel()

        nocc = mol.nelectron // 2
        nvir = mf.mo_energy.size - nocc
        td = tdscf.dRPA(mf)
        td.nstates = min(100, nocc*nvir)
        td.kernel()

        gw_obj = gw_exact.GW(mf, td, frozen=0)
        gw_obj.kernel()
        gw_obj.linearized = True
        gw_obj.kernel(orbs=[nocc-1,nocc])
        self.assertAlmostEqual(gw_obj.mo_energy[nocc-1], -0.44684106, 7)
        self.assertAlmostEqual(gw_obj.mo_energy[nocc]  ,  0.17292032, 7)


if __name__ == "__main__":
    print("Full Tests for GW")
    unittest.main()
