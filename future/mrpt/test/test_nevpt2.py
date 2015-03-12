#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf.mrpt import nevpt2

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null' #None
mol.atom = [
    ['H', ( 0., 0.    , 0.    )],
    ['H', ( 0., 0.    , 0.8   )],
    ['H', ( 0., 0.    , 2.    )],
    ['H', ( 0., 0.    , 2.8   )],
    ['H', ( 0., 0.    , 4.    )],
    ['H', ( 0., 0.    , 4.8   )],
    ['H', ( 0., 0.    , 6.    )],
    ['H', ( 0., 0.    , 6.8   )],
    ['H', ( 0., 0.    , 8.    )],
    ['H', ( 0., 0.    , 8.8   )],
    ['H', ( 0., 0.    , 10.    )],
    ['H', ( 0., 0.    , 10.8   )],
    ['H', ( 0., 0.    , 12     )],
    ['H', ( 0., 0.    , 12.8   )],
]
mol.basis = 'sto3g'
mol.build()
mf = scf.RHF(mol)
mf.kernel()
norb = 6
nelec = 8
mc = mcscf.CASCI(mf, norb, nelec)
mc.kernel()
mo_cas = mf.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
h1e = mc.h1e_for_cas()[0]
h2e = ao2mo.incore.full(mf._eri, mo_cas)
h2e = ao2mo.restore(1, h2e, norb).transpose(0,2,1,3)
dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
                                         mc.ci, mc.ci, norb, nelec)
orbe = mcscf.addons.get_fock(mc, mc.ci, mc.mo_coeff).diagonal()
hdm1 = 2.0*numpy.eye(dm1.shape[0])-dm1.T
eris = nevpt2._ERIS(mc, mc.mo_coeff)
dms = {'1': dm1, '2': dm2, '3': dm3, '4': dm4}

class KnowValues(unittest.TestCase):
    def test_Sr(self):
        norm, e = nevpt2.Sr(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.02024615448982894, 9)
        self.assertAlmostEqual(norm, 0.039479583324952064, 9)

    def test_Si(self):
        norm, e = nevpt2.Si(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.0021282083432783445, 9)
        self.assertAlmostEqual(norm, 0.0037402318601481785, 9)

    def test_Sijrs(self):
        norm, e = nevpt2.Sijrs(mc,orbe, eris)
        self.assertAlmostEqual(e, -0.0071505006221038612, 9)
        self.assertAlmostEqual(norm, 0.023107593129840662, 9)

    def test_Sijr(self):
        norm, e = nevpt2.Sijr(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.0050346131472232781, 9)
        self.assertAlmostEqual(norm, 0.012664063064240805, 9)

    def test_Srsi(self):
        norm, e = nevpt2.Srsi(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.013695648258441447, 9)
        self.assertAlmostEqual(norm, 0.040695897271711433, 9)

    def test_Srs(self):
        norm, e = nevpt2.Srs(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.017531232809731869, 9)
        self.assertAlmostEqual(norm, 0.056323603233950678, 9)

    def test_Sir(self):
        norm, e = nevpt2.Sir(mc,orbe,dms, eris)
        self.assertAlmostEqual(e, -0.033866605721793064, 9)
        self.assertAlmostEqual(norm, 0.074269055796470834, 9)

    def test_energy(self):
        e = nevpt2.sc_nevpt(mc)
        self.assertAlmostEqual(e, -0.10315310550843806, 9)

    def test_energy(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
            O   0 0 0
            O   0 0 1.207''',
            basis = '6-31g',
            spin = 2)
        m = scf.RHF(mol)
        m.scf()
        mc = mcscf.CASCI(m, 6, 8)
        mc.kernel()
        e = nevpt2.sc_nevpt(mc)
        self.assertAlmostEqual(e, -0.16978532268234559, 7)


if __name__ == "__main__":
    print("Full Tests for nevpt2")
    unittest.main()

