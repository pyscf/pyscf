#!/usr/bin/env python

import unittest
import numpy
import copy
from pyscf import gto, lib, scf, dft
from pyscf.prop import gtensor

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = '''
    H  0. , 0. , .917
    F  0. , 0. , 0.'''
mol.basis = 'ccpvdz'
mol.spin = 1
mol.charge = 1
mol.build()

mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.conv_tol_grad = 1e-6
mf.conv_tol = 1e-12
mf.kernel()

nao = mol.nao_nr()
numpy.random.seed(1)
dm0 = numpy.random.random((2,nao,nao))
dm0 = dm0 + dm0.transpose(0,2,1)
dm1 = numpy.random.random((2,3,nao,nao))
dm1 = dm1 - dm1.transpose(0,1,3,2)

class KnowValues(unittest.TestCase):
    def test_nr_lda_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'lda,vwn'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.finger(dat), -0.008807083583654644, 9)

    def test_nr_bp86_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'bp86'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.finger(dat), -0.0088539747015796387, 9)

    def test_nr_b3lyp_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'b3lyp'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.finger(dat), -0.103162219789111, 9)

    def test_nr_uks(self):
        g = gtensor.uhf.GTensor(mf)
        g.dia_soc2e = None
        g.para_soc2e = 'SSO+SOO'
        g.so_eff_charge = True
        g.cphf = False
        dat = g.align(g.kernel())[0]
        self.assertAlmostEqual(lib.finger(dat), 0.39806613807209507, 7)


if __name__ == "__main__":
    print("Full Tests for DFT g-tensor")
    unittest.main()
