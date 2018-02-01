#!/usr/bin/env python
import unittest
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf.cc import ccsd_grad
from pyscf import grad

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()


class KnownValues(unittest.TestCase):
    def test_ccsd_grad(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), -0.036999389889460096, 6)

        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mf0 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc0 = cc.ccsd.CCSD(mf0).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        mf1 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc1= cc.ccsd.CCSD(mf1).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mycc2 = cc.ccsd.CCSD(scf.RHF(mol))
        g_scanner = mycc2.nuc_grad_method().as_scanner()
        g1 = g_scanner(mol)[1]
        self.assertAlmostEqual(g1[0,2], (mycc1.e_tot-mycc0.e_tot)*500, 6)

    def test_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.max_memory = 1
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), 0.10599503839207361, 6)


if __name__ == "__main__":
    print("Tests for CCSD gradients")
    unittest.main()

