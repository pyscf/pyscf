#!/usr/bin/env python
import unittest
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import ci
from pyscf.ci import cisd_grad
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
    def test_cisd_grad(self):
        myci = ci.cisd.CISD(mf)
        myci.max_memory = 1
        myci.conv_tol = 1e-10
        myci.kernel()
        g1 = cisd_grad.kernel(myci, myci.ci, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), -0.032562347119070523, 7)

        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mf0 = scf.RHF(mol).run(conv_tol=1e-14)
        myci0 = ci.cisd.CISD(mf0).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        mf1 = scf.RHF(mol).run(conv_tol=1e-14)
        myci1= ci.cisd.CISD(mf1).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mf2 = scf.RHF(mol).run(conv_tol=1e-14)
        myci2 = ci.cisd.CISD(mf2).run(conv_tol=1e-10)
        g1 = cisd_grad.kernel(myci2, myci2.ci)
        self.assertAlmostEqual(g1[0,2], (myci1.e_tot-myci0.e_tot)*500, 6)

    def test_frozen(self):
        myci = ci.cisd.CISD(mf)
        myci.frozen = [0,1,10,11,12]
        myci.max_memory = 1
        myci.kernel()
        g1 = cisd_grad.kernel(myci, myci.ci, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), 0.10224149952700579, 6)


if __name__ == "__main__":
    print("Tests for CISD gradients")
    unittest.main()

