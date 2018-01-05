#!/usr/bin/env python
import unittest
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import mp
from pyscf.mp import mp2_grad
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
    def test_mp2_grad(self):
        pt = mp.mp2.MP2(mf)
        pt.kernel()
        g1 = pt.nuc_grad_method().kernel(pt.t2, mf_grad=grad.RHF(mf), atmlst=[0,1,2])
        self.assertAlmostEqual(lib.finger(g1), -0.035681131697586257, 6)

    def test_mp2_grad_finite_diff(self):
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mp_scanner = scf.RHF(mol).set(conv_tol=1e-14).apply(mp.MP2).as_scanner()
        e0 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        e1 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mp_scanner(mol)
        g1 = mp_scanner.nuc_grad_method().kernel()
        self.assertAlmostEqual(g1[0,2], (e1-e0)*500, 6)

    def test_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        pt.max_memory = 1
        pt.kernel()
        g1 = mp2_grad.kernel(pt, pt.t2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), 0.12457973399092415, 6)

    def test_as_scanner(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        gscan = pt.nuc_grad_method().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -76.025166662910223, 9)
        self.assertAlmostEqual(lib.finger(g1), 0.12457973399092415, 6)


if __name__ == "__main__":
    print("Tests for MP2 gradients")
    unittest.main()

