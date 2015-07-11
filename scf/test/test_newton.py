#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf.scf import dhf


class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g')

        mf = scf.RHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

    def test_nr_rohf(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
        )
        mf = scf.RHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.5783963795897, 9)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.5783963795897, 9)


    def test_nr_uhf(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
        )
        mf = scf.UHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)


    def test_nr_rhf_symm(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            symmetry = 1,
            basis = '6-31g')

        mf = scf.RHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

    def test_nr_rohf_symm(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
            symmetry = 1,
        )
        mf = scf.RHF(mol)
        mf.irrep_nelec['B2'] = (1,0)
        mf.max_cycle = 10
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.578396379589819, 9)


    def test_nr_uhf_symm(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
            symmetry = 1,
        )
        mf = scf.UHF(mol)
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)


    def test_nr_rks(self):
        from pyscf import dft
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g')

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)

    def test_nr_roks(self):
        from pyscf import dft
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
        )
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)


    def test_nr_uks(self):
        from pyscf import dft
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g',
            charge = 1,
            spin = 1,
        )
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)

        mf.max_cycle = 2
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 50
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 7)


if __name__ == "__main__":
    print("Full Tests for Newton raphson")
    unittest.main()

