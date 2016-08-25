#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import dft


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
        nr.max_cycle = 2
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
        nr.max_cycle = 2
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
        nr.max_cycle = 2
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
        nr.max_cycle = 2
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
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 2
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
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)


    def test_nr_rks_lda(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g')

        mf = dft.RKS(mol)
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_rks(self):
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
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_rks_gen_g_hop(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g')

        mf = dft.RKS(mol)
        mf.grids.build()
        mf.xc = 'b3lyp'
        nao = mol.nao_nr()
        numpy.random.seed(1)
        mo = numpy.random.random((nao,nao))
        mo_occ = numpy.zeros(nao)
        mo_occ[:5] = 2
        nocc, nvir = 5, nao-5
        dm1 = numpy.random.random(nvir*nocc)
        nr = scf.newton(mf)
        g, hop, hdiag = nr.gen_g_hop(mo, mo_occ, mf.get_hcore())
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 40669.392804071264, 7)

    def test_nr_roks(self):
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
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)


    def test_nr_uks_lda(self):
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
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.kernel()
        nr = scf.newton(mf)
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_uks(self):
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
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_uks_fast_newton(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''C 0 0 0
            H  1  1  1
            H -1 -1  1
            H -1  1 -1
            H  1 -1 -1''',
            basis = '6-31g',
            charge = 1,
            spin = 1,
            symmetry = 1,
        )
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf1 = scf.fast_newton(mf)
        self.assertAlmostEqual(mf1.e_tot, -39.69608384046235, 9)

        mf1 = scf.fast_newton(dft.UKS(mol))
        self.assertAlmostEqual(mf1.e_tot, -39.330377813428001, 9)

    def test_nr_rks_fast_newton(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''C 0 0 0
            H  1  1  1
            H -1 -1  1
            H -1  1 -1
            H  1 -1 -1''',
            basis = '6-31g',
            symmetry = 1,
        )
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf1 = scf.fast_newton(mf)
        self.assertAlmostEqual(mf1.e_tot, -40.10277421254213, 9)

    def test_nr_rohf_fast_newton(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''C 0 0 0
            H  1  1  1
            H -1 -1  1
            H -1  1 -1
            H  1 -1 -1''',
            basis = '6-31g',
            charge = 1,
            spin = 1,
            symmetry = 1,
        )
        mf = scf.ROHF(mol)
        mf1 = scf.fast_newton(mf)
        self.assertAlmostEqual(mf1.e_tot, -39.365972147397649, 9)

    def test_uks_gen_g_hop(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = '6-31g')

        mf = dft.UKS(mol)
        mf.grids.build()
        mf.xc = 'b3p86'
        nao = mol.nao_nr()
        numpy.random.seed(1)
        mo =(numpy.random.random((nao,nao)),
             numpy.random.random((nao,nao)))
        mo_occ = numpy.zeros((2,nao))
        mo_occ[:,:5] = 1
        nocc, nvir = 5, nao-5
        dm1 = numpy.random.random(nvir*nocc*2)
        nr = scf.newton(mf)
        g, hop, hdiag = nr.gen_g_hop(mo, mo_occ, (mf.get_hcore(),)*2)
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 35550.357570127475, 7)


if __name__ == "__main__":
    print("Full Tests for Newton solver")
    unittest.main()

