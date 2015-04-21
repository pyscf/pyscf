#!/usr/bin/env python
import unittest
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf, cc



class KnowValues(unittest.TestCase):
    def test_h2o_non_hf_orbital(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.basis = 'ccpvdz'
        mol.build()
        mf = scf.RHF(mol)
        mf.scf()
        mycc = cc.ccsd.CCSD(mf)
        mycc.conv_tol = 1e-12
        eref = mycc.kernel()[0]
        self.assertAlmostEqual(eref, -0.21334323320620596, 8)

        nmo = mf.mo_energy.size
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        u = numpy.eye(nmo)
        numpy.random.seed(1)
        u[:nocc,:nocc] = numpy.linalg.svd(numpy.random.random((nocc,nocc)))[0]
        u[nocc:,nocc:] = numpy.linalg.svd(numpy.random.random((nvir,nvir)))[0]
        def my_ao2mo(mo):
            eris = cc.ccsd._ERIS(mycc, numpy.dot(mf.mo_coeff,u))
            eris.fock = reduce(numpy.dot, (u.T, numpy.diag(mf.mo_energy), u))
            return eris
        mycc.ao2mo = my_ao2mo
        mycc.diis_start_energy_diff = 1e2
        mycc.max_cycle = 1000
        self.assertAlmostEqual(mycc.kernel()[0], eref, 8)

    def test_h2_without_scf(self):
        mol = gto.M(verbose = 0,
            atom = [
                [1 , (0. , 0 , 0)],
                [1 , (0. , 0 , 2.5)]],
            basis = 'ccpvdz')
        mf = scf.RHF(mol)
        ehf = mf.kernel()
        mycc = cc.ccsd.CCSD(mf)
        etotref = ehf + mycc.kernel()[0]
        self.assertAlmostEqual(etotref, -1.0031292508300915, 8)

        nmo = mf.mo_energy.size
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        numpy.random.seed(1)
        u = numpy.eye(nmo) + numpy.random.random((nmo,nmo))*.2
        u, w, vh = numpy.linalg.svd(u)
        u = numpy.dot(u, vh)

        mo1 = numpy.dot(mf.mo_coeff, u)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)

        def my_ao2mo(mo):
            eris = cc.ccsd._ERIS(mycc, mo1)
            fock = mf.get_hcore() + mf.get_veff(mol, dm1)
            eris.fock = reduce(numpy.dot, (mo1.T, fock, mo1))
            return eris
        mycc.ao2mo = my_ao2mo
        mycc.diis_start_energy_diff = 1e2
        mycc.max_cycle = 1000
        self.assertAlmostEqual(mf.energy_tot(dm1)+mycc.kernel()[0], etotref, 8)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()

