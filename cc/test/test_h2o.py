#!/usr/bin/env python
import unittest
import numpy

from pyscf import gto, lib
from pyscf import scf
from pyscf import cc

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = {'H': 'cc-pvdz',
             'O': 'cc-pvdz',}
mol.build()
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
ehf = mf.kernel()


class KnowValues(unittest.TestCase):
    def test_ccsd(self):
        mcc = cc.ccsd.CC(mf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        eris = mcc.ao2mo()
        emp2, t1, t2 = mcc.init_amps(eris)
        self.assertAlmostEqual(abs(t2).sum(), 4.9556571211255909, 6)
        self.assertAlmostEqual(emp2, -0.2040199672883385, 10)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(abs(t1).sum(), 0.0475038989126, 8)
        self.assertAlmostEqual(abs(t2).sum(), 5.4018238455030, 6)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.208967840546667, 10)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.212173678670510, 10)
        self.assertAlmostEqual(abs(t1).sum(), 0.05470123093500083, 8)
        self.assertAlmostEqual(abs(t2).sum(), 5.5605208386554716, 6)

        mcc.kernel()
        self.assertTrue(numpy.allclose(mcc.t2,mcc.t2.transpose(1,0,3,2)))
        self.assertAlmostEqual(mcc.ecc, -0.2133432312951, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 5.63970279799556984, 6)

        nocc, nvir = t1.shape
        tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        ovvv = lib.unpack_tril(eris.ovvv).reshape(nocc,nvir,nvir,nvir)
        tmp = -numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, ovvv)
        t2a = tmp + tmp.transpose(1,0,3,2)
        t2a = t2a[numpy.tril_indices(nocc)]
        t2a += mcc.add_wvvVV(t1, t2, eris)
        mcc.direct = True
        t2b = mcc.add_wvvVV(t1, t2, eris)
        self.assertTrue(numpy.allclose(t2a,t2b))

    def test_ccsd_frozen(self):
        mcc = cc.ccsd.CC(mf, frozen=range(1))
        mcc.conv_tol = 1e-10
        mcc.kernel()
        self.assertAlmostEqual(mcc.ecc, -0.21124878189922872, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 5.4996425901189347, 6)

    def test_h2o_non_hf_orbital(self):
        nmo = mf.mo_energy.size
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        u = numpy.eye(nmo)
        numpy.random.seed(1)
        u[:nocc,:nocc] = numpy.linalg.svd(numpy.random.random((nocc,nocc)))[0]
        u[nocc:,nocc:] = numpy.linalg.svd(numpy.random.random((nvir,nvir)))[0]
        mycc = cc.ccsd.CCSD(mf)
        mycc.conv_tol = 1e-12
        mycc.diis_start_energy_diff = 1e2
        mycc.max_cycle = 1000
        mycc.mo_coeff = mo_coeff=numpy.dot(mf.mo_coeff,u)
        self.assertAlmostEqual(mycc.kernel()[0], -0.21334323320620596, 8)

## FIXME
#    def test_h2o_without_scf(self):
#        mycc = cc.ccsd.CCSD(mf)
#        nmo = mf.mo_energy.size
#        nocc = mol.nelectron // 2
#        nvir = nmo - nocc
#        numpy.random.seed(1)
#        u = numpy.eye(nmo) + numpy.random.random((nmo,nmo))*.2
#        u, w, vh = numpy.linalg.svd(u)
#        u = numpy.dot(u, vh)
#
#        mo1 = numpy.dot(mf.mo_coeff, u)
#        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
#
#        mycc.diis_start_energy_diff = 1e2
#        mycc.max_cycle = 1000
#        mycc.conv_tol = 1e-12
#        mycc.mo_coeff = mo1
#        self.assertAlmostEqual(mf.energy_tot(dm1)+mycc.kernel()[0],
#                               ehf-0.21334323320620596, 8)

    def test_ccsd_lambda(self):
        mcc = cc.ccsd.CC(mf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        mcc.kernel()
        mcc.solve_lambda()
        self.assertAlmostEqual(numpy.linalg.norm(mcc.l1), 0.01326267012100099, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mcc.l2), 0.21257559872380857, 7)

    def test_ccsd_rdm(self):
        mcc = cc.ccsd.CC(mf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        mcc.kernel()
        mcc.solve_lambda()
        dm1 = mcc.make_rdm1()
        dm2 = mcc.make_rdm2()
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 4.4227836730016374, 7)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 20.074629443311355, 7)

if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()

