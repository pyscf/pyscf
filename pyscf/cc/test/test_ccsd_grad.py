#!/usr/bin/env python
from functools import reduce
import unittest
import copy
import numpy
import numpy as np

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf import ao2mo
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
    def test_IX_intermediates(self):
        mol = gto.M()
        mf = scf.RHF(mol)
        mycc = cc.ccsd.CCSD(mf)
        numpy.random.seed(2)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = numpy.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*20
        t1 = numpy.random.random((nocc,nvir))
        t2 = numpy.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = numpy.random.random((nocc,nvir))
        l2 = numpy.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)

        h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                    - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))
        eris = lambda:None
        idx = numpy.tril_indices(nvir)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:][idx[0],idx[1]][:,idx[0],idx[1]].copy()
        eris.fock = fock0

        Ioo, Ivv, Ivo, Xvo = ccsd_grad.IX_intermediates(mycc, t1, t2, l1, l2, eris)
        numpy.random.seed(1)
        h1 = numpy.random.random((nmo,nmo))
        h1 = h1 + h1.T
        self.assertAlmostEqual(numpy.einsum('ij,ij', h1[:nocc,:nocc], Ioo), 2613213.0346526774, 7)
        self.assertAlmostEqual(numpy.einsum('ab,ab', h1[nocc:,nocc:], Ivv), 6873038.9907923322, 7)
        self.assertAlmostEqual(numpy.einsum('ai,ai', h1[nocc:,:nocc], Ivo), 4353360.4241635408, 7)
        self.assertAlmostEqual(numpy.einsum('ai,ai', h1[nocc:,:nocc], Xvo), 203575.42337558540, 7)
        dm1 = ccsd_grad.response_dm1(mycc, t1, t2, l1, l2, eris)
        self.assertAlmostEqual(numpy.einsum('pq,pq', h1[nocc:,:nocc], dm1[nocc:,:nocc]), -486.638981725713393, 7)

        fd2intermediate = lib.H5TmpFile()
        d2 = cc.ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
        mo_coeff = numpy.random.random((nmo,nmo)) - .5
        dm2 = ccsd_grad._rdm2_mo2ao(mycc, d2, mo_coeff)
        self.assertAlmostEqual(lib.finger(dm2), -2279.6732000822421, 9)

    def test_ccsd_grad(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), -0.036999389889460096, 7)

        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mf0 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc0 = cc.ccsd.CCSD(mf0).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        mf1 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc1= cc.ccsd.CCSD(mf1).run(conv_tol=1e-10)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mf2 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc2 = cc.ccsd.CCSD(mf2).run(conv_tol=1e-10)
        l1, l2 = mycc2.solve_lambda()
        g1 = ccsd_grad.kernel(mycc2, mycc2.t1, mycc2.t2, l1, l2)
        self.assertAlmostEqual(g1[0,2], (mycc1.e_tot-mycc0.e_tot)*500, 6)

    def test_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.max_memory = 1
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), 0.10599503839207361, 6)

    def test_frozen_eris(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        eris1 = ccsd_grad._make_frozen_orbital_eris(mycc)
        mycc.max_memory = 1
        eris2 = ccsd_grad._make_frozen_orbital_eris(mycc)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.fooo) - eris1.fooo).max(), 0, 9)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.fvoo) - eris1.fvoo).max(), 0, 9)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.voof) - eris1.voof).max(), 0, 9)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.vovf) - eris1.vovf).max(), 0, 9)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.vvof) - eris1.vvof).max(), 0, 9)
        self.assertAlmostEqual(abs(numpy.asarray(eris2.vvvf) - eris1.vvvf).max(), 0, 9)


if __name__ == "__main__":
    print("Tests for CCSD gradients")
    unittest.main()

