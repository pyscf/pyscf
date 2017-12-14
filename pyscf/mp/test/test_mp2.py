#!/usr/bin/env python
import unittest
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp

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
mf.conv_tol = 1e-14
ehf = mf.scf()


class KnownValues(unittest.TestCase):
    def test_mp2(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,3,1)

        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.204019967288338, 9)

    def test_mp2_outcore(self):
        pt = mp.mp2.MP2(mf)
        pt.max_memory = 1
        e, t2 = pt.kernel()
        self.assertAlmostEqual(e, -0.20401996728747132, 9)
        self.assertAlmostEqual(numpy.linalg.norm(t2), 0.19379397642098622, 9)

    def test_mp2_dm(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,3,1)

        pt = mp.mp2.MP2(mf)
        emp2, t2 = pt.kernel()

        t2s = numpy.zeros((nocc*2,nocc*2,nvir*2,nvir*2))
        t2s[ ::2, ::2, ::2, ::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
        t2s[1::2,1::2,1::2,1::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
        t2s[ ::2,1::2,1::2, ::2] = t2ref0
        t2s[1::2, ::2, ::2,1::2] = t2ref0
        t2s[ ::2,1::2, ::2,1::2] = -t2ref0.transpose(0,1,3,2)
        t2s[1::2, ::2,1::2, ::2] = -t2ref0.transpose(0,1,3,2)
        dm1occ =-.5 * numpy.einsum('ikab,jkab->ij', t2s, t2s)
        dm1vir = .5 * numpy.einsum('ijac,ijbc->ab', t2s, t2s)
        dm1ref = numpy.zeros((nmo,nmo))
        dm1ref[:nocc,:nocc] = dm1occ[ ::2, ::2]+dm1occ[1::2,1::2]
        dm1ref[nocc:,nocc:] = dm1vir[ ::2, ::2]+dm1vir[1::2,1::2]
        for i in range(nocc):
            dm1ref[i,i] += 2
        dm1refao = reduce(numpy.dot, (mf.mo_coeff, dm1ref, mf.mo_coeff.T))
        rdm1 = mp.mp2.make_rdm1_ao(pt, mf.mo_energy, mf.mo_coeff)
        self.assertTrue(numpy.allclose(rdm1, dm1refao))
        self.assertTrue(numpy.allclose(pt.make_rdm1(), dm1ref))

        dm2ref = numpy.zeros((nmo*2,)*4)
        dm2ref[:nocc*2,nocc*2:,:nocc*2,nocc*2:] = t2s.transpose(0,3,1,2) * .5
        dm2ref[nocc*2:,:nocc*2,nocc*2:,:nocc*2] = t2s.transpose(3,0,2,1) * .5
        dm2ref = dm2ref[ ::2, ::2, ::2, ::2] + dm2ref[1::2,1::2,1::2,1::2] \
               + dm2ref[ ::2, ::2,1::2,1::2] + dm2ref[1::2,1::2, ::2, ::2]
        eris = ao2mo.restore(1, ao2mo.full(mf._eri, mf.mo_coeff), mf.mo_coeff.shape[1])
        self.assertAlmostEqual(numpy.einsum('iajb,iajb', eris, dm2ref)*.5, emp2, 9)
        for i in range(nocc):
            for j in range(nocc):
                dm2ref[i,i,j,j] += 4
                dm2ref[i,j,j,i] -= 2
        self.assertTrue(numpy.allclose(pt.make_rdm2(), dm2ref))

    def test_mp2_with_df(self):
        pt = mp.mp2.MP2(mf.density_fit('weigend'))
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.20425449198401671, 9)

        pt = mp.dfmp2.DFMP2(mf.density_fit('weigend'))
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.20425449198401671, 9)

    def test_mp2_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.14660835345250667, 9)

    def test_mp2_outcore_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.max_memory = 0
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.14660835345250667, 9)

    def test_mp2_ao2mo_ovov(self):
        pt = mp.mp2.MP2(mf)
        orbo = mf.mo_coeff[:,:8]
        orbv = mf.mo_coeff[:,8:]
        ftmp = lib.H5TmpFile()
        h5dat = mp.mp2._ao2mo_ovov(pt, orbo, orbv, ftmp, 1)
        ovov = numpy.asarray(h5dat)
        ovov_ref = ao2mo.general(mf._eri, (orbo,orbv,orbo,orbv))
        self.assertAlmostEqual(numpy.linalg.norm(ovov_ref-ovov), 0, 9)



if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()

