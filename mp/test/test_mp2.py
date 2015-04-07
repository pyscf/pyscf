#!/usr/bin/env python
import unittest
from functools import reduce
import numpy
from pyscf import scf
from pyscf import gto
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
mf.scf()


class KnowValues(unittest.TestCase):
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
        self.assertAlmostEqual(emp2, -0.204019967288338, 11)
        self.assertTrue(numpy.allclose(t2, t2ref0))

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
        dm1ref = reduce(numpy.dot, (mf.mo_coeff, dm1ref, mf.mo_coeff.T))

        rdm1 = mp.mp2.make_rdm1(pt, mf.mo_energy, mf.mo_coeff, nocc)
        self.assertTrue(numpy.allclose(rdm1, dm1ref))

    def test_mp2_outcore(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = None
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.RHF(mol)
        mf.max_memory = 0
        mf.scf()
        pt = mp.mp2.MP2(mf)
        e, t2 = pt.kernel()
        self.assertAlmostEqual(e, -0.20401996728747132, 11)
        self.assertAlmostEqual(numpy.linalg.norm(t2), 0.19379397642098622, 9)



if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()

