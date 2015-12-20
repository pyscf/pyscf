#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.dhf.UHF(mol)


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = mf.init_guess_by_minao()
        self.assertAlmostEqual(abs(dm).sum(), 14.899439258242364, 9)

    def test_get_hcore(self):
        h = mf.get_hcore()
        self.assertAlmostEqual(numpy.linalg.norm(h), 159.55593668675903, 9)

    def test_get_ovlp(self):
        s = mf.get_ovlp()
        self.assertAlmostEqual(numpy.linalg.norm(s), 9.0156256929936056, 9)

    def test_1e(self):
        mf = scf.dhf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.892132873081664, 9)

#    def test_analyze(self):
#        numpy.random.seed(1)
#        pop, chg = mf.analyze()
#        self.assertAlmostEqual(numpy.linalg.norm(pop), 2.0355530265140636, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.081567943868265, 9)

    def test_rhf(self):
        mf = scf.dhf.RHF(mol)
        mf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(mf.scf(), -76.081567943868265, 9)

    def test_get_veff(self):
        dm = mf.make_rdm1()
        v = mf.get_veff(mol, dm)
        self.assertAlmostEqual(numpy.linalg.norm(v), 56.050204183850624, 9)

    def test_gaunt(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
                H     0    0        1
                H     1    1        0
                H     0    -0.757   0.587
                H     0    0.757    0.587''',
            basis = 'cc-pvdz',
        )
        n2c = mol.nao_2c()
        n4c = n2c * 2
        #eri0 = numpy.empty((n2c,n2c,n2c,n2c), dtype=numpy.complex)
        eri1 = numpy.empty((n2c,n2c,n2c,n2c), dtype=numpy.complex)
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    lp = 0
                    for l in range(mol.nbas):
                        #buf = mol.intor_by_shell('cint2e_ssp1sps2', (i,j,k,l))
                        #di, dj, dk, dl = buf.shape
                        #eri0[ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf

                        buf = mol.intor_by_shell('cint2e_ssp1ssp2', (i,j,k,l))
                        di, dj, dk, dl = buf.shape
                        eri1[ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di

        erig = numpy.empty((n4c,n4c,n4c,n4c), dtype=numpy.complex)
        tao = numpy.asarray(mol.time_reversal_map())
        idx = abs(tao)-1 # -1 for C indexing convention
        sign_mask = tao<0

        erig[:n2c,n2c:,:n2c,n2c:] = eri1 # ssp1ssp2

        eri2 = eri1.take(idx,axis=0).take(idx,axis=1) # sps1ssp2
        eri2[sign_mask,:] *= -1
        eri2[:,sign_mask] *= -1
        eri2 = -eri2.transpose(1,0,2,3)
        erig[n2c:,:n2c,:n2c,n2c:] = eri2

        eri2 = eri1.take(idx,axis=2).take(idx,axis=3) # ssp1sps2
        eri2[:,:,sign_mask,:] *= -1
        eri2[:,:,:,sign_mask] *= -1
        eri2 = -eri2.transpose(0,1,3,2)
        #self.assertTrue(numpy.allclose(eri0, eri2))
        erig[:n2c,n2c:,n2c:,:n2c] = eri2

        eri2 = eri1.take(idx,axis=0).take(idx,axis=1)
        eri2 = eri1.take(idx,axis=2).take(idx,axis=3) # sps1sps2
        eri2 = eri2.transpose(1,0,2,3)
        eri2 = eri2.transpose(0,1,3,2)
        eri2[sign_mask,:] *= -1
        eri2[:,sign_mask] *= -1
        eri2[:,:,sign_mask,:] *= -1
        eri2[:,:,:,sign_mask] *= -1
        erig[n2c:,:n2c,n2c:,:n2c] = eri2

        dm = numpy.zeros((n4c,n4c), dtype=numpy.complex)
        dm[:n2c,:n2c] = dm[n2c:,n2c:] = numpy.linalg.inv(mol.intor_symmetric('cint1e_ovlp'))
        c1 = .5/mol.light_speed
        vj0 = numpy.einsum('ijkl,lk->ij', erig, dm) * c1**2
        vk0 = numpy.einsum('ijkl,jk->il', erig, dm) * c1**2

        vj1, vk1 = scf.dhf._call_veff_gaunt(mol, dm)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))

    def test_time_rev_matrix(self):
        s = mol.intor_symmetric('cint1e_ovlp')
        ts = scf.dhf.time_reversal_matrix(mol, s)
        self.assertTrue(numpy.allclose(s, ts))


if __name__ == "__main__":
    print("Full Tests for dhf")
    unittest.main()

