#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom = [
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C1", ( 0.73685281,  0.61123287, -0.00800148)],
    ["C2", ( 1.43439081,  1.81898387, -0.00800148)],
    ["C3", ( 0.73673681,  3.02749287, -0.00920048)],
    ["C4", (-0.65808819,  3.02741487, -0.00967948)],
    ["C5", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],
    ["H", ( 2.53407081,  1.81906387, -0.00736748)],
    ["H", ( 1.28693681,  3.97963587, -0.00925948)],
    ["H", (-1.20821019,  3.97969587, -0.01063248)],
    ["H", (-2.45529319,  1.81939187, -0.00886348)],]

mol.basis = {'H': 'cc-pvdz',
             'C1': 'CC PVDZ',
             'C2': 'CC PVDZ',
             'C3': 'cc-pVDZ',
             'C4': gto.basis.parse('''
#BASIS SET: (9s,4p,1d) -> [3s,2p,1d]
C    S
   6665.0000000              0.0006920             -0.0001460        
   1000.0000000              0.0053290             -0.0011540        
    228.0000000              0.0270770             -0.0057250        
     64.7100000              0.1017180             -0.0233120        
     21.0600000              0.2747400             -0.0639550        
      7.4950000              0.4485640             -0.1499810        
      2.7970000              0.2850740             -0.1272620        
      0.5215000              0.0152040              0.5445290        
C    S
      0.1596000              1.0000000        
C    P
      9.4390000              0.0381090        
      2.0020000              0.2094800        
      0.5456000              0.5085570        
C    P
      0.1517000              1.0000000        
C    D
      0.5500000              1.0000000        '''),
             'C': 'CC PVDZ',}
mol.build()

def finger(mat):
    return abs(mat).sum()


class KnowValues(unittest.TestCase):
    def test_intor_nr(self):
        s = mol.intor('cint1e_ovlp_sph')
        self.assertAlmostEqual(finger(s), 622.29059965181796, 11)

    def test_intor_nr1(self):
        s = mol.intor_symmetric('cint1e_ovlp_sph')
        self.assertAlmostEqual(finger(s), 622.29059965181796, 11)

    def test_intor_nr2(self):
        s = mol.intor_asymmetric('cint1e_ovlp_sph')
        self.assertAlmostEqual(finger(s), 622.29059965181796, 11)

    def test_intor_nr_cross(self):
        s = mol.intor('cint1e_ovlp_sph', shls_slice=(0,mol.nbas//4,mol.nbas//4,mol.nbas))
        self.assertAlmostEqual(finger(s), 99.38188078749701, 11)

    def test_intor_r(self):
        s = mol.intor('cint1e_ovlp')
        self.assertAlmostEqual(finger(s), 1592.2297864313475, 11)

    def test_intor_r1(self):
        s = mol.intor_symmetric('cint1e_ovlp')
        self.assertAlmostEqual(finger(s), 1592.2297864313475, 11)

    def test_intor_r2(self):
        s = mol.intor_asymmetric('cint1e_ovlp')
        self.assertAlmostEqual(finger(s), 1592.2297864313475, 11)

    def test_intor_r_comp(self):
        s = mol.intor('cint1e_ipkin', comp=3)
        self.assertAlmostEqual(finger(s), 4409.86758420756, 11)

    def test_intor_nr2e(self):
        mol1 = gto.M(atom=[["O" , (0. , 0.     , 0.)],
                           [1   , (0. , -0.757 , 0.587)],
                           [1   , (0. , 0.757  , 0.587)]],
                     basis = '631g')
        nao = mol1.nao_nr()
        eri0 = numpy.empty((3,nao,nao,nao,nao))
        ip = 0
        for i in range(mol1.nbas):
            jp = 0
            for j in range(mol1.nbas):
                kp = 0
                for k in range(mol1.nbas):
                    lp = 0
                    for l in range(mol1.nbas):
                        buf = mol1.intor_by_shell('cint2e_ip1_sph', (i,j,k,l), 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri0[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di

        eri1 = mol1.intor('cint2e_ip1_sph', comp=3).reshape(3,13,13,13,13)
        self.assertTrue(numpy.allclose(eri0, eri1))

        idx = numpy.tril_indices(13)
        naopair = nao * (nao+1) // 2
        ref = eri0[:,idx[0],idx[1]].reshape(3,naopair,-1)
        eri1 = mol1.intor('cint2e_ip1_sph', comp=3, aosym='s2ij')
        self.assertTrue(numpy.allclose(ref, eri1))

        idx = numpy.tril_indices(13)
        ref = eri0[:,:,:,idx[0],idx[1]].reshape(3,-1,naopair)
        eri1 = mol1.intor('cint2e_ip1_sph', comp=3, aosym='s2kl')
        self.assertTrue(numpy.allclose(ref, eri1))

        idx = numpy.tril_indices(13)
        ref = eri0[:,idx[0],idx[1]][:,:,idx[0],idx[1]].reshape(3,-1,naopair)
        eri1 = mol1.intor('cint2e_ip1_sph', comp=3, aosym='s4')
        self.assertTrue(numpy.allclose(ref, eri1))

    def test_rinv_with_zeta(self):
        mol.set_rinv_orig_((.2,.3,.4))
        mol.set_rinv_zeta_(2.2)
        v1 = mol.intor('cint1e_rinv_sph')
        mol.set_rinv_zeta_(0)
        pmol = gto.M(atom='Ghost .2 .3 .4', unit='b', basis={'Ghost':[[0,(2.2*.5, 1)]]})
        pmol._atm, pmol._bas, pmol._env = \
            gto.conc_env(mol._atm, mol._bas, mol._env,
                         pmol._atm, pmol._bas, pmol._env)
        pmol.natm = len(pmol._atm)
        pmol.nbas = len(pmol._bas)
        shls_slice=[pmol.nbas-1,pmol.nbas,pmol.nbas-1,pmol.nbas]
        v0 = pmol.intor('cint2e_sph', shls_slice=shls_slice)
        nao = pmol.nao_nr()
        v0 = v0.reshape(nao,nao)[:-1,:-1]
        self.assertTrue(numpy.allclose(v0, v1))



if __name__ == "__main__":
    unittest.main()
