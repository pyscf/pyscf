#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import ctypes
import numpy
from pyscf import gto
from pyscf import df

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''O     0    0.       0.
              1     0    -0.757   0.587
              1     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

libri = df.incore.libri
auxmol = df.incore.format_aux_basis(mol)
atm, bas, env = \
        gto.conc_env(mol._atm, mol._bas, mol._env,
                     auxmol._atm, auxmol._bas, auxmol._env)

class KnowValues(unittest.TestCase):
    def test_aux_e2(self):
        j3c = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
        nao = mol.nao_nr()
        naoaux = auxmol.nao_nr()
        j3c = j3c.reshape(nao,nao,naoaux)

        eri0 = numpy.empty((nao,nao,naoaux))
        libri.CINTcgto_spheric.restype = ctypes.c_int
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('cint3c2e_sph',
                                                         shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j3c))

    def test_2c2e(self):
        j2c = df.incore.fill_2c2e(mol, auxmol)
        eri0 = numpy.empty_like(j2c)
        pi = 0
        for i in range(mol.nbas, len(bas)):
            pj = 0
            for j in range(mol.nbas, len(bas)):
                shls = (i, j)
                buf = gto.moleintor.getints_by_shell('cint2c2e_sph',
                                                     shls, atm, bas, env)
                di, dj = buf.shape
                eri0[pi:pi+di,pj:pj+dj] = buf
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j2c))

        j3c = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s2ij')
        cderi = df.incore.cholesky_eri(mol)
        eri0 = numpy.einsum('pi,pk->ik', cderi, cderi)
        eri1 = numpy.einsum('ik,kl->il', j3c, numpy.linalg.inv(j2c))
        eri1 = numpy.einsum('ip,kp->ik', eri1, j3c)
        self.assertTrue(numpy.allclose(eri1, eri0))


if __name__ == "__main__":
    print("Full Tests for df.incore")
    unittest.main()

