#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
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

    def test_outcore(self):
        ftmp = tempfile.NamedTemporaryFile()
        cderi0 = df.incore.cholesky_eri(mol)
        df.outcore.cholesky_eri(mol, ftmp.name)
        cderi1 = df.load_buf(ftmp.name, 0, 1000)
        self.assertTrue(numpy.allclose(cderi1, cderi0))

        df.outcore.cholesky_eri(mol, ftmp.name, ioblk_size=.05)
        cderi1 = df.load_buf(ftmp.name, 0, 1000)
        self.assertTrue(numpy.allclose(cderi1, cderi0))

        nao = mol.nao_nr()
        naux = cderi0.shape[0]
        df.outcore.general(mol, (numpy.eye(nao),)*2, ftmp.name,
                           max_memory=.05, ioblk_size=.02)
        feri = h5py.File(ftmp.name)
        cderi1 = numpy.array(feri['eri_mo'])
        feri.close()
        self.assertTrue(numpy.allclose(cderi1, cderi0.T))

        ####
        buf = numpy.zeros((naux,nao,nao))
        idx = numpy.tril_indices(nao)
        buf[:,idx[0],idx[1]] = cderi0
        buf[:,idx[1],idx[0]] = cderi0
        cderi0 = buf
        df.outcore.cholesky_eri(mol, ftmp.name, aosym='s1', ioblk_size=.05)
        cderi1 = df.load_buf(ftmp.name, 0, 1000).reshape(-1,nao,nao)
        self.assertTrue(numpy.allclose(cderi1, cderi0))

        numpy.random.seed(1)
        co = numpy.random.random((nao,4))
        cv = numpy.random.random((nao,25))
        cderi0 = numpy.einsum('kpq,pi,qj->kij', cderi0, co, cv)
        df.outcore.general(mol, (co,cv), ftmp.name, ioblk_size=.05)
        feri = h5py.File(ftmp.name)
        cderi1 = numpy.array(feri['eri_mo'])
        feri.close()
        self.assertTrue(numpy.allclose(cderi1.T, cderi0.reshape(naux,-1)))

    def test_r_incore(self):
        j3c = df.r_incore.aux_e2(mol, auxmol, intor='cint3c2e_spinor', aosym='s1')
        nao = mol.nao_2c()
        naoaux = auxmol.nao_nr()
        j3c = j3c.reshape(nao,nao,naoaux)

        eri0 = numpy.empty((nao,nao,naoaux), dtype=numpy.complex)
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('cint3c2e_spinor',
                                                         shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j3c))
        eri1 = df.r_incore.aux_e2(mol, auxmol, intor='cint3c2e_spinor',
                                  aosym='s2ij')
        for i in range(naoaux):
            j3c[:,:,i] = lib.unpack_tril(eri1[:,i])
        self.assertTrue(numpy.allclose(eri0, j3c))


if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

