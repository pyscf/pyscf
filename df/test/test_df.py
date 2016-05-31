#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df
from pyscf.df import _ri

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
atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                             auxmol._atm, auxmol._bas, auxmol._env)

class KnowValues(unittest.TestCase):
    def test_aux_e2(self):
        nao = mol.nao_nr()
        naoaux = auxmol.nao_nr()
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

        j3c = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
        self.assertTrue(numpy.allclose(eri0, j3c.reshape(nao,nao,naoaux)))

        idx = numpy.tril_indices(nao)
        j3c = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(eri0[idx], j3c))

    def test_aux_e2_diff_bra_ket(self):
        mol1 = mol.copy()
        mol1.basis = 'sto3g'
        mol1.build(0, 0, verbose=0)
        atm1, bas1, env1 = gto.conc_env(atm, bas, env,
                                        mol1._atm, mol1._bas, mol1._env)
        ao_loc = gto.moleintor.make_loc(bas1, 'cint3c2e_sph')
        shls_slice = (0, mol.nbas,
                      mol.nbas+auxmol.nbas, mol.nbas+auxmol.nbas+mol1.nbas,
                      mol.nbas, mol.nbas+auxmol.nbas)
        j3c = _ri.nr_auxe2('cint3c2e_sph', atm1, bas1, env1, shls_slice,
                           ao_loc, 's1', 1)

        nao = mol.nao_nr()
        naoj = mol1.nao_nr()
        naoaux = auxmol.nao_nr()
        eri0 = numpy.empty((nao,naoj,naoaux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas+auxmol.nbas, len(bas1)):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('cint3c2e_sph',
                                                         shls, atm1, bas1, env1)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0.reshape(-1,naoaux), j3c))

    def test_cholesky_eri(self):
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
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

        df.outcore.cholesky_eri(mol, ftmp.name, ioblk_size=.05)
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

        nao = mol.nao_nr()
        naux = cderi0.shape[0]
        df.outcore.general(mol, (numpy.eye(nao),)*2, ftmp.name,
                           max_memory=.05, ioblk_size=.02)
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

        ####
        buf = numpy.zeros((naux,nao,nao))
        idx = numpy.tril_indices(nao)
        buf[:,idx[0],idx[1]] = cderi0
        buf[:,idx[1],idx[0]] = cderi0
        cderi0 = buf
        df.outcore.cholesky_eri(mol, ftmp.name, aosym='s1', ioblk_size=.05)
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0.reshape(naux,-1)))

        numpy.random.seed(1)
        co = numpy.random.random((nao,4))
        cv = numpy.random.random((nao,25))
        cderi0 = numpy.einsum('kpq,pi,qj->kij', cderi0, co, cv)
        df.outcore.general(mol, (co,cv), ftmp.name, ioblk_size=.05)
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0.reshape(naux,-1)))

        cderi0 = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_ip1_sph',
                                  aosym='s1', comp=3)
        j2c = df.incore.fill_2c2e(mol, auxmol)
        low = scipy.linalg.cholesky(j2c, lower=True)
        cderi0 = [scipy.linalg.solve_triangular(low, j3c.T, lower=True)
                  for j3c in cderi0]
        nao = mol.nao_nr()
        df.outcore.general(mol, (numpy.eye(nao),)*2, ftmp.name,
                           int3c='cint3c2e_ip1_sph', aosym='s1', int2c='cint2c2e_sph',
                           comp=3, max_memory=.05, ioblk_size=.02)
        with h5py.File(ftmp.name) as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

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

    def test_ao2mo(self):
        dfobj = df.DF(mol)
        dfobj.build()
        cderi = dfobj._cderi

        nao = mol.nao_nr()
        eri0 = ao2mo.restore(8, numpy.dot(cderi.T, cderi), nao)
        numpy.random.seed(1)
        mos = numpy.random.random((nao,nao*10))
        mos = (mos[:,:5], mos[:,5:11], mos[:,3:9], mos[:,2:4])
        mo_eri0 = ao2mo.kernel(eri0, mos)

        mo_eri1 = dfobj.ao2mo(mos)
        self.assertTrue(numpy.allclose(mo_eri0, mo_eri1))


if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

