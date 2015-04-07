#!/usr/bin/env python

import os
import ctypes
import _ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

libcvhf1 = lib.load_library('libcvhf')

mol = gto.Mole()
mol.verbose = 0
mol.output = None#'out_h2o'
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = {'H': 'cc-pvdz',
             'O': 'cc-pvdz',}

#mol.atom = [
#    [1   , (0. , -0.757 , 0.587)],
#    [1   , (0. , 0.757  , 0.587)] ]
#
#mol.basis = {'H': 'sto-3g',}
mol.build()
rhf = scf.RHF(mol)
rhf.scf()


nao = mol.nao_nr()
npair = nao*(nao+1)//2
c_atm = numpy.array(mol._atm, dtype=numpy.int32)
c_bas = numpy.array(mol._bas, dtype=numpy.int32)
c_env = numpy.array(mol._env)
natm = ctypes.c_int(c_atm.shape[0])
nbas = ctypes.c_int(c_bas.shape[0])
cintopt = ctypes.c_void_p()
vhfopt  = ctypes.c_void_p()
# for each dm1, call namejk
def runjk(dm1, ncomp, intorname, unpackname, filldot, *namejk):
    fdrv = getattr(libcvhf1, 'CVHFnr_direct_drv')
    intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf1._handle,
                                          intorname))
    funpack = ctypes.c_void_p(_ctypes.dlsym(libcvhf1._handle,
                                            unpackname))
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf1._handle, filldot))

    njk = len(namejk)
    if dm1.ndim == 2:
        n_dm = 1
        dm1 = (dm1,)
    else:
        n_dm = dm1.shape[0]

    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dms = (ctypes.c_void_p*(njk*n_dm))()
    for i, symb in enumerate(namejk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf1._handle, symb))
        for j in range(n_dm):
            dms[i*n_dm+j] = dm1[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.zeros((njk,n_dm*ncomp,nao,nao))

    fdrv(intor, fdot, funpack, fjk, dms,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         cintopt, vhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))
    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    return vjk


class KnowValues(unittest.TestCase):
    def test_direct_jk(self):
        numpy.random.seed(15)

        dm1 = numpy.random.random((nao,nao))
        dm1 = dm1 + dm1.T
        vj0, vk0 = scf._vhf.incore(rhf._eri, dm1, 1)
        vj1, vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                         'CVHFfill_dot_nrs8',
                         'CVHFnrs8_ij_s2kl', 'CVHFnrs8_jk_s2il')
        vj1 = lib.hermi_triu(vj1, 1)
        vk1 = lib.hermi_triu(vk1, 1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        dm1 = numpy.array((dm1,dm1))
        vj1, vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                         'CVHFfill_dot_nrs8',
                         'CVHFnrs8_ij_s2kl', 'CVHFnrs8_jk_s2il')
        vj1[0] = lib.hermi_triu(vj1[0], 1)
        vk1[0] = lib.hermi_triu(vk1[0], 1)
        vj1[1] = lib.hermi_triu(vj1[1], 1)
        vk1[1] = lib.hermi_triu(vk1[1], 1)
        self.assertTrue(numpy.allclose(vj0,vj1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vj1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))

        dm1 = numpy.random.random((nao,nao))
        eri1 = ao2mo.restore(1, rhf._eri, nao)
        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vj1, vj2 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                         'CVHFfill_dot_nrs4',
                         'CVHFnrs4_ij_s2kl', 'CVHFnrs4_kl_s2ij')
        vj1 = lib.hermi_triu(vj1.copy(), 1)
        vj2 = lib.hermi_triu(vj2.copy(), 1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vj0,vj2))

        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                    'CVHFfill_dot_nrs4',
                    'CVHFnrs4_il_s1jk', 'CVHFnrs4_jk_s1il')
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))

        dm1 = dm1 + dm1.T
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                    'CVHFfill_dot_nrs4',
                    'CVHFnrs4_il_s1jk', 'CVHFnrs4_jk_s1il',
                    'CVHFnrs4_il_s2jk', 'CVHFnrs4_jk_s2il')
        vk1[2] = lib.hermi_triu(vk1[2].copy())
        vk1[3] = lib.hermi_triu(vk1[3].copy())
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        dm1 = numpy.random.random((nao,nao))
        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2rect',
                    'CVHFfill_dot_nrs2kl',
                    'CVHFnrs2ij_ij_s1kl', 'CVHFnrs2ij_kl_s2ij',
                    'CVHFnrs2ij_jk_s1il', 'CVHFnrs2ij_il_s1jk')
        vk1[1] = lib.hermi_triu(vk1[1].copy())
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2trilu',
                    'CVHFfill_dot_nrs2kl',
                    'CVHFnrs2ij_ij_s1kl', 'CVHFnrs2ij_kl_s2ij',
                    'CVHFnrs2ij_jk_s1il', 'CVHFnrs2ij_il_s1jk')
        vk1[1] = lib.hermi_triu(vk1[1].copy())
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2tril',
                    'CVHFfill_dot_nrs2ij',
                    'CVHFnrs2kl_ij_s2kl', 'CVHFnrs2kl_kl_s1ij',
                    'CVHFnrs2kl_jk_s1il', 'CVHFnrs2kl_il_s1jk')
        vk1[0] = lib.hermi_triu(vk1[0].copy())
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjk(dm1, 1, 'cint2e_sph', 'CVHFunpack_nrblock2rect',
                    'CVHFfill_dot_nrs1',
                    'CVHFnrs1_ij_s1kl', 'CVHFnrs1_kl_s1ij',
                    'CVHFnrs1_jk_s1il', 'CVHFnrs1_il_s1jk')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))


if __name__ == '__main__':
    print('Full Tests for nrvhf')
    unittest.main()

