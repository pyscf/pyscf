#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ctypes
import _ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

libcvhf2 = lib.load_library('libcvhf')

mol = gto.Mole()
mol.verbose = 0
mol.output = None#'out_h2o'
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = 'cc-pvdz'

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
cvhfopt = lib.c_null_ptr()
cintopt = lib.c_null_ptr()
ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int32)
# for each dm1, call namejk
def runjk(dm1, ncomp, intorname, filldot, *namejk):
    fdrv = getattr(libcvhf2, 'CVHFnr_direct_drv')
    intor = getattr(libcvhf2, intorname)
    fdot = getattr(libcvhf2, filldot)

    njk = len(namejk)
    if dm1.ndim == 2:
        n_dm = 1
        dm1 = (dm1,)
    else:
        n_dm = dm1.shape[0]

    vjk = numpy.zeros((njk,n_dm*ncomp,nao,nao))
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dmsptr = (ctypes.c_void_p*(njk*n_dm))()
    vjkptr = (ctypes.c_void_p*(njk*n_dm))()
    for i, symb in enumerate(namejk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf2._handle, symb))
        for j in range(n_dm):
            dmsptr[i*n_dm+j] = dm1[j].ctypes.data_as(ctypes.c_void_p)
            vjkptr[i*n_dm+j] = vjk[i,j*ncomp].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    shls_slice = (ctypes.c_int*8)(*([0, mol.nbas]*4))

    fdrv(intor, fdot, fjk, dmsptr, vjkptr,
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))
    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    return vjk

def runjks2(dm1, ncomp, intorname, filldot, *namejk):
    vjk = runjk(dm1, ncomp, intorname, filldot, *namejk)
    return [lib.hermi_triu(v, 1) for v in vjk]

def makeri(fname, comp):
    nao = mol.nao_nr()
    eri = numpy.empty((comp,nao,nao,nao,nao))
    ip = 0
    for i in range(mol.nbas):
        jp = 0
        for j in range(mol.nbas):
            kp = 0
            for k in range(mol.nbas):
                lp = 0
                for l in range(mol.nbas):
                    buf = gto.moleintor.getints_by_shell(fname, (i,j,k,l), mol._atm,
                                                         mol._bas, mol._env, comp)
                    di,dj,dk,dl = buf.shape[1:]
                    eri[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                    lp += dl
                kp += dk
            jp += dj
        ip += di
    return eri


class KnowValues(unittest.TestCase):
    def test_direct_jk_s1(self):
        numpy.random.seed(15)

        dm1 = numpy.random.random((nao,nao))
        dm1 = dm1 + dm1.T
        vj0, vk0 = scf._vhf.incore(rhf._eri, dm1, 1)
        vj1, vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs8',
                         'CVHFnrs8_ji_s1kl', 'CVHFnrs8_jk_s1il')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        dm1 = numpy.array((dm1,dm1))
        vj1, vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs8',
                         'CVHFnrs8_ji_s1kl', 'CVHFnrs8_jk_s1il')
        self.assertTrue(numpy.allclose(vj0,vj1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vj1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))

        dm1 = numpy.random.random((nao,nao))
        eri1 = ao2mo.restore(1, rhf._eri, nao)
        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vj1, vj2 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                         'CVHFnrs4_ji_s1kl', 'CVHFnrs4_jk_s1il')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vj2))

        vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                    'CVHFnrs4_li_s1kj', 'CVHFnrs4_jk_s1il')
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))

        dm1 = dm1 + dm1.T
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                    'CVHFnrs4_li_s1kj', 'CVHFnrs4_jk_s1il',
                    'CVHFnrs4_li_s1kj', 'CVHFnrs4_jk_s1il')
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        dm1 = numpy.random.random((nao,nao))
        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs2kl',
                    'CVHFnrs2kl_ji_s1kl', 'CVHFnrs2kl_lk_s1ij',
                    'CVHFnrs2kl_jk_s1il', 'CVHFnrs2kl_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs2ij',
                    'CVHFnrs2ij_ji_s1kl', 'CVHFnrs2ij_lk_s1ij',
                    'CVHFnrs2ij_jk_s1il', 'CVHFnrs2ij_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs1',
                    'CVHFnrs1_ji_s1kl', 'CVHFnrs1_lk_s1ij',
                    'CVHFnrs1_jk_s1il', 'CVHFnrs1_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

    def test_direct_jk_ncomp_s1(self):
        numpy.random.seed(15)
        dm1 = numpy.random.random((nao,nao))
        dm1 = dm1 + dm1.T
        eri0 = makeri('int2e_ip1_sph', 3)
        vj0 = numpy.einsum('pijkl,ji->pkl', eri0, dm1)
        vk0 = numpy.einsum('pijkl,li->pkj', eri0, dm1)
        vj1, vk1 = runjk(dm1, 3, 'int2e_ip1_sph', 'CVHFdot_nrs1',
                         'CVHFnrs1_ji_s1kl', 'CVHFnrs1_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        vj1, vk1 = runjk(dm1, 3, 'int2e_ip1_sph', 'CVHFdot_nrs2kl',
                       'CVHFnrs2kl_ji_s1kl', 'CVHFnrs2kl_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        eri0 = makeri('int2e_ig1_sph', 3)
        vj0 = numpy.einsum('pijkl,ji->pkl', eri0, dm1)
        vk0 = numpy.einsum('pijkl,li->pkj', eri0, dm1)
        vj1, vk1 = runjk(dm1, 3, 'int2e_ig1_sph', 'CVHFdot_nrs1',
                         'CVHFnrs1_ji_s1kl', 'CVHFnrs1_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        vj1, vk1 = runjk(dm1, 3, 'int2e_ig1_sph', 'CVHFdot_nrs2kl',
                       'CVHFnrs2kl_ji_s1kl', 'CVHFnrs2kl_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        vj1, vk1 = runjk(dm1, 3, 'int2e_ig1_sph', 'CVHFdot_nrs4',
                       'CVHFnra4ij_ji_s1kl', 'CVHFnra4ij_li_s1kj')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_jk_s2(self):
        numpy.random.seed(15)

        dm1 = numpy.random.random((nao,nao))
        dm1 = dm1 + dm1.T
        vj0, vk0 = scf._vhf.incore(rhf._eri, dm1, 1)
        vj1, vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs8',
                         'CVHFnrs8_ji_s2kl', 'CVHFnrs8_jk_s2il')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        eri1 = ao2mo.restore(1, rhf._eri, nao)
        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vj1, vj2 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                         'CVHFnrs4_ji_s2kl', 'CVHFnrs4_jk_s2il')
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vj2))

        vj1, vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                           'CVHFnrs4_li_s2kj', 'CVHFnrs4_jk_s2il')
        self.assertTrue(numpy.allclose(vk0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs4',
                    'CVHFnrs4_li_s2kj', 'CVHFnrs4_jk_s2il',
                    'CVHFnrs4_li_s2kj', 'CVHFnrs4_jk_s2il')
        self.assertTrue(numpy.allclose(vk0,vk1[0]))
        self.assertTrue(numpy.allclose(vk0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vj0 = numpy.einsum('ijkl,kl->ij', eri1, dm1)
        vk0 = numpy.einsum('ijkl,jk->il', eri1, dm1)
        vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs2kl',
                    'CVHFnrs2kl_ji_s2kl', 'CVHFnrs2kl_lk_s2ij',
                    'CVHFnrs2kl_jk_s2il', 'CVHFnrs2kl_li_s2kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs2ij',
                    'CVHFnrs2ij_ji_s2kl', 'CVHFnrs2ij_lk_s2ij',
                    'CVHFnrs2ij_jk_s2il', 'CVHFnrs2ij_li_s2kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vk1 = runjks2(dm1, 1, 'int2e_sph', 'CVHFdot_nrs1',
                    'CVHFnrs1_ji_s2kl', 'CVHFnrs1_lk_s2ij',
                    'CVHFnrs1_jk_s2il', 'CVHFnrs1_li_s2kj')
        self.assertTrue(numpy.allclose(vj0,vk1[0]))
        self.assertTrue(numpy.allclose(vj0,vk1[1]))
        self.assertTrue(numpy.allclose(vk0,vk1[2]))
        self.assertTrue(numpy.allclose(vk0,vk1[3]))

        vj0, vk0 = scf._vhf.incore(rhf._eri, dm1, 1)
        vj1, vk1 = runjk(dm1, 1, 'int2e_sph', 'CVHFdot_nrs8',
                         'CVHFnrs8_ji_s2kl', 'CVHFnrs8_jk_s2il')
        vj1 = lib.hermi_triu(vj1, 1)
        vk1 = lib.hermi_triu(vk1, 1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_sr_vhf_q_cond(self):
        for omega in [.1, .2, .3]:
            for l in [0, 1, 2, 3]:
                for a in [.15, .5, 2.5]:
                    rs = numpy.arange(1, 10) * 2.
                    mol = gto.M(atom=['H 0 0 0'] + [f'H 0 0 {r}' for r in rs],
                                basis=[[l,[a,1]]], unit='B')
                    nbas = mol.nbas
                    q_cond = numpy.empty((6,nbas,nbas), dtype=numpy.float32)
                    ao_loc = mol.ao_loc
                    cintopt = lib.c_null_ptr()
                    with mol.with_short_range_coulomb(omega):
                        with mol.with_integral_screen(1e-26):
                            libcvhf2.CVHFnr_sr_int2e_q_cond(
                                libcvhf2.int2e_sph, cintopt,
                                q_cond.ctypes.data_as(ctypes.c_void_p),
                                ao_loc.ctypes.data_as(ctypes.c_void_p),
                                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                                mol._env.ctypes.data_as(ctypes.c_void_p))

                    s_index = q_cond[2]
                    si_0 = s_index[0,0]
                    si_others = s_index.diagonal()[1:]
                    with mol.with_short_range_coulomb(omega):
                        ints = [abs(mol.intor_by_shell('int2e', (0,0,i,i))).max()
                                for i in range(1, mol.nbas)]

                        aij = akl = a * 2
                        omega2 = mol.omega**2
                        theta = 1/(2/aij+1/omega2)
                        rr = rs**2
                        estimator = rr * numpy.exp(si_0 + si_others - theta*rr)
                        assert all(estimator / ints > 1)


if __name__ == '__main__':
    print('Full Tests for nr_direct')
    unittest.main()
