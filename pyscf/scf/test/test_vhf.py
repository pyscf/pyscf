#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import lib
from pyscf.scf import _vhf


def setUpModule():
    global mol, mf, nao, nmo
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = '/dev/null',
        atom = '''
    O     0    0        0
    H     0    -0.757   0.587
    H     0    0.757    0.587''',
        basis = '631g',
    )

    mf = scf.RHF(mol)
    mf.scf()
    nao, nmo = mf.mo_coeff.shape

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_incore_s4(self):
        eri4 = ao2mo.restore(4, mf._eri, nmo)
        dm = mf.make_rdm1()
        vj0, vk0 = _vhf.incore(eri4, dm, hermi=1)
        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_mapdm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        eri0 = mol.intor('int2e_ip1_sph', comp=3).reshape(3,nao,nao,nao,nao)
        vj0 = numpy.einsum('nijkl,lk->nij', eri0, dm)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_mapdm1(self):
        numpy.random.seed(1)
        nao = mol.nao_nr(cart=True)
        dm = numpy.random.random((nao,nao))
        vhfopt = _vhf.VHFOpt(mol, 'int2e_cart', 'CVHFnrs8_prescreen',
                             'CVHFsetnr_direct_scf',
                             'CVHFsetnr_direct_scf_dm')
        vj0, vk0 = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                               vhfopt=vhfopt, hermi=0, cart=True)
        vj = _vhf.direct_mapdm('int2e_cart', 's1', 'kl->s1ij', dm, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vj0, vj))

        vk = _vhf.direct_mapdm('int2e_cart', 's1', 'jk->s1il', [dm]*2, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vk0, vk[0]))
        self.assertTrue(numpy.allclose(vk0, vk[1]))

    def test_direct_bindm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        vj0, vk0 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        dms = (dm,dm)
        vj1, vk1 = _vhf.direct_bindm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dms, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_bindm1(self):
        numpy.random.seed(1)
        nao = mol.nao_nr(cart=True)
        dm = numpy.random.random((nao,nao))
        vhfopt = _vhf.VHFOpt(mol, 'int2e_cart', 'CVHFnrs8_prescreen',
                             'CVHFsetnr_direct_scf',
                             'CVHFsetnr_direct_scf_dm')
        vj0, vk0 = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                               vhfopt=vhfopt, hermi=0, cart=True)
        vj = _vhf.direct_bindm('int2e_cart', 's1', 'kl->s1ij', dm, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vj0,vj))

    def test_rdirect_mapdm(self):
        numpy.random.seed(1)
        n2c = nao*2
        dm = (numpy.random.random((n2c,n2c)) +
              numpy.random.random((n2c,n2c)) * 1j)
        eri0 = mol.intor('int2e_g1_spinor', comp=3).reshape(3,n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.rdirect_mapdm('int2e_g1_spinor', 'a4ij',
                                      ('lk->s2ij', 'jk->s1il'),
                                      dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

        vj1 = _vhf.rdirect_mapdm('int2e_g1_spinor', 's1', 'lk->s1ij',
                                 dm, 3, mol._atm, mol._bas, mol._env)
        vj0 = numpy.einsum('nijkl,lk->nij', eri0, dm)
        self.assertTrue(numpy.allclose(vj0,vj1))

    def test_rdirect_bindm(self):
        n2c = nao*2
        numpy.random.seed(1)
        dm = (numpy.random.random((n2c,n2c)) +
              numpy.random.random((n2c,n2c)) * 1j)
        dm = dm + dm.conj().T

        eri0 = mol.intor('int2e_spsp1_spinor').reshape(n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('int2e_spsp1_spinor', 's4', 'jk->s1il',
                                 dm, 1, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

        opt_llll = _vhf.VHFOpt(mol, 'int2e_spinor',
                               'CVHFrkbllll_prescreen',
                               'CVHFrkbllll_direct_scf',
                               'CVHFrkbllll_direct_scf_dm')
        opt_llll._this.contents.r_vkscreen = _vhf._fpointer('CVHFrkbllll_vkscreen')
        eri0 = mol.intor('int2e_spinor').reshape(n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('int2e_spinor', 's1', 'jk->s1il',
                                 dm, 1, mol._atm, mol._bas, mol._env, opt_llll)
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_nr_direct_ex_drv(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao, nao))
        dm = dm + dm.conj().T
        shls_excludes = [0, 4] * 4
        vref = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm, 1, mol._atm, mol._bas, mol._env)
        self.assertAlmostEqual(lib.fp(vref), 5.0067176755619975, 12)

        v0 = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm[:6,:6], 1,
                               mol._atm, mol._bas, mol._env, shls_slice=shls_excludes)
        v1 = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm, 1,
                               mol._atm, mol._bas, mol._env, shls_excludes=shls_excludes)
        v1[:6,:6] += v0
        self.assertAlmostEqual(abs(v1 - vref).max(), 0, 12)

        vref = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm, 1, mol._atm, mol._bas, mol._env)
        self.assertAlmostEqual(lib.fp(vref), 48.61070262547175, 12)

        v0 = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm[:6,:6], 1,
                               mol._atm, mol._bas, mol._env, shls_slice=shls_excludes)
        v1 = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm, 1,
                               mol._atm, mol._bas, mol._env, shls_excludes=shls_excludes)
        v1[:6,:6] += v0
        self.assertAlmostEqual(abs(v1 - vref).max(), 0, 12)

    def test_direct_sr_vhf(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        omega = 0.15
        with mol.with_short_range_coulomb(omega):
            ref = _vhf.direct(dm, mol._atm, mol._bas, mol._env, optimize_sr=False)
            ref = numpy.array(ref)

        vhfopt = _VHFOpt(mol, 'int2e', omega=omega)
        vhfopt.init_cvhf_direct(mol)
        with mol.with_short_range_coulomb(omega):
            vjk = _vhf.direct(dm, mol._atm, mol._bas, mol._env, optimize_sr=True)
            vjk = numpy.array(vjk)

        self.assertAlmostEqual(abs(ref - vjk).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vjk), 25.317344717490613, 12)

MIN_CUTOFF = 1e-44
libcvhf = _vhf.libcvhf

class _VHFOpt(_vhf._VHFOpt):
    def __init__(self, mol, intor=None, prescreen='CVHFnoscreen',
                 qcondname=None, dmcondname=None, direct_scf_tol=1e-14,
                 omega=None):
        assert omega is not None
        with mol.with_short_range_coulomb(omega):
            _vhf._VHFOpt.__init__(self, mol, intor, prescreen, qcondname, dmcondname)
        self.omega = omega
        self._this.direct_scf_tol = numpy.log(direct_scf_tol)

    def init_cvhf_direct(self, mol, intor=None, qcondname=None):
        nbas = mol.nbas
        q_cond = numpy.empty((6,nbas,nbas), dtype=numpy.float32)
        ao_loc = mol.ao_loc
        cintopt = self._cintopt
        with mol.with_short_range_coulomb(self.omega):
            libcvhf.CVHFsetnr_sr_direct_scf(
                libcvhf.int2e_sph, cintopt,
                q_cond.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p))

        self._q_cond = q_cond
        logq_cond = q_cond.ctypes.data_as(ctypes.c_void_p)
        self._this.q_cond = logq_cond

    def set_dm(self, dm, atm=None, bas=None, env=None):
        assert dm[0].ndim == 2
        ao_loc = self.mol.ao_loc_nr()
        dm_cond = [lib.condense('NP_absmax', d, ao_loc, ao_loc) for d in dm]
        dm_cond = numpy.max(dm_cond, axis=0)
        dm_cond += MIN_CUTOFF  # to remove divide-by-zero error
        self._dm_cond = numpy.asarray(dm_cond, order='C', dtype=np.float32)
        self._this.dm_cond = self._dm_cond.ctypes.data_as(ctypes.c_void_p)


if __name__ == "__main__":
    print("Full Tests for _vhf")
    unittest.main()
