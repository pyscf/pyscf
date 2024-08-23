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

import unittest
import numpy
import numpy as np
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc.df import aft, aft_jk, ft_ao, FFTDF
from pyscf.pbc.tools import k2gamma
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import (unique_with_wrap_around,
                                       group_by_conj_pairs)


def setUpModule():
    global cell
    cell = gto.Cell()
    cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
    cell.a = numpy.eye(3) * 2.5
    cell.mesh = [21] * 3
    cell.build()

def tearDownModule():
    global cell
    del cell

def _update_vk_fake_gamma_debug(vk, Gpq, dmf, wcoulG, kpti_idx, kptj_idx, swap_2e,
                                k_to_compute):
    '''
    dmf is the factorized dm, dm = dmf * dmf.conj().T
    Computing exchange matrices with dmf:
    vk += np.einsum('ngij,njp,nkp,nglk,g->nil', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    vk += np.einsum('ngij,nlp,nip,nglk,g->nkj', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmfR, dmfI = dmf
    nG = len(wcoulG)
    n_dm, s_nao, nkpts, nocc = dmfR.shape
    nao = vkR.shape[-1]

    GpqR = GpqR.transpose(2,1,0)
    GpqI = GpqI.transpose(2,1,0)
    assert GpqR.flags.c_contiguous
    GpqR = GpqR.reshape(s_nao, -1)
    GpqI = GpqI.reshape(s_nao, -1)

    for i in range(n_dm):
        moR = dmfR[i].reshape(s_nao,nkpts*nocc)
        moI = dmfI[i].reshape(s_nao,nkpts*nocc)
        ipGR = lib.dot(moR.T, GpqR).reshape(nkpts,nocc,nao,nG)
        ipGI = lib.dot(moR.T, GpqI).reshape(nkpts,nocc,nao,nG)
        ipGIi = lib.dot(moI.T, GpqR).reshape(nkpts,nocc,nao,nG)
        ipGRi = lib.dot(moI.T, GpqI).reshape(nkpts,nocc,nao,nG)
        for k, (ki, kj) in enumerate(zip(kpti_idx, kptj_idx)):
            if k_to_compute[ki]:
                ipGR1 = np.array(ipGR[kj].transpose(1,0,2), order='C')
                ipGI1 = np.array(ipGI[kj].transpose(1,0,2), order='C')
                ipGR1 -= ipGRi[kj].transpose(1,0,2)
                ipGI1 += ipGIi[kj].transpose(1,0,2)
                ipGR2 = ipGR1 * wcoulG
                ipGI2 = ipGI1 * wcoulG
                zdotNC(ipGR1.reshape(nao,-1), ipGI1.reshape(nao,-1),
                       ipGR2.reshape(nao,-1).T, ipGI2.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)
            if swap_2e and k_to_compute[kj]:
                ipGR1 = np.array(ipGR[ki].transpose(1,0,2), order='C')
                ipGI1 = np.array(ipGI[ki].transpose(1,0,2), order='C')
                ipGR1 += ipGRi[ki].transpose(1,0,2)
                ipGI1 -= ipGIi[ki].transpose(1,0,2)
                ipGR2 = ipGR1 * wcoulG
                ipGI2 = ipGI1 * wcoulG
                zdotCN(ipGR1.reshape(nao,-1), ipGI1.reshape(nao,-1),
                       ipGR2.reshape(nao,-1).T, ipGI2.reshape(nao,-1).T,
                       1, vkR[i,kj], vkI[i,kj], 1)

class KnownValues(unittest.TestCase):
    def test_jk(self):
        mf0 = scf.RHF(cell)
        dm = mf0.get_init_guess()

        mydf = aft.AFTDF(cell)
        mydf.mesh = [11]*3
        vj, vk = mydf.get_jk(dm, exxdiv='ewald')
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 3.0455881073561235*(4./3.66768353356587)**2, 8)
        self.assertAlmostEqual(ek1, 7.7905480251964629*(4./3.66768353356587)**2, 7)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mydf.get_jk(dm, hermi=0, exxdiv='ewald')
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 12.234106555081793, 7)
        self.assertAlmostEqual(ek1, 43.988705494650802, 6)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv='ewald')
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 12.233546641482697, 8)
        self.assertAlmostEqual(ek1, 43.946958026023722, 7)

    def test_jk_complex_dm(self):
        scaled_center = [0.3728,0.5524,0.7672]
        kpt = cell.make_kpts([1,1,1], scaled_center=scaled_center)[0]
        mf = scf.RHF(cell, kpt=kpt)
        dm = mf.init_guess_by_1e()

        mydf = aft.AFTDF(cell, kpts=[kpt])
        vj1, vk1 = mydf.get_jk(dm, kpts=kpt, exxdiv='ewald')
        vjs, vks = mydf.get_jk([dm], kpts=[kpt], exxdiv='ewald')
        vj , vk  = vjs[0], vks[0]

        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        ej  = numpy.einsum('ij,ji->', vj , dm)
        ek  = numpy.einsum('ij,ji->', vk , dm)

        # kpts and single kpt AFTDF must match exactly
        self.assertAlmostEqual(ej1, ej, 10)
        self.assertAlmostEqual(ek1, ek, 10)

    def test_aft_j(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = aft.AFTDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        vj = aft_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vj[0]), (0.93946193432413905+0.00010862804196223034j)/4, 9)
        self.assertAlmostEqual(lib.fp(vj[1]), (0.94866073525335626+0.005571199307452865j)  /4, 9)
        self.assertAlmostEqual(lib.fp(vj[2]), (1.1492194255929766+0.0093705761598793739j)  /4, 9)
        self.assertAlmostEqual(lib.fp(vj[3]), (1.1397493412770023+0.010731970529096637j)   /4, 9)

    def test_aft_k(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = aft.AFTDF(cell)
        mydf.kpts = kpts
        vk = aft_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (4.3373802352168278-0.062977052131451577j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[1]), (2.878809181709983+0.028843869853690692j) /8, 8)
        self.assertAlmostEqual(lib.fp(vk[2]), (3.7027622609953061-0.052034330663180237j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[3]), (5.0939994842559422+0.060094478876149444j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[4]), (4.2942087551592651-0.061138484763336887j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[5]), (3.9689429688683679+0.048471952758750547j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[6]), (3.6342630872923456-0.054892635365850449j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[7]), (3.3483735224533548+0.040877095049528467j)/8, 8)

        ref = FFTDF(cell, kpts=kpts).get_jk(dm, kpts=kpts)[1]
        self.assertAlmostEqual(abs(ref-vk).max(), 0, 8)

    def test_aft_k1(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        nao = cell.nao_nr()
        mydf = aft.AFTDF(cell)
        mydf.kpts = kpts
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = aft_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (8.7518173818250702-0.11793770445839372j) /8, 8)
        self.assertAlmostEqual(lib.fp(vk[1]), (5.7682393685317894+0.069482280306391239j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[2]), (7.1890462727492324-0.088727079644645671j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[3]), (10.08358152800003+0.1278144339422369j   )/8, 8)
        self.assertAlmostEqual(lib.fp(vk[4]), (8.393281242945573-0.099410704957774876j) /8, 8)
        self.assertAlmostEqual(lib.fp(vk[5]), (7.9413682328898769+0.1015563120870652j)  /8, 8)
        self.assertAlmostEqual(lib.fp(vk[6]), (7.3743790120272408-0.096290683129384574j)/8, 8)
        self.assertAlmostEqual(lib.fp(vk[7]), (6.8144379626901443+0.08071261392857812j) /8, 8)

    def test_aft_k2(self):
        kpts = cell.make_kpts([1]*3)
        nkpts = len(kpts)
        numpy.random.seed(1)
        nao = cell.nao_nr()
        nocc = 2
        mo = (numpy.random.random((nkpts,nao,nocc)) +
              numpy.random.random((nkpts,nao,nocc))*1j)
        mo_occ = numpy.ones((nkpts,nocc))
        dm = numpy.einsum('npi,nqi->npq', mo, mo.conj())
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)
        mydf = aft.AFTDF(cell)
        mydf.kpts = kpts
        vk = aft_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk), 0.12513784226311198-0.10318660336428756j, 8)
        vk1 = aft_jk.get_k_for_bands(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(abs(vk-vk1).max(), 0, 9)

    def test_aft_k3(self):
        kpts = cell.make_kpts([6,1,1])
        nkpts = len(kpts)
        numpy.random.seed(1)
        nao = cell.nao_nr()
        nocc = 2
        mo = (numpy.random.random((nkpts,nao,nocc)) +
              numpy.random.random((nkpts,nao,nocc))*1j)
        mo_occ = numpy.ones((nkpts,nocc))
        dm = numpy.einsum('npi,nqi->npq', mo, mo.conj())
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)
        mydf = aft.AFTDF(cell)
        mydf.k_conj_symmetry = False
        mydf.kpts = kpts
        vk = aft_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk), 5.872042619636364+0.39662848875321643j, 7)
        vk1 = aft_jk.get_k_for_bands(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(abs(vk-vk1).max(), 0, 8)
        ref = FFTDF(cell, kpts=kpts).get_jk(dm, kpts=kpts)[1]
        self.assertAlmostEqual(abs(ref-vk).max(), 0, 7)

    def test_update_vk(self):
        L = 5.
        cell = gto.Cell()
        np.random.seed(1)
        cell.a = np.diag([L,L,L]) + np.random.rand(3,3)
        cell.mesh = np.array([27]*3)
        cell.atom = '''H    1.8   2.       .4
                       Li    1.    1.       1.'''
        cell.basis = {'H': [[0, (1.0, 1)],
                            [1, (0.8, 1)]
                           ],
                      'Li': [[0, (1.0, 1)],
                             [0, (0.6, 1)],
                            [1, (0.8, 1)]
                           ]}
        cell.build()
        kmesh = [6,1,1]
        kpts = cell.make_kpts(kmesh)
        mf = scf.KRHF(cell, kpts)
        mf.run()
        mf.mo_coeff = np.array(mf.mo_coeff)

        dms = mf.make_rdm1()
        ref = FFTDF(cell, kpts=kpts).get_jk(dms, kpts=kpts)[1]
        self.assertAlmostEqual(lib.fp(ref), -7.614964681516531, 8)

        df = aft.AFTDF(cell, kpts)
        dms = np.array(dms)[None,:]
        nkpts = kpts.shape[0]
        weight = 1./nkpts
        nao = cell.nao
        nocc = cell.nelec[0]
        dmf = np.asarray(mf.mo_coeff[:,:,:nocc] * np.sqrt(mf.mo_occ)[:,None,:nocc], order='C')
        dmf = dmf[None,:]

        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        rcut = ft_ao.estimate_rcut(cell)
        supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
        supmol = supmol.strip_basis(rcut)
        t_rev_pairs = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')

        moR, moI = aft_jk._mo_k2gamma(supmol, dmf, kpts, t_rev_pairs)
        mof = [moR, None]
        dm0 = [dms.real.copy(), dms.imag.copy()]
        dmf = [dmf.real.copy(), dmf.imag.copy()]
        ft_kern0 = supmol.gen_ft_kernel(return_complex=False, kpts=kpts)
        ft_kern1 = aft_jk._gen_ft_kernel_fake_gamma(cell, supmol, 's1')

        uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        scaled_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)

        n_dm = 1
        Gv, Gvbase, kws = cell.get_Gv_weights(cell.mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        k_to_compute = np.zeros(nkpts, dtype=np.int8)
        k_to_compute[t_rev_pairs[:,0]] = 1
        ngrids = Gv.shape[0]
        Gblksize = 8000

        for ft_kern, update_vk, dm in (
            (ft_kern0, aft_jk._update_vk_, dm0),
            (ft_kern0, aft_jk._update_vk1_, dm0),
            (ft_kern0, aft_jk._update_vk_dmf, dmf),
            (ft_kern0, aft_jk._update_vk1_dmf, dmf),
            (ft_kern1, aft_jk._update_vk_fake_gamma, mof)):
            vkR = np.zeros((n_dm,nkpts,nao,nao))
            vkI = np.zeros((n_dm,nkpts,nao,nao))
            vk = [vkR, vkI]
            for kpt, ki_idx, kj_idx, self_conj in aft_jk.kk_adapted_iter(cell, kpts):
                Gv, Gvbase, kws = cell.get_Gv_weights(cell.mesh)
                wcoulG = pbctools.get_coulG(cell, kpt, False, mf, cell.mesh, Gv)
                wcoulG *= kws
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt)
                    update_vk(vk, Gpq, dm, wcoulG[p0:p1] * weight, ki_idx, kj_idx,
                              not self_conj, k_to_compute, t_rev_pairs)
                    Gpq = None

            vk = vkR + vkI * 1j
            for k, k_conj in t_rev_pairs:
                if k != k_conj:
                    vk[:,k_conj] = vk[:,k].conj()
            self.assertAlmostEqual(abs(vk-ref).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for aft jk")
    unittest.main()
