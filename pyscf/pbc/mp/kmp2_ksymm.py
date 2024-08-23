#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import time
import numpy as np
from scipy.linalg import block_diag
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.mp import kmp2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', False)

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    if with_t2:
        return kernel_with_t2(mp, mo_energy, mo_coeff, verbose, with_t2)
    else:
        t2 = None

    t0 = (logger.process_clock(), logger.perf_counter())
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s2')
    _, igroup = np.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = igroup.ravel()
    igroup = list(igroup) + [len(kijab)]

    emp2_ss = emp2_os = 0.
    nao2mo = 0
    icount = 0
    for i in range(len(igroup)-1):
        istart = igroup[i]
        iend = igroup[i+1]
        kab = []
        for j in range(istart, iend):
            a, b = kijab[j][2:]
            kab.append([a, b])
            kab.append([b, a])
        kab = np.unique(np.asarray(kab), axis=0)

        ki = kijab[istart][0]
        kj = kijab[istart][1]
        kpts_i = kd.kpts[ki]
        kpts_j = kd.kpts[kj]
        orbo_i = mo_coeff[ki][:,:nocc]
        orbo_j = mo_coeff[kj][:,:nocc]

        for (ka, kb) in kab:
            kpts_a = kd.kpts[ka]
            kpts_b = kd.kpts[kb]
            orbv_a = mo_coeff[ka][:,nocc:]
            orbv_b = mo_coeff[kb][:,nocc:]
            oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                 (kpts_i,kpts_a,kpts_j,kpts_b),
                                 compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            nao2mo += 1

        for j in range(istart, iend):
            ka = kijab[j][2]
            kb = kijab[j][3]
            # Remove zero/padded elements from denominator
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2_ijab = np.conj(oovv_ij[ka]/eijab)
            idx_ibz = k4_bz2ibz[ki*nkpts**2 + kj*nkpts + ka]
            assert(icount == idx_ibz)
            edi = einsum('ijab,ijab', t2_ijab, oovv_ij[ka]).real * 2
            exi = -einsum('ijab,ijba', t2_ijab, oovv_ij[kb]).real
            emp2_ss += (edi*0.5 + exi) * weight[idx_ibz] * nkpts**3
            emp2_os += edi*0.5 * weight[idx_ibz] * nkpts**3
            icount += 1

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    assert(icount == len(kijab))
    logger.debug(mp, "Number of ao2mo transformations performed in KMP2: %d", nao2mo)
    logger.timer(mp, 'KMP2', *t0)
    return emp2, t2

def kernel_with_t2(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    #we need almost all t2 for computing rdm, so simply use kmp2 without symmetry
    kd = mp.kpts
    mp.kpts = kd.kpts
    emp2, t2 = kmp2.kernel(mp, mo_energy, mo_coeff, verbose, with_t2)
    mp.kpts = kd
    return emp2, t2

@lib.with_doc(kmp2.make_rdm1.__doc__)
def make_rdm1(mp, t2=None, kind="compact"):
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    d_imds = _gamma1_intermediates(mp, t2=t2)
    result = []
    padding_idxs = kmp2.padding_k_idx(mp, kind="joint")
    for (oo, vv), idxs in zip(zip(*d_imds), padding_idxs):
        oo += np.eye(*oo.shape)
        d = block_diag(oo, vv)
        d += d.conj().T
        if kind == "padded":
            result.append(d)
        else:
            result.append(d[np.ix_(idxs, idxs)])
    return result

def make_t2_for_rdm1(mp):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts
    kpts = kd.kpts
    k_ibz = kd.ibz2bz

    mo_coeff, mo_energy = kmp2._add_padding(mp, mp.mo_coeff, mp.mo_energy)
    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))
    fao2mo = mp._scf.with_df.ao2mo
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")
    t2 = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    nao2mo = 0
    for ki in range(nkpts):
        orbo_i = mo_coeff[ki][:,:nocc]
        for kj in range(ki, nkpts):
            orbo_j = mo_coeff[kj][:,:nocc]
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                if ki in k_ibz or kj in k_ibz or ka in k_ibz or kb in k_ibz:
                    nao2mo +=1
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                  (kpts[ki],kpts[ka],kpts[kj],kpts[kb]),
                                  compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts

                    eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                    n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                    eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                    ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                    n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                    ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

                    eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                    t2[ki,kj,ka] = np.conj(oovv / eijab)
    logger.debug(mp, "Number of ao2mo transformations performed in KMP2: %d", nao2mo)
    return t2

def _gamma1_intermediates(mp, t2=None):
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        t2 = make_t2_for_rdm1(mp)
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    kd = mp.kpts
    nkpts = kd.nkpts
    nkpts_ibz = kd.nkpts_ibz
    dtype = t2.dtype

    dm1occ = np.zeros((nkpts_ibz, nocc, nocc), dtype=dtype)
    dm1vir = np.zeros((nkpts_ibz, nvir, nvir), dtype=dtype)

    k_ibz = kd.ibz2bz
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                t2a = t2[ki,kj,ka]
                t2b = t2[ki,kj,kb]
                if ki > kj:
                    t2a = t2[kj,ki,kb].transpose(1,0,3,2)
                    t2b = t2[kj,ki,ka].transpose(1,0,3,2)
                if kb in k_ibz:
                    dm1vir[kd.bz2ibz[kb]] += einsum('ijax,ijay->yx', t2a.conj(), t2a) * 2 -\
                                             einsum('ijax,ijya->yx', t2a.conj(), t2b)
                if kj in k_ibz:
                    dm1occ[kd.bz2ibz[kj]] += einsum('ixab,iyab->xy', t2a.conj(), t2a) * 2 -\
                                             einsum('ixab,iyba->xy', t2a.conj(), t2b)
    return -dm1occ, dm1vir


class KsymAdaptedKMP2(kmp2.KMP2):
    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            logger.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        # TODO: compute e_hf for non-canonical SCF
        self.e_hf = self._scf.e_tot

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    make_rdm1 = make_rdm1

    def make_rdm2(self):
        raise NotImplementedError

KRMP2 = KMP2 = KsymAdaptedKMP2

from pyscf.pbc import scf
scf.khf_ksymm.KRHF.MP2 = lib.class_as_method(KRMP2)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([2,2,2], space_group_symmetry=True)
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mymp = mp.KMP2(kmf)
    emp2, t2 = mymp.kernel()
    print(emp2 - -0.13314158977189)
