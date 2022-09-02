# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
import time
import h5py
import tempfile

### Incore integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

    log = logger.Logger(myadc.stdout, myadc.verbose)
    kpts = myadc.kpts
    nkpts = myadc.nkpts
    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    dtype = myadc.mo_coeff[0].dtype

    mo_coeff = myadc.mo_coeff = padded_mo_coeff(myadc, myadc.mo_coeff)

    fao2mo = myadc._scf.with_df.ao2mo

    kconserv = myadc.khelper.kconserv
    khelper = myadc.khelper

    orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')

    fao2mo = myadc._scf.with_df.ao2mo
    eris = lambda:None

    log.info('using incore ERI storage')
    eris.oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
    eris.oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
    eris.ovoo = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=dtype)
    eris.ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
    eris.ovvv = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=dtype)
    eris.ovvo = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=dtype)

    for (ikp,ikq,ikr) in khelper.symm_map.keys():
        iks = kconserv[ikp,ikq,ikr]
        eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                         (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
        if dtype == np.float64:
            eri_kpt = eri_kpt.real
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
        for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
            eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr)
            eris.oooo[kp,kq,kr] = eri_kpt_symm[:nocc,:nocc,:nocc,:nocc]/nkpts
            eris.oovv[kp,kq,kr] = eri_kpt_symm[:nocc,:nocc, nocc:,nocc:]/nkpts
            eris.ovoo[kp,kq,kr] = eri_kpt_symm[:nocc,nocc:,:nocc,:nocc]/nkpts
            eris.ovov[kp,kq,kr] = eri_kpt_symm[:nocc,nocc:,:nocc,nocc:]/nkpts
            eris.ovvv[kp,kq,kr] = eri_kpt_symm[:nocc,nocc:,nocc:,nocc:]/nkpts
            eris.ovvo[kp,kq,kr] = eri_kpt_symm[:nocc,nocc:,nocc:,:nocc]/nkpts

    if (myadc.method == "adc(2)-x" and myadc.higher_excitations is True) or (myadc.method == "adc(3)"):
        eris.vvvv = myadc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

    return eris


def transform_integrals_outcore(myadc):

    from pyscf.pbc import tools
    from pyscf.pbc.cc.ccsd import _adjust_occ

    log = logger.Logger(myadc.stdout, myadc.verbose)
    kpts = myadc.kpts
    nkpts = myadc.nkpts
    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc

    dtype = myadc.mo_coeff[0].dtype

    mo_coeff = myadc.mo_coeff = padded_mo_coeff(myadc, myadc.mo_coeff)

    fao2mo = myadc._scf.with_df.ao2mo

    kconserv = myadc.khelper.kconserv
    khelper = myadc.khelper

    eris = lambda:None
    eris.feri = feri = lib.H5TmpFile()

    # The momentum conservation array
    kconserv = myadc.khelper.kconserv

    eris.feri = feri = lib.H5TmpFile()
    eris.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
    eris.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
    eris.ovoo = feri.create_dataset('ovoo', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=dtype)
    eris.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
    eris.ovvv = feri.create_dataset('ovvv', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=dtype)
    eris.ovvo = feri.create_dataset('ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=dtype)
    eris.vvvv = feri.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)

    cput1 = time.process_time(), time.time()
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                orbo_p = mo_coeff[kp][:, :nocc]
                orbo_q = mo_coeff[kq][:, :nocc]
                buf_kpt = fao2mo((orbo_p, orbo_q, mo_coeff[kr], mo_coeff[ks]),
                                 (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                if mo_coeff[0].dtype == np.float64:
                    buf_kpt = buf_kpt.real
                buf_kpt = buf_kpt.reshape(nocc, nocc, nmo, nmo)
                dtype = buf_kpt.dtype
                eris.oooo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :nocc, :nocc] / nkpts
                eris.oovv[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:] / nkpts
    cput1 = log.timer_debug1('transforming oopq', *cput1)

    # <ia|pq> = (ip|aq)
    cput1 = time.process_time(), time.time()
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                orbo_p = mo_coeff[kp][:, :nocc]
                orbv_q = mo_coeff[kq][:, nocc:]
                buf_kpt = fao2mo((orbo_p, orbv_q, mo_coeff[kr], mo_coeff[ks]),
                                 (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
                if mo_coeff[0].dtype == np.float64:
                    buf_kpt = buf_kpt.real
                buf_kpt = buf_kpt.reshape(nocc,nvir,nmo, nmo)
                eris.ovoo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :nocc, :nocc] / nkpts
                eris.ovov[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :nocc, nocc:] / nkpts
                eris.ovvo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, nocc:, :nocc] / nkpts
                eris.ovvv[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, nocc:, nocc:] / nkpts
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

    if (myadc.method == "adc(2)-x" and myadc.higher_excitations is True) or (myadc.method == "adc(3)"):
        mem_now = lib.current_memory()[0]
        if nvir ** 4 * 16 / 1e6 + mem_now < myadc.max_memory:
            for (ikp, ikq, ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp, ikq, ikr]
                orbv_p = mo_coeff[ikp][:, nocc:]
                orbv_q = mo_coeff[ikq][:, nocc:]
                orbv_r = mo_coeff[ikr][:, nocc:]
                orbv_s = mo_coeff[iks][:, nocc:]
                # unit cell is small enough to handle vvvv in-core
                buf_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
                                 kpts[[ikp,ikq,ikr,iks]], compact=False)
                if dtype == np.float64:
                    buf_kpt = buf_kpt.real
                buf_kpt = buf_kpt.reshape((nvir, nvir, nvir, nvir))
                for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                    buf_kpt_symm = khelper.transform_symm(buf_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                    eris.vvvv[kp, kr, kq] = buf_kpt_symm / nkpts
        else:
            #raise MemoryError('Minimal memory requirements %s MB'
            #                  % (mem_now + nvir ** 4 / 1e6 * 16 * 2))
            for (ikp, ikq, ikr) in khelper.symm_map.keys():
                for a in range(nvir):
                    orbva_p = orbv_p[:, a].reshape(-1, 1)
                    buf_kpt = fao2mo((orbva_p, orbv_q, orbv_r, orbv_s),
                                     (kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]), compact=False)
                    if mo_coeff[0].dtype == np.float64:
                        buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape((1, nvir, nvir, nvir)).transpose(0, 2, 1, 3)

                    eris.vvvv[ikp, ikr, ikq, a, :, :, :] = buf_kpt[0, :, :, :] / nkpts
                    # Store symmetric permutations
                    eris.vvvv[ikr, ikp, iks, :, a, :, :] = buf_kpt.transpose(1, 0, 3, 2)[
                        :, 0, :, :] / nkpts
                    eris.vvvv[ikq, iks, ikp, :, :, a, :] = buf_kpt.transpose(2, 3, 0, 1).conj()[
                        :, :, 0, :] / nkpts
                    eris.vvvv[iks, ikq, ikr, :, :, :, a] = buf_kpt.transpose(3, 2, 1, 0).conj()[
                        :, :, :, 0] / nkpts
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

    return eris


def transform_integrals_df(myadc):
    from pyscf.ao2mo import _ao2mo
    cell = myadc.cell
    kpts = myadc.kpts
    nkpts = myadc.nkpts
    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    if myadc._scf.with_df._cderi is None:
        myadc._scf.with_df.build()
    dtype = myadc.mo_coeff[0].dtype

    mo_coeff = myadc.mo_coeff = padded_mo_coeff(myadc, myadc.mo_coeff)

    kconserv = myadc.khelper.kconserv

    # The momentum conservation array
    kconserv = myadc.khelper.kconserv

    with_df = myadc.with_df
    naux = with_df.get_naoaux()
    eris = lambda:None

    eris.dtype = dtype = np.result_type(dtype)
    eris.Lpq_mo = Lpq_mo = np.empty((nkpts, nkpts), dtype=object)
    Loo = np.empty((nkpts,nkpts,naux,nocc,nocc),dtype=dtype)
    Lvo = np.empty((nkpts,nkpts,naux,nvir,nocc),dtype=dtype)
    eris.Lvv = np.empty((nkpts,nkpts,naux,nvir,nvir),dtype=dtype)
    eris.Lov = np.empty((nkpts,nkpts,naux,nocc,nvir),dtype=dtype)

    eris.vvvv = None
    eris.ovvv = None

    with df._load3c(myadc._scf.with_df._cderi, 'j3c') as fload:
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                Lpq_ao = np.asarray(fload(kpti, kptj))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), tao, ao_loc)
                Lpq_mo[ki, kj] = out.reshape(-1, nmo, nmo)

                Loo[ki,kj] = Lpq_mo[ki,kj][:,:nocc,:nocc]
                eris.Lov[ki,kj] = Lpq_mo[ki,kj][:,:nocc,nocc:]
                Lvo[ki,kj] = Lpq_mo[ki,kj][:,nocc:,:nocc]
                eris.Lvv[ki,kj] = Lpq_mo[ki,kj][:,nocc:,nocc:]

    eris.feri = feri = lib.H5TmpFile()

    eris.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
    eris.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
    eris.ovoo = feri.create_dataset('ovoo', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=dtype)
    eris.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
    eris.ovvo = feri.create_dataset('ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=dtype)
    #eris.ovvv = feri.create_dataset('ovvv', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=dtype)

    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp,kq,kr]
                eris.oooo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', Loo[kp,kq], Loo[kr,ks])/nkpts
                eris.oovv[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', Loo[kp,kq], eris.Lvv[kr,ks])/nkpts
                eris.ovoo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Loo[kr,ks])/nkpts
                eris.ovov[kp,kq,kr] = lib.einsum(
                    'Lpq,Lrs->pqrs', eris.Lov[kp,kq], eris.Lov[kr,ks])/nkpts
                eris.ovvo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Lvo[kr,ks])/nkpts
                #eris.ovvv[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Lvv[kr,ks])/nkpts

    return eris

def calculate_chunk_size(myadc):

    avail_mem = (myadc.max_memory - lib.current_memory()[0]) * 0.5
    nocc = [np.count_nonzero(myadc.mo_occ[ikpt]) for ikpt in range(myadc.nkpts)]
    nocc = np.amax(nocc)
    nmo = [len(myadc.mo_occ[ikpt]) for ikpt in range(myadc.nkpts)]
    nmo = np.max(nocc) + np.max(np.array(nmo) - np.array(nocc))
    nvir = nmo - nocc
    vvv_mem = (nvir**3) * 8/1e6

    chnk_size =  int(avail_mem/vvv_mem)

    if chnk_size <= 0 :
        chnk_size = 1

    return chnk_size
