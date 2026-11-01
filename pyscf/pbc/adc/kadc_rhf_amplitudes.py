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

import time
import numpy as np
from pyscf import ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx, _padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM

from pyscf.pbc import tools
import h5py
import tempfile


# Note : All integrals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba


def _batch_contract(script, a, a_indices, b, b_indices, out, buffer, alpha=1):
    compatible = (buffer is not None and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                  and a.flags.c_contiguous and b.flags.c_contiguous
                  and a.dtype == b.dtype == out.dtype == buffer.dtype)
    block_size = max(int(np.prod(a.shape[3:])), int(np.prod(b.shape[3:])))
    block_size *= a.dtype.itemsize
    batch_size = min(buffer.shape[0], buffer.nbytes // (2 * block_size)) if compatible else 0

    if batch_size < 2:
        for ia, ib in zip(zip(*a_indices), zip(*b_indices)):
            result = lib.einsum(script, a[ia], b[ib], optimize=True)
            if alpha != 1:
                result *= alpha
            out += result
        return out

    lhs, rhs = script.split('->')
    a_sub, b_sub = lhs.split(',')
    batch_script = f'x{a_sub},x{b_sub}->{rhs}'
    flat_buffer = buffer.reshape(-1)
    half = flat_buffer.size // 2
    nkpts = buffer.shape[0]
    for start in range(0, nkpts, batch_size):
        stop = min(start + batch_size, nkpts)
        count = stop - start
        a_shape = (count,) + a.shape[3:]
        b_shape = (count,) + b.shape[3:]
        buf_a = flat_buffer[:int(np.prod(a_shape))].reshape(a_shape)
        buf_b = flat_buffer[half:half + int(np.prod(b_shape))].reshape(b_shape)
        source_a = a.reshape((nkpts ** 3,) + a.shape[3:])
        source_b = b.reshape((nkpts ** 3,) + b.shape[3:])
        idx_a = np.ravel_multi_index(tuple(x[start:stop] for x in a_indices), (nkpts,) * 3)
        idx_b = np.ravel_multi_index(tuple(x[start:stop] for x in b_indices), (nkpts,) * 3)
        np.take(source_a, idx_a, axis=0, out=buf_a)
        np.take(source_b, idx_b, axis=0, out=buf_b)
        result = lib.einsum(batch_script, buf_a, buf_b, optimize=True)
        if alpha != 1:
            result *= alpha
        out += result
    return out


def _amplitude_storage(myadc, dtype):
    mem_needed = kadc_ao2mo._amplitude_memory(myadc, dtype)
    mem_now = lib.current_memory()[0]
    if (myadc.incore_complete or mem_now + mem_needed < myadc.max_memory * .9):
        myadc._amp_h5file = myadc._amp_tmpfile = None
        return None, lambda name, shape, dtype: np.zeros(shape, dtype=dtype)
    tmpfile = tempfile.TemporaryFile()
    file = h5py.File(tmpfile, 'a')
    myadc._amp_h5file = file
    myadc._amp_tmpfile = tmpfile
    return file, file.create_dataset


def compute_amplitudes_energy(myadc, eris, verbose=None, if_corr=True):

    t1, t2, myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    if if_corr:
        e_corr = myadc.compute_energy(t2, eris)
    else:
        e_corr = 0.

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    cput0 = (time.process_time(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nmo = myadc.nmo
    nocc = myadc.nocc
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    cell = myadc.cell
    kpts = myadc.kpts
    madelung = tools.madelung(cell, kpts)

    # release the scratch file from a previous kernel call, if any
    myadc._close_amp_scratch()
    myadc.t1 = None
    myadc.t2 = None
    myadc._adc_es = None
    myadc.imds.t2_1_vvvv = None

    # Compute first-order doubles t2 (tijab)
    t2_shape = (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)
    f, create_dataset = _amplitude_storage(myadc, eris.ovov.dtype)
    log.info('using %s amplitude storage', 'incore' if f is None else 'HDF5')
    t2_1 = create_dataset('t2_1', t2_shape, dtype=eris.ovov.dtype)

    mo_energy = myadc.mo_energy
    mo_coeff = myadc.mo_coeff
    mo_coeff, mo_energy = _add_padding(myadc, mo_coeff, mo_energy)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(myadc, kind="split")

    kconserv = myadc.khelper.kconserv
    k_index = np.arange(nkpts)
    buffer = None
    batch_arrays = (t2_1, eris.ovoo, eris.oooo, eris.oovv, eris.ovvo)
    use_batch = all(isinstance(x, np.ndarray) and x.flags.c_contiguous and x.dtype == t2_1.dtype
                    for x in batch_arrays)
    if nkpts > 1 and (myadc.approx_trans_moments is False or myadc.method == 'adc(3)') and use_batch:
        buffer = np.empty(t2_shape, dtype=t2_1.dtype)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        if ki > kj or (ki == kj and ka > kb):
            continue

        # For discussion of LARGE_DENOM, see t1new update above
        eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                       [0, nvir, ka, mo_e_v, nonzero_vpadding],
                       fac=[1.0, -1.0])

        ejb = _get_epq([0, nocc, kj, mo_e_o, nonzero_opadding],
                       [0, nvir, kb, mo_e_v, nonzero_vpadding],
                       fac=[1.0, -1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        value = eris.ovov[ki, ka, kj].conj().transpose((0, 2, 1, 3)) / eijab
        t2_1[ki, kj, ka] = value
        if ki != kj or ka != kb:
            t2_1[kj, ki, kb] = value.transpose(1, 0, 3, 2)

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    t1_2 = None
    if myadc.approx_trans_moments is False or myadc.method == "adc(3)":
        # Compute second-order singles t1 (tij)
        t1_2 = np.zeros((nkpts, nocc, nvir), dtype=t2_1.dtype)
        eris_ovoo = eris.ovoo
        for ki in range(nkpts):
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = kconserv[ki, kc, kk]
                    ka = kconserv[kc, kk, kd]

                    if eris.ovvv is None:
                        chnk_size = myadc.chnk_size
                        chnk_size = min(chnk_size, nocc)
                        a = 0
                        for p in range(0, nocc, chnk_size):
                            eris_ovvv = dfadc.get_ovvv_df(
                                myadc, eris.Lov[kk, kd], eris.Lvv[ka, kc],
                                p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                            k = eris_ovvv.shape[0]
                            t1_2[ki] += 1.5 * lib.einsum('kdac,ikcd->ia',
                                                         eris_ovvv, t2_1[ki, kk, kc, :, a:a + k], optimize=True)
                            t1_2[ki] -= 0.5 * lib.einsum('kdac,kicd->ia',
                                                         eris_ovvv, t2_1[kk, ki, kc, a:a + k, :], optimize=True)
                            del eris_ovvv
                            eris_ovvv = dfadc.get_ovvv_df(
                                myadc, eris.Lov[kk, kc], eris.Lvv[ka, kd],
                                p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                            t1_2[ki] -= 0.5 * lib.einsum('kcad,ikcd->ia',
                                                         eris_ovvv, t2_1[ki, kk, kc, :, a:a + k], optimize=True)
                            t1_2[ki] += 0.5 * lib.einsum('kcad,kicd->ia',
                                                         eris_ovvv, t2_1[kk, ki, kc, a:a + k, :], optimize=True)
                            del eris_ovvv
                            a += k
                    else:
                        eris_ovvv = eris.ovvv[:]
                        t1_2[ki] += 1.5 * lib.einsum('kdac,ikcd->ia',
                                                     eris_ovvv[kk, kd, ka], t2_1[ki, kk, kc], optimize=True)
                        t1_2[ki] -= 0.5 * lib.einsum('kdac,kicd->ia',
                                                     eris_ovvv[kk, kd, ka], t2_1[kk, ki, kc], optimize=True)
                        t1_2[ki] -= 0.5 * lib.einsum('kcad,ikcd->ia',
                                                     eris_ovvv[kk, kc, ka], t2_1[ki, kk, kc], optimize=True)
                        t1_2[ki] += 0.5 * lib.einsum('kcad,kicd->ia',
                                                     eris_ovvv[kk, kc, ka], t2_1[kk, ki, kc], optimize=True)
                        del eris_ovvv

                kl = k_index
                kc = kconserv[kk, ki, kl]
                ka = kconserv[kl, kc, kk]
                kk_index = np.full(nkpts, kk)

                _batch_contract('lcki,klac->ia', eris_ovoo, (kl, kc, kk_index),
                                   t2_1, (kk_index, kl, ka), t1_2[ki], buffer, alpha=-1.5)
                _batch_contract('lcki,lkac->ia', eris_ovoo, (kl, kc, kk_index),
                                   t2_1, (kl, kk_index, ka), t1_2[ki], buffer, alpha=.5)
                _batch_contract('kcli,lkac->ia', eris_ovoo, (kk_index, kc, kl),
                                   t2_1, (kl, kk_index, ka), t1_2[ki], buffer, alpha=-.5)
                _batch_contract('kcli,klac->ia', eris_ovoo, (kk_index, kc, kl),
                                   t2_1, (kk_index, kl, ka), t1_2[ki], buffer, alpha=.5)

        for ki in range(nkpts):
            ka = ki
            eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                           [0, nvir, ka, mo_e_v, nonzero_vpadding],
                           fac=[1.0, -1.0])
            t1_2[ki] = t1_2[ki] / eia

        cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t2_2 = None
    t1_3 = None
    t2_1_vvvv = None

    if (myadc.method == "adc(2)-x" and myadc.approx_trans_moments is False) or (myadc.method == "adc(3)"):
        # Compute second-order doubles t2 (tijab)
        t2_1_vvvv = create_dataset(
            't2_1_vvvv', (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.ovov.dtype)
        eris_oooo = eris.oooo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv

        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka, kc, kb]
            for ki in range(nkpts):
                kj = kconserv[ka, ki, kb]
                if ki > kj or (ki == kj and ka > kb):
                    continue
                if isinstance(eris.vvvv, np.ndarray):
                    eris_vvvv = eris.vvvv.reshape(nkpts, nkpts, nkpts, nvir * nvir, nvir * nvir)
                    t2_1_a = t2_1[:].reshape(nkpts, nkpts, nkpts, nocc * nocc, nvir * nvir)
                    value = np.dot(t2_1_a[ki, kj, kc],
                                   eris_vvvv[kc, kd, ka].conj()).reshape(nocc, nocc, nvir, nvir)
                elif eris.vvvv is None:
                    value = contract_ladder(myadc, t2_1[ki, kj, kc], eris.Lvv, ka, kb, kc)
                else:
                    value = contract_ladder(myadc, t2_1[ki, kj, kc], eris.vvvv, kc, kd, ka)
                t2_1_vvvv[ki, kj, ka] += value
                if ki != kj or ka != kb:
                    t2_1_vvvv[kj, ki, kb] += value.transpose(1, 0, 3, 2)

        t2_2 = np.array(t2_1_vvvv, copy=True)

        if myadc.exxdiv is not None:
            t2_2 -= 2.0 * madelung * t2_1

        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            if ki > kj or (ki == kj and ka > kb):
                continue
            kk = k_index
            ki_index = np.full(nkpts, ki)
            kj_index = np.full(nkpts, kj)
            ka_index = np.full(nkpts, ka)
            out = t2_2[ki, kj, ka]

            kc = kconserv[ki, ka, kk]
            kb = kconserv[kj, kk, kc]
            _batch_contract('kjbc,kica->ijab', eris_oovv, (kk, kj_index, kb),
                               t2_1, (kk, ki_index, kc), out, buffer, alpha=-1)

            kc = kconserv[kk, ka, kj]
            kb = kconserv[ki, kk, kc]
            _batch_contract('kibc,jkca->ijab', eris_oovv, (kk, ki_index, kb),
                               t2_1, (kj_index, kk, kc), out, buffer, alpha=-1)

            kc = kconserv[ka, kj, kk]
            kb = kconserv[ki, kc, kk]
            _batch_contract('kjac,ikcb->ijab', eris_oovv, (kk, kj_index, ka_index),
                               t2_1, (ki_index, kk, kc), out, buffer, alpha=-1)

            kc = kconserv[ka, ki, kk]
            kb = kconserv[kk, kc, kj]
            _batch_contract('kiac,kjcb->ijab', eris_oovv, (kk, ki_index, ka_index),
                               t2_1, (kk, kj_index, kc), out, buffer, alpha=-1)

            kl = k_index
            kk = kconserv[kj, kl, ki]
            _batch_contract('kilj,klab->ijab', eris_oooo, (kk, ki_index, kl),
                               t2_1, (kk, kl, ka_index), out, buffer, alpha=1)

            kk = k_index
            kc = kconserv[ki, ka, kk]
            kb = kconserv[kc, kk, kj]
            _batch_contract('kcbj,kica->ijab', eris_ovvo, (kk, kc, kb),
                               t2_1, (kk, ki_index, kc), out, buffer, alpha=2)

            kc = kconserv[kk, ka, ki]
            _batch_contract('kcbj,ikca->ijab', eris_ovvo, (kk, kc, kb),
                               t2_1, (ki_index, kk, kc), out, buffer, alpha=-1)

            kc = kconserv[kk, ki, ka]
            kb = kconserv[kk, kj, kc]
            _batch_contract('kcai,kjcb->ijab', eris_ovvo, (kk, kc, ka_index),
                               t2_1, (kk, kj_index, kc), out, buffer, alpha=2)

            kc = kconserv[kk, ki, ka]
            kb = kconserv[kj, kk, kc]
            _batch_contract('kcai,jkcb->ijab', eris_ovvo, (kk, kc, ka_index),
                               t2_1, (kj_index, kk, kc), out, buffer, alpha=-1)

            kb = kconserv[ki, ka, kj]
            eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                           [0, nvir, ka, mo_e_v, nonzero_vpadding],
                           fac=[1.0, -1.0])

            ejb = _get_epq([0, nocc, kj, mo_e_o, nonzero_opadding],
                           [0, nvir, kb, mo_e_v, nonzero_vpadding],
                           fac=[1.0, -1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            out /= eijab
            t2_2[ki, kj, ka] = out
            if ki != kj or ka != kb:
                t2_2[kj, ki, kb] = out.transpose(1, 0, 3, 2)

        cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

        if (myadc.method == "adc(3)" and myadc.approx_trans_moments is False):
            raise NotImplementedError('3rd order singles amplitudes not implemented')

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t2, eris):

    nkpts = myadc.nkpts

    emp2 = 0.0
    eris_ovov = eris.ovov
    t2_amp = np.array(t2[0], copy=True)

    if (myadc.method == "adc(3)"):
        t2_amp += t2[1][:]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 2 * lib.einsum('ijab,iajb', t2_amp[ki, kj, ka], eris_ovov[ki, ka, kj], optimize=True)
        emp2 -= 1 * lib.einsum('ijab,jaib', t2_amp[ki, kj, ka], eris_ovov[kj, ka, ki], optimize=True)

    del t2_amp
    emp2 = emp2.real / nkpts
    return emp2


def contract_ladder(myadc, t_amp, vvvv, ka, kb, kc):

    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    kconserv = myadc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    t_amp = np.ascontiguousarray(t_amp.reshape(nocc * nocc, nvir * nvir))
    t = np.zeros((nocc, nocc, nvir, nvir), dtype=t_amp.dtype)
    chnk_size = myadc.chnk_size
    if chnk_size is None:
        chnk_size = myadc.get_chnk_size()
    chnk_size = min(chnk_size, nvir)
    a = 0
    if isinstance(vvvv, np.ndarray):
        vv1 = vvvv[kc, ka]
        vv2 = vvvv[kd, kb]
        for p in range(0, nvir, chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vv1, vv2, p, chnk_size) / nkpts
            vvvv_p = vvvv_p.reshape(-1, nvir * nvir)
            k = vvvv_p.shape[0]
            t += np.dot(t_amp[:, a:a + k], vvvv_p.conj()).reshape(nocc, nocc, nvir, nvir)
            del vvvv_p
            a += k
    else:
        for p in range(0, nvir, chnk_size):
            vvvv_p = vvvv[ka, kb, kc, p:p + chnk_size, :, :, :].reshape(-1, nvir * nvir)
            k = vvvv_p.shape[0]
            t += np.dot(t_amp[:, a:a + k], vvvv_p.conj()).reshape(nocc, nocc, nvir, nvir)
            del vvvv_p
            a += k

    return t
