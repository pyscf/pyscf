#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

import ctypes
import tempfile
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _dfnevpt2_eris_outcore(mc, mo_coeff, with_df):
    '''
    Construction of ERIs required by NEVPT2 in MO basis using DF intermediates.

    This could have been a child class of _ERIS in mcscf/df.py but in CASSCF the ERIs
    such as ppaa and papa are also stored on disk, while the current NEVPT2 implementation only
    needs cvcv on disk. Additionally the MCSCF class does create some other intermediates which
    are not needed for NEVPT2. Hence, a separate function is implemented here.

    Steps:
    1. Transform DF integrals to MO basis in blocks of auxiliary functions to get (L|pq)
    2. Using (L|pq), construct the papa, and ppaa as done in mcscf/df.py
    3. Construct the pacv and cvcv intermediates required for NEVPT2
    '''
    log = logger.Logger(mc.stdout, mc.verbose)

    nao, nmo = mo_coeff.shape
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nvir = nmo - nocc
    nav = ncas + nvir
    naoaux = with_df.get_naoaux()

    mem_now = lib.current_memory()[0]
    max_memory = max(4000, 0.9*mc.max_memory-mem_now)

    # Step-1: transform DF integrals to MO basis to get (L|pq)
    t1 = t0 = (logger.process_clock(), logger.perf_counter())

    ppaa = np.empty((nmo,nmo,ncas,ncas))
    papa = np.empty((nmo,ncas,nmo,ncas))

    mo = np.asarray(mo_coeff, order='F')

    fxpp = lib.H5TmpFile()

    blksize = max(4, int(min(with_df.blockdim, (max_memory*.95e6/8-naoaux*nmo*ncas)/3/nmo**2)))

    bufpa = np.empty((naoaux,nmo,ncas))
    bufcv = np.empty((naoaux,ncore,nvir))
    bufs1 = np.empty((blksize,nmo,nmo))

    dgemm = lib.numpy_helper._dgemm
    fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    fxpp_keys = []
    b0 = 0
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        bufpp = bufs1[:naux]
        fdrv(ftrans, fmmm,
                bufpp.ctypes.data_as(ctypes.c_void_p),
                eri1.ctypes.data_as(ctypes.c_void_p),
                mo.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux), ctypes.c_int(nao),
                (ctypes.c_int*4)(0, nmo, 0, nmo),
                ctypes.c_void_p(0), ctypes.c_int(0))
        fxpp_keys.append([str(k), b0, b0+naux])
        fxpp[str(k)] = bufpp.transpose(1,2,0)
        bufpa[b0:b0+naux] = bufpp[:,:,ncore:nocc]
        bufcv[b0:b0+naux] = bufpp[:, :ncore, nocc:]
        b0 += naux

    bufs1 = bufpp = None
    t1 = log.timer('density fitting ao2mo step-1', *t0)

    # Step-2.1: from the transfomed (L|pq), build papa
    mem_now = lib.current_memory()[0]
    nblk = int(max(8, min(nmo, ((max_memory-mem_now)*1e6/8-bufpa.size)/(ncas**2*nmo))))
    bufs1 = np.empty((nblk,ncas,nmo,ncas))
    for p0, p1 in prange(0, nmo, nblk):
        tmp = bufs1[:p1-p0]
        dgemm('T', 'N', (p1-p0)*ncas, nmo*ncas, naoaux,
                bufpa.reshape(naoaux,-1), bufpa.reshape(naoaux,-1),
                tmp.reshape(-1,nmo*ncas), 1, 0, p0*ncas, 0, 0)
        papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)

    bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
    bufs1 = bufpa = None
    t1 = log.timer('density fitting papa step-2.1', *t1)

    # Step-2.2: from the transfomed (L|pq), build ppaa
    mem_now = lib.current_memory()[0]
    nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nmo*naoaux+ncas**2*nmo))))
    bufs1 = np.empty((nblk,nmo,naoaux))
    bufs2 = np.empty((nblk,nmo,ncas,ncas))
    for p0, p1 in prange(0, nmo, nblk):
        nrow = p1 - p0
        buf = bufs1[:nrow]
        tmp = bufs2[:nrow].reshape(-1,ncas**2)
        for key, col0, col1 in fxpp_keys:
            buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
        lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
        ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
    bufs1 = bufs2 = buf = None
    t1 = log.timer('density fitting ppaa step-2.2', *t1)

    # Step-3: from the transfomed (L|pq), build pacv and cvcv
    tmpdir = lib.param.TMPDIR
    cvcvfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    # Edge cases
    if ncore * nvir == 0 or ncore * nvir == 0:
        f5 = lib.H5TmpFile(cvcvfile.name, 'w')
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        cvcv[:,:] = 0
        pacv = np.zeros((nmo,ncas,ncore,nvir))
    else:
        mem_now = lib.current_memory()[0]
        nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nav*naoaux+nav*ncore*nvir))))
        pacv = np.empty((nmo,ncas,ncore,nvir))
        bufs1 = np.empty((nblk,nav,naoaux))
        bufs2 = np.empty((nblk,nav,ncore,nvir))
        bufcv = bufcv.reshape(naoaux, ncore*nvir)
        f5 = lib.H5TmpFile(cvcvfile.name, 'w')
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        for p0, p1 in prange(0, nmo, nblk):
            nrow = p1 - p0
            buf = bufs1[:nrow]
            tmp = bufs2[:nrow].reshape(-1,ncore*nvir)
            for key, col0, col1 in fxpp_keys:
                buf[:nrow,:,col0:col1] = fxpp[key][p0:p1,ncore:]

            lib.dot(buf.reshape(-1,naoaux), bufcv, 1, tmp)

            tmp2 = tmp.reshape(-1,nav,ncore,nvir)
            pacv[p0:p1] = tmp2[:,:ncas]
            if p0 < ncore:
                m  = min(p1, ncore) - p0
                r0 = p0*nvir
                r1 = min(p1, ncore)*nvir
                cvcv[r0:r1,:] = tmp2[:m,ncas:].reshape(m*nvir,ncore*nvir)

        bufs1 = bufs2 = buf = bufcv = None
    t1 = log.timer('density fitting pacv and cvcv step-3', *t1)
    t0 = log.timer('density fitting ao2mo', *t0)
    return papa, ppaa, pacv, cvcvfile

def _mem_usage(ncore, ncas, nmo):
    '''Estimate memory usage (in MB) for DF-NEVPT2 ERIs
        1. outcore memory for storing cvcv on disk
        2. incore memory for storing all ERIs in memory
    '''
    nvir = nmo - ncore - ncas
    papa = ppaa = nmo**2*ncas**2
    pacv = nmo*ncas*ncore*nvir
    cvcv = ncore*nvir*ncore*nvir
    outcore = (papa + ppaa + pacv) * 8/1e6
    incore = outcore + cvcv*8/1e6
    return incore, outcore

def _ERIS(mc, mo, with_df, method='incore'):
    ncore = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nvir = nmo - ncore - ncas
    moa = mo[:, ncore:ncore+ncas]
    moc = mo[:, :ncore]
    mov = mo[:, ncore+ncas:]

    max_memory = max(4000, 0.9*mc.max_memory-lib.current_memory()[0])

    mem_incore, mem_outcore = _mem_usage(ncore, ncas, nmo)
    mem_now = lib.current_memory()[0]
    if (method == 'incore' and with_df is not None and
        (mem_incore+mem_now < mc.max_memory*.9)):
        papa = with_df.ao2mo([mo, moa, mo, moa], compact=False)
        ppaa = with_df.ao2mo([mo, mo, moa, moa], compact=False)
        pacv = with_df.ao2mo([mo, moa, moc, mov], compact=False)
        cvcv = with_df.ao2mo([moc, mov, moc, mov], compact=False)
        papa = papa.reshape(nmo, ncas, nmo, ncas)
        ppaa = ppaa.reshape(nmo, nmo, ncas, ncas)
        pacv = pacv.reshape(nmo, ncas, ncore, nvir)
    elif with_df is not None and (mem_outcore < max_memory):
        papa, ppaa, pacv, cvcv = _dfnevpt2_eris_outcore(mc, mo, with_df)
    else:
        raise RuntimeError('DF-NEVPT2 ERIs cannot be constructed with the available memory %d MB'
                           % (max_memory))
    dmcore = np.dot(mo[:,:ncore], mo[:,:ncore].conj().T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(np.dot, (mo.T, vj*2-vk, mo))
    h1eff = reduce(np.dot, (mo.conj().T, mc.get_hcore(), mo)) + vhfcore

    # Assemble a dictionary of ERIs
    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = h1eff
    return eris
