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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
UCCSD(T)
'''


import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd

def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    cpu1 = cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nocca, noccb = mycc.nocc
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb

    if mycc.incore_complete:
        ftmp = None
    else:
        ftmp = lib.H5TmpFile()
    t1aT = t1a.T.copy()
    t1bT = t1b.T.copy()
    t2aaT = t2aa.transpose(2,3,0,1).copy()
    t2bbT = t2bb.transpose(2,3,0,1).copy()

    eris_vooo = numpy.asarray(eris.ovoo).transpose(1,3,0,2).conj().copy()
    eris_VOOO = numpy.asarray(eris.OVOO).transpose(1,3,0,2).conj().copy()
    eris_vOoO = numpy.asarray(eris.ovOO).transpose(1,3,0,2).conj().copy()
    eris_VoOo = numpy.asarray(eris.OVoo).transpose(1,3,0,2).conj().copy()

    eris_vvop, eris_VVOP, eris_vVoP, eris_VvOp = _sort_eri(mycc, eris, ftmp, log)
    cpu1 = log.timer_debug1('UCCSD(T) sort_eri', *cpu1)

    dtype = numpy.result_type(t1a.dtype, t2aa.dtype, eris_vooo.dtype)
    et_sum = numpy.zeros(1, dtype=dtype)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    # aaa
    bufsize = max(8, int((max_memory*.5e6/8-nocca**3*3*lib.num_threads())*.4/max(1,nocca*nmoa)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(nocca, dtype=int)
    contract = _gen_contract_aaa(t1aT, t2aaT, eris_vooo, eris.focka,
                                 eris.mo_energy[0], orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvira, bufsize))):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/8):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_aaa', *cpu1)

    # bbb
    bufsize = max(8, int((max_memory*.5e6/8-noccb**3*3*lib.num_threads())*.4/max(1,noccb*nmob)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(noccb, dtype=int)
    contract = _gen_contract_aaa(t1bT, t2bbT, eris_VOOO, eris.fockb,
                                 eris.mo_energy[1], orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvirb, bufsize))):
            cache_row_a = numpy.asarray(eris_VVOP[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_VVOP[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/8):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_bbb', *cpu1)

    # Premature termination for fully spin-polarized systems
    if nocca*noccb == 0:
        et_sum *= .25
        if abs(et_sum[0].imag) > 1e-4:
            logger.warn(mycc, 'Non-zero imaginary part of UCCSD(T) energy was found %s',
                        et_sum[0])
        et = et_sum[0].real
        log.timer('UCCSD(T)', *cpu0)
        log.note('UCCSD(T) correction = %.15g', et)
        return et

    # Cache t2abT in t2ab to reduce memory footprint
    assert (t2ab.flags.c_contiguous)
    t2abT = lib.transpose(t2ab.copy().reshape(nocca*noccb,nvira*nvirb), out=t2ab)
    t2abT = t2abT.reshape(nvira,nvirb,nocca,noccb)
    # baa
    bufsize = int(max(12, (max_memory*.5e6/8-noccb*nocca**2*5)*.7/max(1,nocca*nmob)))
    ts = t1aT, t1bT, t2aaT, t2abT
    fock = (eris.focka, eris.fockb)
    vooo = (eris_vooo, eris_vOoO, eris_VoOo)
    contract = _gen_contract_baa(ts, vooo, fock, eris.mo_energy, orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in lib.prange(0, nvirb, int(bufsize/nvira+1)):
            cache_row_a = numpy.asarray(eris_VvOp[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_vVoP[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvira, bufsize/6/2):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_baa', *cpu1)

    t2baT = numpy.ndarray((nvirb,nvira,noccb,nocca), buffer=t2abT,
                          dtype=t2abT.dtype)
    t2baT[:] = t2abT.copy().transpose(1,0,3,2)
    # abb
    ts = t1bT, t1aT, t2bbT, t2baT
    fock = (eris.fockb, eris.focka)
    mo_energy = (eris.mo_energy[1], eris.mo_energy[0])
    vooo = (eris_VOOO, eris_VoOo, eris_vOoO)
    contract = _gen_contract_baa(ts, vooo, fock, mo_energy, orbsym, log)
    for a0, a1 in lib.prange(0, nvira, int(bufsize/nvirb+1)):
        with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
            cache_row_a = numpy.asarray(eris_vVoP[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_VvOp[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvirb, bufsize/6/2):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_abb', *cpu1)

    # Restore t2ab
    lib.transpose(t2baT.transpose(1,0,3,2).copy().reshape(nvira*nvirb,nocca*noccb),
                  out=t2ab)
    et_sum *= .25
    if abs(et_sum[0].imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part of UCCSD(T) energy was found %s',
                    et_sum[0])
    et = et_sum[0].real
    log.timer('UCCSD(T)', *cpu0)
    log.note('UCCSD(T) correction = %.15g', et)
    return et

def _gen_contract_aaa(t1T, t2T, vooo, fock, mo_energy, orbsym, log):
    nvir, nocc = t1T.shape
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = fock[nocc:,:nocc].copy()

    cpu2 = [logger.process_clock(), logger.perf_counter()]
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    if len(oo_sym) == 0:
        nirrep = 0
    else:
        nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    dtype = numpy.result_type(t2T.dtype, vooo.dtype, fock.dtype)
    if dtype == numpy.complex128:
        drv = _ccsd.libcc.CCuccsd_t_zaaa
    else:
        drv = _ccsd.libcc.CCuccsd_t_aaa
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
    return contract

def _gen_contract_baa(ts, vooo, fock, mo_energy, orbsym, log):
    t1aT, t1bT, t2aaT, t2abT = ts
    focka, fockb = fock
    vooo, vOoO, VoOo = vooo
    nvira, nocca = t1aT.shape
    nvirb, noccb = t1bT.shape
    mo_ea = numpy.asarray(mo_energy[0], order='C')
    mo_eb = numpy.asarray(mo_energy[1], order='C')
    fvo = focka[nocca:,:nocca].copy()
    fVO = fockb[noccb:,:noccb].copy()

    cpu2 = [logger.process_clock(), logger.perf_counter()]
    dtype = numpy.result_type(t2aaT.dtype, vooo.dtype)
    if dtype == numpy.complex128:
        drv = _ccsd.libcc.CCuccsd_t_zbaa
    else:
        drv = _ccsd.libcc.CCuccsd_t_baa
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_ea.ctypes.data_as(ctypes.c_void_p),
            mo_eb.ctypes.data_as(ctypes.c_void_p),
            t1aT.ctypes.data_as(ctypes.c_void_p),
            t1bT.ctypes.data_as(ctypes.c_void_p),
            t2aaT.ctypes.data_as(ctypes.c_void_p),
            t2abT.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            vOoO.ctypes.data_as(ctypes.c_void_p),
            VoOo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            fVO.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocca), ctypes.c_int(noccb),
            ctypes.c_int(nvira), ctypes.c_int(nvirb),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
    return contract

def _sort_eri(mycc, eris, h5tmp, log):
    cpu1 = (logger.process_clock(), logger.perf_counter())
    nocca, noccb = mycc.nocc
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb

    if mycc.t2 is None:
        dtype = eris.ovov.dtype
    else:
        dtype = numpy.result_type(mycc.t2[0], eris.ovov.dtype)

    if mycc.incore_complete or h5tmp is None:
        eris_vvop = numpy.empty((nvira,nvira,nocca,nmoa), dtype)
        eris_VVOP = numpy.empty((nvirb,nvirb,noccb,nmob), dtype)
        eris_vVoP = numpy.empty((nvira,nvirb,nocca,nmob), dtype)
        eris_VvOp = numpy.empty((nvirb,nvira,noccb,nmoa), dtype)
    else:
        eris_vvop = h5tmp.create_dataset('vvop', (nvira,nvira,nocca,nmoa), dtype)
        eris_VVOP = h5tmp.create_dataset('VVOP', (nvirb,nvirb,noccb,nmob), dtype)
        eris_vVoP = h5tmp.create_dataset('vVoP', (nvira,nvirb,nocca,nmob), dtype)
        eris_VvOp = h5tmp.create_dataset('VvOp', (nvirb,nvira,noccb,nmoa), dtype)

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.9)

    blksize = min(nvira, max(16, int(max_memory*1e6/8/max(1,nvira*nocca*nmoa))))
    with lib.call_in_background(eris_vvop.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((nocca,nmoa,nvira), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvira, blksize):
            ovov = numpy.asarray(eris.ovov[:,j0:j1])
            ovvv = eris.get_ovvv(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                bufopv[:,:nocca,:] = ovov[:,j-j0].conj()
                bufopv[:,nocca:,:] = ovvv[:,j-j0].conj()
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvirb, max(16, int(max_memory*1e6/8/max(1,nvirb*noccb*nmob))))
    with lib.call_in_background(eris_VVOP.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((noccb,nmob,nvirb), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvirb, blksize):
            ovov = numpy.asarray(eris.OVOV[:,j0:j1])
            ovvv = eris.get_OVVV(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                bufopv[:,:noccb,:] = ovov[:,j-j0].conj()
                bufopv[:,noccb:,:] = ovvv[:,j-j0].conj()
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvira, max(16, int(max_memory*1e6/8/max(1,nvirb*nocca*nmob))))
    with lib.call_in_background(eris_vVoP.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((nocca,nmob,nvirb), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvira, blksize):
            ovov = numpy.asarray(eris.ovOV[:,j0:j1])
            ovvv = eris.get_ovVV(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                bufopv[:,:noccb,:] = ovov[:,j-j0].conj()
                bufopv[:,noccb:,:] = ovvv[:,j-j0].conj()
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvirb, max(16, int(max_memory*1e6/8/max(1,nvira*noccb*nmoa))))
    OVov = numpy.asarray(eris.ovOV).transpose(2,3,0,1)
    with lib.call_in_background(eris_VvOp.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((noccb,nmoa,nvira), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvirb, blksize):
            ovov = OVov[:,j0:j1]
            ovvv = eris.get_OVvv(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                bufopv[:,:nocca,:] = ovov[:,j-j0].conj()
                bufopv[:,nocca:,:] = ovvv[:,j-j0].conj()
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)
    return eris_vvop, eris_VVOP, eris_vVoP, eris_VvOp


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-12
    mcc.ccsd()
    t1a = t1b = mcc.t1
    t2ab = mcc.t2
    t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
    mycc = cc.UCCSD(scf.addons.convert_to_uhf(rhf))
    eris = mycc.ao2mo()
    e3a = kernel(mycc, eris, (t1a,t1b), (t2aa,t2ab,t2bb))
    print(e3a - -0.00099642337843278096)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(10)
    mf.mo_coeff = numpy.random.random((2,nao,nmo))

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    t1a  = .1 * numpy.random.random((nocca,nvira))
    t1b  = .1 * numpy.random.random((noccb,nvirb))
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mycc = cc.UCCSD(mf)
    eris = mycc.ao2mo(mf.mo_coeff)
    e3a = kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 9877.2780859693339)

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    t1 = mycc.spatial2spin(t1, eris.orbspin)
    t2 = mycc.spatial2spin(t2, eris.orbspin)
    from pyscf.cc import gccsd_t_slow
    et = gccsd_t_slow.kernel(mycc, eris, t1, t2)
    print(et - 9877.2780859693339)
