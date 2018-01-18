#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd

'''
UCCSD(T)
'''

def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    mo_ea = eris.focka.diagonal().copy()
    mo_eb = eris.fockb.diagonal().copy()

    ftmp = lib.H5TmpFile()
    ftmp['t2ab'] = t2ab
    t1aT = t1a.T.copy()
    t1bT = t1b.T.copy()
    t2aaT = t2aa.transpose(2,3,0,1).copy()
    t2bbT = t2bb.transpose(2,3,0,1).copy()

    eris_vooo = numpy.asarray(eris.ovoo).transpose(1,2,0,3).copy()
    eris_VOOO = numpy.asarray(eris.OVOO).transpose(1,2,0,3).copy()
    eris_vOoO = numpy.asarray(eris.ovOO).transpose(1,2,0,3).copy()
    eris_VoOo = numpy.asarray(eris.OVoo).transpose(1,2,0,3).copy()

    _sort_eri(mycc, eris, ftmp, log)
    eris_vvop = ftmp['vvop']
    eris_VVOP = ftmp['VVOP']
    eris_vVoP = ftmp['vVoP']
    eris_VvOp = ftmp['VvOp']
    cpu1 = log.timer_debug1('UCCSD(T) sort_eri', *cpu1)

    et_sum = [0]
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mycc.max_memory - mem_now)
    # aaa
    bufsize = max(1, int((max_memory*1e6/8-nocca**3*100)*.7/(nocca*nmoa)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(mo_ea.size, dtype=int)
    contract = _gen_contract_aaa(t1aT, t2aaT, eris_vooo, mo_ea, orbsym, log)
    for a0, a1 in reversed(list(lib.prange_tril(0, nvira, bufsize))):
        with lib.call_in_background(contract) as ctr:
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/6):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
                cache_row_b = cache_col_b = None
            cache_row_a = cache_col_a = None
    cpu1 = log.timer_debug1('contract_aaa', *cpu1)

    # bbb
    bufsize = max(1, int((max_memory*1e6/8-noccb**3*100)*.7/(noccb*nmob)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(mo_eb.size, dtype=int)
    contract = _gen_contract_aaa(t1bT, t2bbT, eris_VOOO, mo_eb, orbsym, log)
    for a0, a1 in reversed(list(lib.prange_tril(0, nvirb, bufsize))):
        with lib.call_in_background(contract) as ctr:
            cache_row_a = numpy.asarray(eris_VVOP[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_VVOP[:a0,a0:a1], order='C')
            ctr(et_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/6):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
                cache_row_b = cache_col_b = None
            cache_row_a = cache_col_a = None
    cpu1 = log.timer_debug1('contract_bbb', *cpu1)

    # Cache t2abT in t2ab to reduce memory footprint
    t2abT = lib.transpose(t2ab.reshape(nocca*noccb,nvira*nvirb).copy(), out=t2ab)
    t2abT = t2abT.reshape(nvira,nvirb,nocca,noccb)
    # baa
    bufsize = max(1, int((max_memory*.9e6/8-noccb*nocca**2*7)*.3/nocca*nmob))
    ts = t1aT, t1bT, t2aaT, t2abT
    vooo = (eris_vooo, eris_vOoO, eris_VoOo)
    contract = _gen_contract_baa(ts, vooo, (mo_ea,mo_eb), orbsym, log)
    for a0, a1 in lib.prange(0, nvirb, int(bufsize/nvira+1)):
        with lib.call_in_background(contract) as ctr:
            cache_row_a = numpy.asarray(eris_VvOp[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_vVoP[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvira, bufsize):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
                cache_row_b = cache_col_b = None
            cache_row_a = cache_col_a = None
    cpu1 = log.timer_debug1('contract_baa', *cpu1)

    t2baT = numpy.ndarray((nvirb,nvira,noccb,nocca), buffer=t2abT)
    t2baT[:] = t2abT.copy().transpose(1,0,3,2)
    # abb
    ts = t1bT, t1aT, t2bbT, t2baT
    vooo = (eris_VOOO, eris_VoOo, eris_vOoO)
    contract = _gen_contract_baa(ts, vooo, (mo_eb,mo_ea), orbsym, log)
    for a0, a1 in lib.prange(0, nvira, int(bufsize/nvirb+1)):
        with lib.call_in_background(contract) as ctr:
            cache_row_a = numpy.asarray(eris_vVoP[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_VvOp[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvirb, bufsize):
                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
                cache_row_b = cache_col_b = None
            cache_row_a = cache_col_a = None
    cpu1 = log.timer_debug1('contract_abb', *cpu1)

    t2ab[:] = ftmp['t2ab']
    et = et_sum[0] * .25
    log.timer('UCCSD(T)', *cpu0)
    log.note('UCCSD(T) correction = %.15g', et)
    return et

def _gen_contract_aaa(t1T, t2T, vooo, mo_energy, orbsym, log):
    nvir, nocc = t1T.shape

    cpu2 = [time.clock(), time.time()]
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv = _ccsd.libcc.CCuccsd_t_aaa
        drv.restype = ctypes.c_double
        et = drv(mo_energy.ctypes.data_as(ctypes.c_void_p),
                 t1T.ctypes.data_as(ctypes.c_void_p),
                 t2T.ctypes.data_as(ctypes.c_void_p),
                 vooo.ctypes.data_as(ctypes.c_void_p),
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
        et_sum[0] += et
        return et
    return contract

def _gen_contract_baa(ts, vooo, mo_energy, orbsym, log):
    t1aT, t1bT, t2aaT, t2abT = ts
    vooo, vOoO, VoOo = vooo
    mo_ea, mo_eb = mo_energy
    nvira, nocca = t1aT.shape
    nvirb, noccb = t1bT.shape

    cpu2 = [time.clock(), time.time()]
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv = _ccsd.libcc.CCuccsd_t_baa
        drv.restype = ctypes.c_double
        et = drv(mo_ea.ctypes.data_as(ctypes.c_void_p),
                 mo_eb.ctypes.data_as(ctypes.c_void_p),
                 t1aT.ctypes.data_as(ctypes.c_void_p),
                 t1bT.ctypes.data_as(ctypes.c_void_p),
                 t2aaT.ctypes.data_as(ctypes.c_void_p),
                 t2abT.ctypes.data_as(ctypes.c_void_p),
                 vooo.ctypes.data_as(ctypes.c_void_p),
                 vOoO.ctypes.data_as(ctypes.c_void_p),
                 VoOo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nocca), ctypes.c_int(noccb),
                 ctypes.c_int(nvira), ctypes.c_int(nvirb),
                 ctypes.c_int(a0), ctypes.c_int(a1),
                 ctypes.c_int(b0), ctypes.c_int(b1),
                 cache_row_a.ctypes.data_as(ctypes.c_void_p),
                 cache_col_a.ctypes.data_as(ctypes.c_void_p),
                 cache_row_b.ctypes.data_as(ctypes.c_void_p),
                 cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
        et_sum[0] += et
        return et
    return contract

def _sort_eri(mycc, eris, h5tmp, log):
    cpu1 = (time.clock(), time.time())
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb

    eris_vvop = h5tmp.create_dataset('vvop', (nvira,nvira,nocca,nmoa), 'f8')
    eris_VVOP = h5tmp.create_dataset('VVOP', (nvirb,nvirb,noccb,nmob), 'f8')
    eris_vVoP = h5tmp.create_dataset('vVoP', (nvira,nvirb,nocca,nmob), 'f8')
    eris_VvOp = h5tmp.create_dataset('VvOp', (nvirb,nvira,noccb,nmoa), 'f8')

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.9)

    blksize = min(nvira, max(16, int(max_memory*1e6/8/(nvira*nocca*nmoa))))
    with lib.call_in_background(eris_vvop.__setitem__) as save:
        bufopv = numpy.empty((nocca,nmoa,nvira))
        buf1 = numpy.empty_like(bufopv)
        buf = numpy.empty((nocca,nvira,nvira))
        for j0, j1 in lib.prange(0, nvira, blksize):
            ovov = numpy.asarray(eris.ovov[:,j0:j1])
            ovvv = numpy.asarray(eris.ovvv[:,j0:j1])
            for j in range(j0,j1):
                oov = ovov[:,j-j0]
                ovv = lib.unpack_tril(ovvv[:,j-j0], out=buf)
                bufopv[:,:nocca,:] = oov
                bufopv[:,nocca:,:] = ovv
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvirb, max(16, int(max_memory*1e6/8/(nvirb*noccb*nmob))))
    with lib.call_in_background(eris_VVOP.__setitem__) as save:
        bufopv = numpy.empty((noccb,nmob,nvirb))
        buf1 = numpy.empty_like(bufopv)
        buf = numpy.empty((noccb,nvirb,nvirb))
        for j0, j1 in lib.prange(0, nvirb, blksize):
            ovov = numpy.asarray(eris.OVOV[:,j0:j1])
            ovvv = numpy.asarray(eris.OVVV[:,j0:j1])
            for j in range(j0,j1):
                oov = ovov[:,j-j0]
                ovv = lib.unpack_tril(ovvv[:,j-j0], out=buf)
                bufopv[:,:noccb,:] = oov
                bufopv[:,noccb:,:] = ovv
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvira, max(16, int(max_memory*1e6/8/(nvirb*nocca*nmob))))
    with lib.call_in_background(eris_vVoP.__setitem__) as save:
        bufopv = numpy.empty((nocca,nmob,nvirb))
        buf1 = numpy.empty_like(bufopv)
        buf = numpy.empty((nocca,nvirb,nvirb))
        for j0, j1 in lib.prange(0, nvira, blksize):
            ovov = numpy.asarray(eris.ovOV[:,j0:j1])
            ovvv = numpy.asarray(eris.ovVV[:,j0:j1])
            for j in range(j0,j1):
                oov = ovov[:,j-j0]
                ovv = lib.unpack_tril(ovvv[:,j-j0], out=buf)
                bufopv[:,:noccb,:] = oov
                bufopv[:,noccb:,:] = ovv
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    blksize = min(nvirb, max(16, int(max_memory*1e6/8/(nvira*noccb*nmoa))))
    with lib.call_in_background(eris_VvOp.__setitem__) as save:
        bufopv = numpy.empty((noccb,nmoa,nvira))
        buf1 = numpy.empty_like(bufopv)
        buf = numpy.empty((noccb,nvira,nvira))
        for j0, j1 in lib.prange(0, nvirb, blksize):
            ovvo = numpy.asarray(eris.OVvo[:,j0:j1])
            ovvv = numpy.asarray(eris.OVvv[:,j0:j1])
            for j in range(j0,j1):
                ovo = ovvo[:,j-j0]
                ovv = lib.unpack_tril(ovvv[:,j-j0], out=buf)
                bufopv[:,:nocca,:] = ovo.transpose(0,2,1)
                bufopv[:,nocca:,:] = ovv
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovvo = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)


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
    print(e3a - 8193.064821311109)

