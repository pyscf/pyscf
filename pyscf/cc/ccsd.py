#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
RCCSD for real integrals
8-fold permutation symmetry has been used
(ij|kl) = (ji|kl) = (kl|ij) = ...
'''


import ctypes
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import _ccsd
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


# t1: ia
# t2: ijab
def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None, callback=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        if callback is not None:
            callback(locals())
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = numpy.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = numpy.asarray(mycc.iterative_damping)
            if isinstance(t1, tuple): # e.g. UCCSD
                t1new = tuple((1-alpha) * numpy.asarray(t1_part) + alpha * numpy.asarray(t1new_part)
                    for t1_part, t1new_part in zip(t1, t1new))
                t2new = tuple((1-alpha) * numpy.asarray(t2_part) + alpha * numpy.asarray(t2new_part)
                    for t2_part, t2new_part in zip(t2, t2new))
            else:
                t1new = (1-alpha) * numpy.asarray(t1) + alpha * numpy.asarray(t1new)
                t2new *= alpha
                t2new += (1-alpha) * numpy.asarray(t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        mycc.cycles = istep + 1
        log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, name, eccsd, eccsd - eold, normt)
        cput1 = log.timer(f'{name} iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, eccsd, t1, t2


def update_amps(mycc, t1, t2, eris):
    if mycc.cc2:
        raise NotImplementedError
    assert (isinstance(eris, _ChemistsERIs))

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    t1new = numpy.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris, t2sym='jiba')
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc] - numpy.diag(mo_e_o)
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:] - numpy.diag(mo_e_v)
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    if mycc.incore_complete:
        fswap = None
    else:
        fswap = lib.H5TmpFile()
    fwVOov, fwVooV = _add_ovvv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap)
    time1 = log.timer_debug1('ovvv', *time1)

    woooo = numpy.asarray(eris.oooo).transpose(0,2,1,3).copy()

    unit = nocc**2*nvir*7 + nocc**3 + nocc*nvir**2
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-nocc**4)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    for p0, p1 in lib.prange(0, nvir, blksize):
        wVOov = fwVOov[p0:p1]
        wVooV = fwVooV[p0:p1]
        eris_ovoo = eris.ovoo[:,p0:p1]
        eris_oovv = numpy.empty((nocc,nocc,p1-p0,nvir))
        def load_oovv(p0, p1):
            eris_oovv[:] = eris.oovv[:,:,p0:p1]
        with lib.call_in_background(load_oovv, sync=not mycc.async_io) as prefetch_oovv:
            #:eris_oovv = eris.oovv[:,:,p0:p1]
            prefetch_oovv(p0, p1)
            foo += numpy.einsum('kc,kcji->ij', 2*t1[:,p0:p1], eris_ovoo)
            foo += numpy.einsum('kc,icjk->ij',  -t1[:,p0:p1], eris_ovoo)
            tmp = lib.einsum('la,jaik->lkji', t1[:,p0:p1], eris_ovoo)
            woooo += tmp + tmp.transpose(1,0,3,2)
            tmp = None

            wVOov -= lib.einsum('jbik,ka->bjia', eris_ovoo, t1)
            t2new[:,:,p0:p1] += wVOov.transpose(1,2,0,3)

            wVooV += lib.einsum('kbij,ka->bija', eris_ovoo, t1)
            eris_ovoo = None
        load_oovv = prefetch_oovv = None

        eris_ovvo = numpy.empty((nocc,p1-p0,nvir,nocc))
        def load_ovvo(p0, p1):
            eris_ovvo[:] = eris.ovvo[:,p0:p1]
        with lib.call_in_background(load_ovvo, sync=not mycc.async_io) as prefetch_ovvo:
            #:eris_ovvo = eris.ovvo[:,p0:p1]
            prefetch_ovvo(p0, p1)
            t1new[:,p0:p1] -= numpy.einsum('jb,jiab->ia', t1, eris_oovv)
            wVooV -= eris_oovv.transpose(2,0,1,3)
            wVOov += wVooV*.5  #: bjia + bija*.5
        load_ovvo = prefetch_ovvo = None

        t2new[:,:,p0:p1] += (eris_ovvo*0.5).transpose(0,3,1,2)
        eris_voov = eris_ovvo.conj().transpose(1,0,3,2)
        t1new[:,p0:p1] += 2*numpy.einsum('jb,aijb->ia', t1, eris_voov)
        eris_ovvo = None

        tmp  = lib.einsum('ic,kjbc->ibkj', t1, eris_oovv)
        tmp += lib.einsum('bjkc,ic->jbki', eris_voov, t1)
        t2new[:,:,p0:p1] -= lib.einsum('ka,jbki->jiba', t1, tmp)
        eris_oovv = tmp = None

        fov[:,p0:p1] += numpy.einsum('kc,aikc->ia', t1, eris_voov) * 2
        fov[:,p0:p1] -= numpy.einsum('kc,akic->ia', t1, eris_voov)

        tau  = numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
        tau += t2[:,:,p0:p1]
        theta  = tau.transpose(1,0,2,3) * 2
        theta -= tau
        fvv -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo += lib.einsum('aikb,kjab->ij', eris_voov, theta)
        tau = theta = None

        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo += lib.einsum('ijab,aklb->ijkl', tau, eris_voov)
        tau = None

        def update_wVooV(q0, q1, tau):
            wVooV[:] += lib.einsum('bkic,jkca->bija', eris_voov[:,:,:,q0:q1], tau)
        with lib.call_in_background(update_wVooV, sync=not mycc.async_io) as update_wVooV:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tau  = t2[:,:,q0:q1] * .5
                tau += numpy.einsum('ia,jb->ijab', t1[:,q0:q1], t1)
                #:wVooV += lib.einsum('bkic,jkca->bija', eris_voov[:,:,:,q0:q1], tau)
                update_wVooV(q0, q1, tau)
        tau = update_wVooV = None
        def update_t2(q0, q1, tmp):
            t2new[:,:,q0:q1] += tmp.transpose(2,0,1,3)
            tmp *= .5
            t2new[:,:,q0:q1] += tmp.transpose(0,2,1,3)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tmp = lib.einsum('jkca,ckib->jaib', t2[:,:,p0:p1,q0:q1], wVooV)
                #:t2new[:,:,q0:q1] += tmp.transpose(2,0,1,3)
                #:tmp *= .5
                #:t2new[:,:,q0:q1] += tmp.transpose(0,2,1,3)
                update_t2(q0, q1, tmp)
                tmp = None

        wVOov += eris_voov
        eris_VOov = -.5 * eris_voov.transpose(0,2,1,3)
        eris_VOov += eris_voov
        eris_voov = None
        def update_wVOov(q0, q1, tau):
            wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau)
        with lib.call_in_background(update_wVOov, sync=not mycc.async_io) as update_wVOov:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tau  = t2[:,:,q0:q1].transpose(1,3,0,2) * 2
                tau -= t2[:,:,q0:q1].transpose(0,3,1,2)
                tau -= numpy.einsum('ia,jb->ibja', t1[:,q0:q1]*2, t1)
                #:wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau)
                update_wVOov(q0, q1, tau)
                tau = None
        def update_t2(q0, q1, theta):
            t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                theta  = t2[:,:,p0:p1,q0:q1] * 2
                theta -= t2[:,:,p0:p1,q0:q1].transpose(1,0,2,3)
                #:t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
                update_t2(q0, q1, theta)
                theta = None
        eris_VOov = wVOov = wVooV = update_wVOov = None
        time1 = log.timer_debug1('voov [%d:%d]'%(p0, p1), *time1)
    fwVOov = fwVooV = fswap = None

    for p0, p1 in lib.prange(0, nvir, blksize):
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        t1new += numpy.einsum('jb,ijba->ia', fov[:,p0:p1], theta)
        t1new -= lib.einsum('jbki,kjba->ia', eris.ovoo[:,p0:p1], theta)

        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        tau += t2[:,:,p0:p1]
        t2new[:,:,p0:p1] += .5 * lib.einsum('ijkl,klab->ijab', woooo, tau)
        theta = tau = None

    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    t2new += lib.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= lib.einsum('ki,kjab->ijab', ft_ij, t2)

    eia = mo_e_o[:,None] - mo_e_v
    t1new += numpy.einsum('ib,ab->ia', t1, fvv)
    t1new -= numpy.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2new[i,:i] += t2new[:i,i].transpose(0,2,1)
            t2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            t2new[:i,i] = t2new[i,:i].transpose(0,2,1)
        t2new[i,i] = t2new[i,i] + t2new[i,i].T
        t2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new


def _add_ovvv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir+1) // 2

    if fswap is None:
        wVOov = numpy.zeros((nvir,nocc,nocc,nvir))
    else:
        wVOov = fswap.create_dataset('wVOov', (nvir,nocc,nocc,nvir), 'f8')
    wooVV = numpy.zeros((nocc,nocc*nvir_pair))

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2*3 + nocc**2*nvir + 2
    blksize = min(nvir, max(BLKMIN, int((max_memory*.95e6/8-wooVV.size)/unit)))
    if not mycc.direct:
        unit = nocc*nvir**2*3 + nocc**2*nvir + 2 + nocc*nvir**2 + nocc*nvir
        blksize = min(nvir, max(BLKMIN, int((max_memory*.95e6/8-wooVV.size-nocc**2*nvir)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    def load_ovvv(buf, p0):
        if p0 < nvir:
            p1 = min(nvir, p0+blksize)
            buf[:p1-p0] = eris.ovvv[:,p0:p1].transpose(1,0,2)

    with lib.call_in_background(load_ovvv, sync=not mycc.async_io) as prefetch:
        buf = numpy.empty((blksize,nocc,nvir_pair))
        buf_prefetch = numpy.empty((blksize,nocc,nvir_pair))

        load_ovvv(buf_prefetch, 0)
        for p0, p1 in lib.prange(0, nvir, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, p1)

            eris_vovv = buf[:p1-p0]

            #:wooVV -= numpy.einsum('jc,ciba->jiba', t1[:,p0:p1], eris_vovv)
            lib.ddot(numpy.asarray(t1[:,p0:p1], order='C'),
                     eris_vovv.reshape(p1-p0,-1), -1, wooVV, 1)

            eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,nvir_pair))
            eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)

            fvv += 2*numpy.einsum('kc,ckab->ab', t1[:,p0:p1], eris_vovv)
            fvv[:,p0:p1] -= numpy.einsum('kc,bkca->ab', t1, eris_vovv)

            if not mycc.direct:
                vvvo = eris_vovv.transpose(0,2,3,1).copy()
                for i in range(nocc):
                    tau = t2[i,:,p0:p1] + numpy.einsum('a,jb->jab', t1[i,p0:p1], t1)
                    tmp = lib.einsum('jcd,cdbk->jbk', tau, vvvo)
                    t2new[i] -= lib.einsum('ka,jbk->jab', t1, tmp)
                    tau = tmp = None

            wVOov[p0:p1] = lib.einsum('biac,jc->bija', eris_vovv, t1)

            theta = t2[:,:,p0:p1].transpose(1,2,0,3) * 2
            theta -= t2[:,:,p0:p1].transpose(0,2,1,3)
            t1new += lib.einsum('icjb,cjba->ia', theta, eris_vovv)
            theta = None
            time1 = log.timer_debug1('vovv [%d:%d]'%(p0, p1), *time1)

    if fswap is None:
        wooVV = lib.unpack_tril(wooVV.reshape(nocc**2,nvir_pair))
        return wVOov, wooVV.reshape(nocc,nocc,nvir,nvir).transpose(2,1,0,3)
    else:
        fswap.create_dataset('wVooV', (nvir,nocc,nocc,nvir), 'f8')
        wooVV = wooVV.reshape(nocc,nocc,nvir_pair)
        tril2sq = lib.square_mat_in_trilu_indices(nvir)
        for p0, p1 in lib.prange(0, nvir, blksize):
            fswap['wVooV'][p0:p1] = wooVV[:,:,tril2sq[p0:p1]].transpose(2,1,0,3)
        return fswap['wVOov'], fswap['wVooV']

def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=None, t2sym=None):
    '''t2sym: whether t2 has the symmetry t2[ijab]==t2[jiba] or
    t2[ijab]==-t2[jiab] or t2[ijab]==-t2[jiba]
    '''
    #TODO: Guess the symmetry of t2 amplitudes
    #if t2sym is None:
    #    if t2.shape[0] != t2.shape[1]:
    #        t2sym = ''
    #    elif abs(t2-t2.transpose(1,0,3,2)).max() < 1e-12:
    #        t2sym = 'jiba'
    #    elif abs(t2+t2.transpose(1,0,2,3)).max() < 1e-12:
    #        t2sym = '-jiab'
    #    elif abs(t2+t2.transpose(1,0,3,2)).max() < 1e-12:
    #        t2sym = '-jiba'

    if t2sym in ('jiba', '-jiba', '-jiab'):
        Ht2tril = _add_vvvv_tril(mycc, t1, t2, eris, with_ovvv=with_ovvv)
        nocc, nvir = t2.shape[1:3]
        Ht2 = _unpack_t2_tril(Ht2tril, nocc, nvir, out, t2sym)
    else:
        Ht2 = _add_vvvv_full(mycc, t1, t2, eris, out, with_ovvv)
    return Ht2

def _add_vvvv_tril(mycc, t1, t2, eris, out=None, with_ovvv=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    Using symmetry t2[ijab] = t2[jiba] and Ht2[ijab] = Ht2[jiba], compute the
    lower triangular part of  Ht2
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    if with_ovvv is None:
        with_ovvv = mycc.direct
    nocc, nvir = t2.shape[1:3]
    nocc2 = nocc*(nocc+1)//2
    if t1 is None:
        tau = t2[numpy.tril_indices(nocc)]
    else:
        tau = numpy.empty((nocc2,nvir,nvir), dtype=t2.dtype)
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]

    if mycc.direct:   # AO-direct CCSD
        mo = getattr(eris, 'mo_coeff', None)
        if mo is None:  # If eris does not have the attribute mo_coeff
            mo = _mo_without_core(mycc, mycc.mo_coeff)
        nao, nmo = mo.shape
        aos = numpy.asarray(mo[:,nocc:].T, order='F')
        tau = _ao2mo.nr_e2(tau.reshape(nocc2,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
        tau = tau.reshape(nocc2,nao,nao)
        time0 = log.timer_debug1('vvvv-tau', *time0)

        buf = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out, log)
        buf = buf.reshape(nocc2,nao,nao)
        Ht2tril = _ao2mo.nr_e2(buf, mo.conj(), (nocc,nmo,nocc,nmo), 's1', 's1')
        Ht2tril = Ht2tril.reshape(nocc2,nvir,nvir)

        if with_ovvv:
            #: tmp = numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
            #: t2new -= tmp + tmp.transpose(1,0,3,2)
            tmp = _ao2mo.nr_e2(buf, mo.conj(), (nocc,nmo,0,nocc), 's1', 's1')
            Ht2tril -= lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1).reshape(nocc2,nvir,nvir)
            tmp = _ao2mo.nr_e2(buf, mo.conj(), (0,nocc,nocc,nmo), 's1', 's1')
            #: Ht2tril -= numpy.einsum('xkb,ka->xab', tmp.reshape(-1,nocc,nvir), t1)
            tmp = lib.transpose(tmp.reshape(nocc2,nocc,nvir), axes=(0,2,1), out=buf)
            tmp = lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1, 1,
                           numpy.ndarray((nocc2*nvir,nvir), buffer=tau), 0)
            tmp = lib.transpose(tmp.reshape(nocc2,nvir,nvir), axes=(0,2,1), out=buf)
            Ht2tril -= tmp.reshape(nocc2,nvir,nvir)
    else:
        assert (not with_ovvv)
        Ht2tril = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out, log)
    return Ht2tril

def _add_vvvv_full(mycc, t1, t2, eris, out=None, with_ovvv=False):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    without using symmetry t2[ijab] = t2[jiba] in t2 or Ht2
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    if t1 is None:
        tau = t2
    else:
        tau = numpy.einsum('ia,jb->ijab', t1, t1)
        tau += t2

    if mycc.direct:   # AO-direct CCSD
        if with_ovvv:
            raise NotImplementedError
        mo = getattr(eris, 'mo_coeff', None)
        if mo is None:  # If eris does not have the attribute mo_coeff
            mo = _mo_without_core(mycc, mycc.mo_coeff)
        nocc, nvir = t2.shape[1:3]
        nao, nmo = mo.shape
        aos = numpy.asarray(mo[:,nocc:].T, order='F')
        tau = _ao2mo.nr_e2(tau.reshape(nocc**2,nvir,nvir), aos, (0,nao,0,nao), 's1', 's1')
        tau = tau.reshape(nocc,nocc,nao,nao)
        time0 = log.timer_debug1('vvvv-tau mo2ao', *time0)

        buf = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out, log)
        buf = buf.reshape(nocc**2,nao,nao)
        Ht2 = _ao2mo.nr_e2(buf, mo.conj(), (nocc,nmo,nocc,nmo), 's1', 's1')
    else:
        assert (not with_ovvv)
        Ht2 = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out, log)

    return Ht2.reshape(t2.shape)


def _contract_vvvv_t2(mycc, mol, vvvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    if vvvv is None or len(vvvv.shape) == 2:
        # AO-direct or vvvv in 4-fold symmetry
        return _contract_s4vvvv_t2(mycc, mol, vvvv, t2, out, verbose)
    else:
        return _contract_s1vvvv_t2(mycc, mol, vvvv, t2, out, verbose)


def _contract_s4vvvv_t2(mycc, mol, vvvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    where vvvv has to be real and has the 4-fold permutation symmetry

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    assert (t2.dtype == numpy.double)
    if t2.size == 0:
        return numpy.zeros_like(t2)

    _dgemm = lib.numpy_helper._dgemm
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mycc, verbose)

    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    nvir2 = nvira * nvirb
    Ht2 = numpy.ndarray(x2.shape, dtype=x2.dtype, buffer=out)
    Ht2[:] = 0

    def contract_blk_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        #:Ht2[:,j0:j1] += numpy.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm('N', 'N', nocc2, jc*nvirb, ic*nvirb,
               x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
               Ht2.reshape(-1,nvir2), 1, 1, i0*nvirb, 0, j0*nvirb)

        if i0 > j0:
            #:Ht2[:,i0:i1] += numpy.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm('N', 'T', nocc2, ic*nvirb, jc*nvirb,
                   x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
                   Ht2.reshape(-1,nvir2), 1, 1, j0*nvirb, 0, i0*nvirb)

    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    if vvvv is None:   # AO-direct CCSD
        ao_loc = mol.ao_loc_nr()
        assert (nvira == nvirb == ao_loc[-1])

        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        blksize = max(BLKMIN, numpy.sqrt(max_memory*.9e6/8/nvirb**2/2.5))
        blksize = int(min((nvira+3)/4, blksize))
        sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
        blksize = max(x[2] for x in sh_ranges)
        eribuf = numpy.empty((blksize,blksize,nvirb,nvirb))
        loadbuf = numpy.empty((blksize,blksize,nvirb,nvirb))
        fint = gto.moleintor.getints4c

        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip]:
                eri = fint(intor, mol._atm, mol._bas, mol._env,
                           shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                           ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                tmp = numpy.ndarray((i1-i0,nvirb,j1-j0,nvirb), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nvirb))
                contract_blk_(tmp, i0, i1, j0, j1)
                time0 = log.timer_debug1('AO-vvvv [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time0)

            eri = fint(intor, mol._atm, mol._bas, mol._env,
                       shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                       ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            eri = lib.unpack_tril(eri, axis=0)
            tmp = numpy.ndarray((i1-i0,nvirb,i1-i0,nvirb), buffer=loadbuf)
            _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                   eri.ctypes.data_as(ctypes.c_void_p),
                                   (ctypes.c_int*4)(i0, i1, i0, i1),
                                   ctypes.c_int(nvirb))
            eri = None
            contract_blk_(tmp, i0, i1, i0, i1)
            time0 = log.timer_debug1('AO-vvvv [%d:%d,%d:%d]' %
                                     (ish0,ish1,ish0,ish1), *time0)

    else:
        nvir_pair = nvirb * (nvirb+1) // 2
        unit = nvira*nvir_pair*2 + nvirb**2*nvira/4 + 1

        if mycc.async_io:
            fmap = lib.map_with_prefetch
            unit += nvira*nvir_pair
        else:
            fmap = map

        blksize = numpy.sqrt(max(BLKMIN**2, max_memory*.95e6/8/unit))
        blksize = int(min((nvira+3)/4, blksize))

        def load(v_slice):
            i0, i1 = v_slice
            off0 = i0*(i0+1)//2
            off1 = i1*(i1+1)//2
            return numpy.asarray(vvvv[off0:off1], order='C')

        tril2sq = lib.square_mat_in_trilu_indices(nvira)
        loadbuf = numpy.empty((blksize,blksize,nvirb,nvirb))

        slices = list(lib.prange(0, nvira, blksize))
        for istep, wwbuf in enumerate(fmap(load, lib.prange(0, nvira, blksize))):
            i0, i1 = slices[istep]
            off0 = i0*(i0+1)//2
            for j0, j1 in lib.prange(0, i1, blksize):
                eri = wwbuf[tril2sq[i0:i1,j0:j1]-off0]
                tmp = numpy.ndarray((i1-i0,nvirb,j1-j0,nvirb), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nvirb))
                contract_blk_(tmp, i0, i1, j0, j1)
            wwbuf = None
            time0 = log.timer_debug1('vvvv [%d:%d]'%(i0,i1), *time0)
    return Ht2.reshape(t2.shape)

def _contract_s1vvvv_t2(mycc, mol, vvvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    where vvvv can be real or complex and no permutation symmetry is available in vvvv.

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    # vvvv == None means AO-direct CCSD. It should redirect to
    # _contract_s4vvvv_t2(mycc, mol, vvvv, t2, out, verbose)
    assert (vvvv is not None)
    if t2.size == 0:
        return numpy.zeros_like(t2)

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mycc, verbose)

    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    dtype = numpy.result_type(t2, vvvv)
    Ht2 = numpy.ndarray(x2.shape, dtype=dtype, buffer=out)

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nvirb**2*nvira*2 + nocc2*nvirb + 1
    blksize = min(nvira, max(BLKMIN, int(max_memory*1e6/8/unit)))

    for p0,p1 in lib.prange(0, nvira, blksize):
        Ht2[:,p0:p1] = lib.einsum('xcd,acbd->xab', x2, vvvv[p0:p1])
        time0 = log.timer_debug1('vvvv [%d:%d]' % (p0,p1), *time0)
    return Ht2.reshape(t2.shape)

def _unpack_t2_tril(t2tril, nocc, nvir, out=None, t2sym='jiba'):
    t2 = numpy.ndarray((nocc,nocc,nvir,nvir), dtype=t2tril.dtype, buffer=out)
    idx,idy = numpy.tril_indices(nocc)
    if t2sym == 'jiba':
        t2[idy,idx] = t2tril.transpose(0,2,1)
        t2[idx,idy] = t2tril
    elif t2sym == '-jiba':
        t2[idy,idx] = -t2tril.transpose(0,2,1)
        t2[idx,idy] = t2tril
    elif t2sym == '-jiab':
        t2[idy,idx] =-t2tril
        t2[idx,idy] = t2tril
        t2[numpy.diag_indices(nocc)] = 0
    return t2

def _unpack_4fold(c2vec, nocc, nvir, anti_symm=True):
    t2 = numpy.zeros((nocc**2,nvir**2), dtype=c2vec.dtype)
    if nocc > 1 and nvir > 1:
        t2tril = c2vec.reshape(nocc*(nocc-1)//2,nvir*(nvir-1)//2)
        otril = numpy.tril_indices(nocc, k=-1)
        vtril = numpy.tril_indices(nvir, k=-1)
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[0]*nvir+vtril[1])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[1]*nvir+vtril[0])
        if anti_symm:  # anti-symmetry when exchanging two particle indices
            t2tril = -t2tril
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[1]*nvir+vtril[0])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[0]*nvir+vtril[1])
    return t2.reshape(nocc,nocc,nvir,nvir)

def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nov*(nov+1)//2
    vector = numpy.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    lib.pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov), out=vector[nov:])
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].copy().reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, numpy.asarray(t2, order='C')

def amplitudes_to_vector_s4(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    vector = numpy.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    otril = numpy.tril_indices(nocc, k=-1)
    vtril = numpy.tril_indices(nvir, k=-1)
    lib.take_2d(t2.reshape(nocc**2,nvir**2), otril[0]*nocc+otril[1],
                vtril[0]*nvir+vtril[1], out=vector[nov:])
    return vector

def vector_to_amplitudes_s4(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    t1 = vector[:nov].copy().reshape(nocc,nvir)
    t2 = numpy.zeros((nocc,nocc,nvir,nvir), dtype=vector.dtype)
    t2 = _unpack_4fold(vector[nov:size], nocc, nvir)
    return t1, t2


def energy(mycc, t1=None, t2=None, eris=None):
    '''CCSD correlation energy'''
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvo = eris.ovvo[:,p0:p1]
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        e += 2 * numpy.einsum('ijab,iabj', tau, eris_ovvo)
        e -=     numpy.einsum('jiab,iabj', tau, eris_ovvo)
    if abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in %s energy %s',
                    mycc.__class__.__name__, e)
    return e.real

def restore_from_diis_(mycc, diis_file, inplace=True):
    '''Reuse an existed DIIS object in the CCSD calculation.

    The CCSD amplitudes will be restored from the DIIS object to generate t1
    and t2 amplitudes. The t1/t2 amplitudes of the CCSD object will be
    overwritten by the generated t1 and t2 amplitudes. The amplitudes vector
    and error vector will be reused in the CCSD calculation.
    '''
    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    adiis.restore(diis_file, inplace=inplace)

    ccvec = adiis.extrapolate()
    mycc.t1, mycc.t2 = mycc.vector_to_amplitudes(ccvec)
    if inplace:
        mycc.diis = adiis
    return mycc

def get_t1_diagnostic(t1):
    '''Returns the t1 amplitude norm, normalized by number of correlated electrons.'''
    nelectron = 2 * t1.shape[0]
    return numpy.sqrt(numpy.linalg.norm(t1)**2 / nelectron)

def get_d1_diagnostic(t1):
    '''D1 diagnostic given in

        Janssen, et. al Chem. Phys. Lett. 290 (1998) 423
    '''
    f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]
    d1norm_ij = f(numpy.linalg.eigh(numpy.einsum('ia,ja->ij',t1,t1)))
    d1norm_ab = f(numpy.linalg.eigh(numpy.einsum('ia,ib->ab',t1,t1)))
    d1norm = max(d1norm_ij, d1norm_ab)
    return d1norm

def get_d2_diagnostic(t2):
    '''D2 diagnostic given in

        Nielsen, et. al Chem. Phys. Lett. 310 (1999) 568

    Note: This is currently only defined in the literature for restricted
    closed-shell systems.
    '''
    f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]
    d2norm_ij = f(numpy.linalg.eigh(numpy.einsum('ikab,jkab->ij',t2,t2)))
    d2norm_ab = f(numpy.linalg.eigh(numpy.einsum('ijac,ijbc->ab',t2,t2)))
    d2norm = max(d2norm_ij, d2norm_ab)
    return d2norm

def set_frozen(mycc, method='auto', window=(-1000.0, 1000.0), is_gcc=False):
    if method == 'auto':
        from pyscf.data import elements
        mycc.frozen = elements.chemcore(mycc.mol, spinorb=is_gcc)
    elif method == 'window':
        emin, emax = window
        mo_e = numpy.asarray(mycc._scf.mo_energy)
        if mo_e.ndim == 1:
            fr1 = list(numpy.flatnonzero(mo_e < emin))
            fr2 = list(numpy.flatnonzero(mo_e > emax))
            frozen = fr1 + fr2
        elif mo_e.ndim == 2:
            fr1a = list(numpy.flatnonzero(mo_e[0] < emin))
            fr2a = list(numpy.flatnonzero(mo_e[0] > emax))
            fr1b = list(numpy.flatnonzero(mo_e[1] < emin))
            fr2b = list(numpy.flatnonzero(mo_e[1] > emax))
            frozen = [fr1a+fr2a, fr1b+fr2b]
        mycc.frozen = frozen
    return mycc

def as_scanner(cc):
    '''Generating a scanner/solver for CCSD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CCSD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CCSD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, cc
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> cc_scanner = cc.CCSD(scf.RHF(mol)).as_scanner()
        >>> e_tot = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    if isinstance(cc, lib.SinglePointScanner):
        return cc

    logger.info(cc, 'Set %s as a scanner', cc.__class__)
    name = cc.__class__.__name__ + CCSD_Scanner.__name_mixin__
    return lib.set_class(CCSD_Scanner(cc), (CCSD_Scanner, cc.__class__), name)

class CCSD_Scanner(lib.SinglePointScanner):

    def __init__(self, cc):
        self.__dict__.update(cc.__dict__)
        self._scf = cc._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        if self.t2 is not None:
            last_size = self.vector_size()
        else:
            last_size = 0

        self.reset(mol)

        mf_scanner = self._scf
        mf_scanner(mol)
        self.mo_coeff = mf_scanner.mo_coeff
        self.mo_occ = mf_scanner.mo_occ
        if last_size != self.vector_size():
            self.t1 = self.t2 = None
        self.kernel(self.t1, self.t2, **kwargs)
        return self.e_tot


class CCSDBase(lib.StreamObject):
    '''restricted CCSD

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-7.
        conv_tol_normt : float
            converge threshold for norm(t1,t2).  Default is 1e-5.
        max_cycle : int
            max number of iterations.  Default is 50.
        diis_space : int
            DIIS space size.  Default is 6.
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        iterative_damping : float
            The self consistent damping parameter.
        direct : bool
            AO-direct CCSD. Default is False.
        async_io : bool
            Allow for asynchronous function execution. Default is True.
        incore_complete : bool
            Avoid all I/O (also for DIIS). Default is False.
        level_shift : float
            A shift on virtual orbital energies to stabilize the CCSD iteration
        frozen : int or list
            If integer is given, the inner-most orbitals are frozen from CC
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in CC calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> mycc = cc.CCSD(mf).set(frozen = 2).run()
            >>> # auto-generate the number of core orbitals to be frozen (1 in this case)
            >>> mycc = cc.CCSD(mf).set_frozen().run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> mycc.set(frozen = [0,1,16,17,18]).run()

        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.

    Saved results:

        converged : bool
            Whether the CCSD iteration converged
        e_corr : float
            CCSD correlation correction
        e_tot : float
            Total CCSD energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
        l1, l2 :
            Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
        cycles : int
            The number of iteration cycles performed
    '''

    max_cycle = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    iterative_damping = getattr(__config__, 'cc_ccsd_CCSD_iterative_damping', 1.0)
    conv_tol_normt = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)

    diis = getattr(__config__, 'cc_ccsd_CCSD_diis', True)
    diis_space = getattr(__config__, 'cc_ccsd_CCSD_diis_space', 6)
    diis_file = None
    diis_start_cycle = getattr(__config__, 'cc_ccsd_CCSD_diis_start_cycle', 0)
    # FIXME: Should we avoid DIIS starting early?
    diis_start_energy_diff = getattr(__config__, 'cc_ccsd_CCSD_diis_start_energy_diff', 1e9)

    direct = getattr(__config__, 'cc_ccsd_CCSD_direct', False)
    async_io = getattr(__config__, 'cc_ccsd_CCSD_async_io', True)
    incore_complete = getattr(__config__, 'cc_ccsd_CCSD_incore_complete', False)
    cc2 = getattr(__config__, 'cc_ccsd_CCSD_cc2', False)
    callback = None

    _keys = {
        'max_cycle', 'conv_tol', 'iterative_damping',
        'conv_tol_normt', 'diis', 'diis_space', 'diis_file',
        'diis_start_cycle', 'diis_start_energy_diff', 'direct',
        'async_io', 'incore_complete', 'cc2', 'callback',
        'mol', 'verbose', 'stdout', 'frozen', 'level_shift',
        'mo_coeff', 'mo_occ', 'cycles', 'converged_lambda', 'emp2', 'e_hf',
        'converged', 'e_corr', 't1', 't2', 'l1', 'l2', 'chkfile',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import hf
        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError(
                'CCSD Warning: The first argument mf is a DFT object. '
                'CCSD calculation should be initialized with HF object.\n'
                'DFT can be converted to HF object with the mf.to_hf() method\n')

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        self.level_shift = 0

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = False
        self.cycles = None
        self.converged_lambda = False
        self.emp2 = None
        self.e_hf = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self._nocc = None
        self._nmo = None
        self.chkfile = mf.chkfile

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('chkfile', 'callback'))

    @property
    def ecc(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf

    def set_frozen(self, method='auto', window=(-1000.0, 1000.0)):
        from pyscf import cc
        is_gcc = isinstance(self, cc.gccsd.GCCSD)
        return set_frozen(self, method=method, window=window, is_gcc=is_gcc)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('CC2 = %g', self.cc2)
        log.info('CCSD nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def get_init_guess(self, eris=None):
        return self.init_amps(eris)[1:]
    def init_amps(self, eris=None):
        time0 = logger.process_clock(), logger.perf_counter()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        e_hf = self.e_hf
        if e_hf is None: e_hf = self.get_e_hf(mo_coeff=self.mo_coeff)
        mo_e = eris.mo_energy
        nocc = self.nocc
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]

        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
        emp2 = 0
        for p0, p1 in lib.prange(0, nvir, blksize):
            eris_ovov = eris.ovov[:,p0:p1]
            t2[:,:,p0:p1] = (eris_ovov.transpose(0,2,1,3).conj()
                             / lib.direct_sum('ia,jb->ijab', eia[:,p0:p1], eia))
            emp2 += 2 * numpy.einsum('ijab,iajb', t2[:,:,p0:p1], eris_ovov)
            emp2 -=     numpy.einsum('jiab,iajb', t2[:,:,p0:p1], eris_ovov)
        self.emp2 = emp2.real

        logger.info(self, 'Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g',
                    e_hf + self.emp2, self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy
    _add_vvvv = _add_vvvv
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)
    def ccsd(self, t1=None, t2=None, eris=None):
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)
        logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    as_scanner = as_scanner
    restore_from_diis_ = restore_from_diis_

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        raise NotImplementedError

    def ccsd_t(self, t1=None, t2=None, eris=None):
        raise NotImplementedError

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomip_method(self):
        raise NotImplementedError

    def eomea_method(self):
        raise NotImplementedError

    def eomee_method(self):
        raise NotImplementedError

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_mf=True):
        '''Un-relaxed 1-particle density matrix in MO space'''
        raise NotImplementedError

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None):
        # Pseudo code how eris are implemented:
        # nocc = self.nocc
        # nmo = self.nmo
        # nvir = nmo - nocc
        # eris = _ChemistsERIs()
        # eri = ao2mo.incore.full(self._scf._eri, mo_coeff)
        # eri = ao2mo.restore(1, eri, nmo)
        # eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
        # eris.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
        # eris.ovvo = eri[nocc:,:nocc,nocc:,:nocc].copy()
        # eris.ovov = eri[nocc:,:nocc,:nocc,nocc:].copy()
        # eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
        # ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
        # eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir))
        # eris.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:], nvir)
        # eris.fock = numpy.diag(self._scf.mo_energy)
        # return eris

        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        if (adiis and
            istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes_to_vector(t1, t2)
            t1, t2 = self.vector_to_amplitudes(adiis.update(vec))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def vector_size(self, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        return nov + nov*(nov+1)//2

    def dump_chk(self, t1_t2=None, frozen=None, mo_coeff=None, mo_occ=None):
        if not self.chkfile:
            return self
        if t1_t2 is None: t1_t2 = self.t1, self.t2
        t1, t2 = t1_t2

        if frozen is None: frozen = self.frozen
        # "None" cannot be serialized by the chkfile module
        if frozen is None:
            frozen = 0

        cc_chk = {'e_corr': self.e_corr,
                  't1': t1,
                  't2': t2,
                  'frozen': frozen}

        if mo_coeff is not None: cc_chk['mo_coeff'] = mo_coeff
        if mo_occ is not None: cc_chk['mo_occ'] = mo_occ
        if self._nmo is not None: cc_chk['_nmo'] = self._nmo
        if self._nocc is not None: cc_chk['_nocc'] = self._nocc

        lib.chkfile.save(self.chkfile, 'ccsd', cc_chk)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    # to_gpu can be reused only when __init__ still takes mf
    def to_gpu(self):
        mf = self.base.to_gpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('pyscf', 'gpu4pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

class CCSD(CCSDBase):
    __doc__ = CCSDBase.__doc__

    def dump_flags(self, verbose=None):
        CCSDBase.dump_flags(self, verbose)
        if self.verbose >= logger.DEBUG1 and self.__class__ == CCSD:
            nocc = self.nocc
            nvir = self.nmo - self.nocc
            flops = _flops(nocc, nvir)
            logger.debug1(self, 'total FLOPs %s', flops)
        return self

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import ccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return ccsd_t.kernel(self, eris, t1, t2, self.verbose)

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEESinglet(self).kernel(nroots, koopmans, guess, eris)

    def eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEETriplet(self).kernel(nroots, koopmans, guess, eris)

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEESpinFlip(self).kernel(nroots, koopmans, guess, eris)

    def eomip_method(self):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMIP(self)

    def eomea_method(self):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEA(self)

    def eomee_method(self):
        from pyscf.cc import eom_rccsd
        return eom_rccsd.EOMEE(self)

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_mf=True):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                  with_frozen=with_frozen, with_mf=with_mf)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                  with_frozen=with_frozen, with_dm1=with_dm1)

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.cc import dfccsd
        mycc = dfccsd.RCCSD(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mycc.with_df = with_df
        if mycc.with_df.auxbasis != auxbasis:
            mycc.with_df = mycc.with_df.copy()
            mycc.with_df.auxbasis = auxbasis
        return mycc

    def nuc_grad_method(self):
        from pyscf.grad import ccsd
        return ccsd.Gradients(self)

    def get_t1_diagnostic(self, t1=None):
        if t1 is None: t1 = self.t1
        return get_t1_diagnostic(t1)

    def get_d1_diagnostic(self, t1=None):
        if t1 is None: t1 = self.t1
        return get_d1_diagnostic(t1)

    def get_d2_diagnostic(self, t2=None):
        if t2 is None: t2 = self.t2
        return get_d2_diagnostic(t2)

CC = RCCSD = CCSD


class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.oooo = None
        self.ovoo = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

# Note: Recomputed fock matrix and HF energy since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))

        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        # Note self.mo_energy can be different to fock.diagonal().
        # self.mo_energy is used in the initial guess function (to generate
        # MP2 amplitudes) and CCSD update_amps preconditioner.
        # fock.diagonal() should only be used to compute the expectation value
        # of Slater determinants.
        mo_e = self.mo_energy = self.fock.diagonal().real
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD.\n'
                            'CCSD may be difficult to converge. Increasing '
                            'CCSD Attribute level_shift may improve '
                            'convergence.', gap)
        except ValueError:  # gap.size == 0
            pass
        return self

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        ovw = numpy.asarray(self.ovvv[slices])
        nocc, nvir, nvir_pair = ovw.shape
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
        nvir1 = ovvv.shape[2]
        return ovvv.reshape(nocc,nvir,nvir1,nvir1)

    def _contract_vvvv_t2(self, mycc, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, numpy.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:  # AO-direct contraction
            vvvv = None
        else:
            vvvv = self.vvvv
        return _contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, verbose)

    def _contract_vvvv_oov(self, mycc, r2, out=None):
        raise NotImplementedError

    def _contract_vvvv_ovv(self, mycc, r2, out=None):
        raise NotImplementedError

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    #:eri1 = ao2mo.restore(1, eri1, nmo)
    #:eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    #:eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    #:eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    #:eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    #:eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    #:ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    #:eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir)).reshape(nocc,nvir,-1)
    #:eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

    if eri1.ndim == 4:
        eri1 = ao2mo.restore(4, eri1, nmo)

    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = numpy.empty((nocc,nocc,nocc,nocc))
    eris.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
    eris.ovvo = numpy.empty((nocc,nvir,nvir,nocc))
    eris.ovov = numpy.empty((nocc,nvir,nocc,nvir))
    eris.ovvv = numpy.empty((nocc,nvir,nvir_pair))
    eris.vvvv = numpy.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = numpy.empty((nmo,nmo,nmo))
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.oovv = oovv
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
        dij = i - nocc + 1
        lib.pack_tril(buf[nocc:i+1,nocc:,nocc:],
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    from pyscf.scf.hf import RHF
    assert isinstance(mycc._scf, RHF)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')

    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.oovv[p0:p1] = eri[:,:,nocc:,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.ovoo[:,p0:p1] = eri[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = eri[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = eri[:,:,:nocc,nocc:].transpose(1,0,2,3)
        vvv = lib.pack_tril(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.ovvv[:,p0:p1] = vvv.reshape(p1-p0,nocc,nvpair).transpose(1,0,2)

    cput1 = logger.process_clock(), logger.perf_counter()
    if not mycc.direct:
        max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])
        eris.feri2 = lib.H5TmpFile()
        ao2mo.full(mol, orbv, eris.feri2, max_memory=max_memory, verbose=log)
        eris.vvvv = eris.feri2['eri_mo']
        cput1 = log.timer_debug1('transforming vvvv', *cput1)

    fswap = lib.H5TmpFile()
    max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])
    int2e = mol._add_suffix('int2e')
    ao2mo.outcore.half_e1(mol, (mo_coeff,orbo), fswap, int2e,
                          's4', 1, max_memory, verbose=log)

    ao_loc = mol.ao_loc_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
    blksize = min(nmo, max(BLKMIN, blksize))
    log.debug1('blksize %d', blksize)
    cput2 = cput1

    fload = ao2mo.outcore._load_from_h5g
    buf = numpy.empty((blksize*nocc,nao_pair))
    buf_prefetch = numpy.empty_like(buf)
    def load(buf_prefetch, p0, rowmax):
        if p0 < rowmax:
            p1 = min(rowmax, p0+blksize)
            fload(fswap['0'], p0*nocc, p1*nocc, buf_prefetch)

    outbuf = numpy.empty((blksize*nocc,nmo**2))
    with lib.call_in_background(load, sync=not mycc.async_io) as prefetch:
        prefetch(buf_prefetch, 0, nocc)
        for p0, p1 in lib.prange(0, nocc, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, p1, nocc)

            nrow = (p1 - p0) * nocc
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_occ_frac(p0, p1, dat)
        cput2 = log.timer_debug1('transforming oopp', *cput2)

        prefetch(buf_prefetch, nocc, nmo)
        for p0, p1 in lib.prange(0, nvir, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, nocc+p1, nmo)

            nrow = (p1 - p0) * nocc
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_vir_frac(p0, p1, dat)
            cput2 = log.timer_debug1('transforming ovpp [%d:%d]'%(p0,p1), *cput2)

    cput1 = log.timer_debug1('transforming oppp', *cput1)
    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_df_eris_outcore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2

    naux = mycc._scf.with_df.get_naoaux()
    Loo = numpy.empty((naux,nocc,nocc))
    Lov = numpy.empty((naux,nocc,nvir))
    Lvo = numpy.empty((naux,nvir,nocc))
    Lvv = numpy.empty((naux,nvir_pair))
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0
    for eri1 in mycc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
        Lvv[p0:p1] = lib.pack_tril(Lpq[:,nocc:,nocc:].reshape(-1,nvir,nvir))
    Loo = Loo.reshape(naux,nocc*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    Lvo = Lvo.reshape(naux,nocc*nvir)

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv)).reshape(nocc,nocc,nvir,nvir)
    eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocc,nvir,nvir,nocc)
    eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovvv[:] = lib.ddot(Lov.T, Lvv).reshape(nocc,nvir,nvir_pair)
    eris.vvvv[:] = lib.ddot(Lvv.T, Lvv)
    log.timer('CCSD integral transformation', *cput0)
    return eris

def _flops(nocc, nvir):
    '''Total float points'''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nocc*(nocc+1)/2*nvir**4*2 +                 # vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 +
            nocc**3*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 4)      # Wiabj


if __name__ == '__main__':
    from pyscf import scf

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf() # -76.0267656731

    mf = rhf.density_fit(auxbasis='weigend')
    mf._eri = None
    mcc = CCSD(mf)
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    print(abs(t2).sum() - 4.9318753386922278)
    print(emp2 - -0.20401737899811551)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(abs(t1).sum() - 0.046961325647584914)
    print(abs(t2).sum() - 5.378260578551683   )

    mcc = CCSD(rhf)
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    print(abs(t2).sum() - 4.9556571218177)
    print(emp2 - -0.2040199672883385)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(abs(t1).sum()-0.0475038989126)
    print(abs(t2).sum()-5.401823846018721)
    print(energy(mcc, t1, t2, eris) - -0.208967840546667)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(energy(mcc, t1, t2, eris) - -0.212173678670510)
    print(abs(t1).sum() - 0.05470123093500083)
    print(abs(t2).sum() - 5.5605208391876539)

    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.max_memory = 1
    mcc.direct = True
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    e, v = mcc.ipccsd(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    e, v = mcc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
