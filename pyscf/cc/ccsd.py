#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
RCCSD for real integrals
8-fold permutation symmetry has been used
(ij|kl) = (ji|kl) = (kl|ij) = ...
'''

import time
import ctypes
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import _ccsd

BLKMIN = 4

# t1: ia
# t2: ijab
def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    log = logger.new_logger(mycc, verbose)

    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t1 is None:
        nocc, nvir = t2.shape[0], t2.shape[2]
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if mycc.diis:
            t1, t2 = mycc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(mycc, t1, t2, eris):
    if mycc.cc2:
        raise NotImplementedError
    assert(isinstance(eris, _ChemistsERIs))

    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir
    fock = eris.fock

    t1new = numpy.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris)
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc].copy()
    foo[numpy.diag_indices(nocc)] = 0
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:].copy()
    fvv[numpy.diag_indices(nvir)] = 0
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    unit = nocc**2*nvir*7 + nocc**3
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nocc, max(BLKMIN, int((max_memory*.9e6/8-nocc**4)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    fswap = lib.H5TmpFile()
    fwVOov, fwVooV = _add_vovv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap)
    time1 = log.timer_debug1('vovv', *time1)

    woooo = _cp(eris.oooo).transpose(0,2,1,3).copy()

    for p0, p1 in lib.prange(0, nvir, blksize):
        wVOov = _cp(fwVOov[p0:p1])
        wVooV = _cp(fwVooV[p0:p1])
        eris_vooo = _cp(eris.vooo[p0:p1])
        foo += numpy.einsum('kc,ckji->ij', 2*t1[:,p0:p1], eris_vooo)
        foo += numpy.einsum('kc,cijk->ij',  -t1[:,p0:p1], eris_vooo)
        tmp = lib.einsum('la,ajik->lkji', t1[:,p0:p1], eris_vooo)
        woooo += tmp + tmp.transpose(1,0,3,2)
        tmp = None

        wVOov -= lib.einsum('bjik,ka->bjia', eris_vooo, t1)
        t2new[:,:,p0:p1] += wVOov.transpose(1,2,0,3)

        wVooV += lib.einsum('bkij,ka->bija', eris_vooo, t1)
        eris_vooo = None

        eris_vvoo = _cp(eris.vvoo[p0:p1])
        t1new[:,p0:p1] -= numpy.einsum('jb,abji->ia', t1, eris_vvoo)
        wVooV -= eris_vvoo.transpose(0,2,3,1)

        eris_voov = _cp(eris.voov[p0:p1])
        t2new[:,:,p0:p1] += eris_voov.transpose(1,2,0,3) * .5
        t1new[:,p0:p1] += 2*numpy.einsum('jb,aijb->ia', t1, eris_voov)

        tmp  = lib.einsum('ic,cjbk->ijbk', t1, eris_vvoo.transpose(1,2,0,3))
        tmp += lib.einsum('bjkc,ic->jibk', eris_voov, t1)
        t2new[:,:,p0:p1] -= numpy.einsum('ka,jibk->jiba', t1, tmp)
        eris_vvoo = tmp = None

        fov[:,p0:p1] += numpy.einsum('kc,aikc->ia', t1, eris_voov) * 2
        fov[:,p0:p1] -= numpy.einsum('kc,akic->ia', t1, eris_voov)

        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
        theta = tau.transpose(1,0,2,3)*2 - tau
        fvv -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo += lib.einsum('aikb,kjab->ij', eris_voov, theta)
        tau = theta = None

        wVOov += wVooV*.5  #: bjia + bija*.5

        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo += lib.einsum('ijab,aklb->ijkl', tau, eris_voov)
        tau = None

        tau = t2*.5 + numpy.einsum('ia,jb->ijab', t1, t1)
        wVooV += lib.einsum('bkic,jkca->bija', eris_voov, tau)
        tau = None
        for q0, q1 in lib.prange(0, nvir, blksize):
            tmp = lib.einsum('jkca,ckib->jaib', t2[:,:,p0:p1,q0:q1], wVooV)
            t2new[:,:,:,q0:q1] += tmp.transpose(0,2,3,1)
            t2new[:,:,q0:q1,:] += tmp.transpose(0,2,1,3) * .5
            tmp = None

        wVOov += eris_voov
        eris_VOov = eris_voov - eris_voov.transpose(0,2,1,3)*.5
        eris_voov = None
        for q0, q1 in lib.prange(0, nvir, blksize):
            tau = t2[:,:,:,q0:q1].transpose(0,2,1,3) * 2 - t2[:,:,q0:q1].transpose(0,3,1,2)
            tau -= numpy.einsum('ia,jb->ibja', t1[:,q0:q1]*2, t1)
            wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau)
            tau = None
        for q0, q1 in lib.prange(0, nvir, blksize):
            theta = t2[:,:,p0:p1,q0:q1] * 2 - t2[:,:,p0:p1,q0:q1].transpose(1,0,2,3)
            t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
            theta = None
        eris_VOov = wVOov = wVooV = None
        time1 = log.timer_debug1('voov [%d:%d]'%(p0, p1), *time1)
    fwVOov = fwVooV = fswap = None

    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_vooo = _cp(eris.vooo[p0:p1])
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        t1new += numpy.einsum('jb,ijba->ia', fov[:,p0:p1], theta)
        t1new -= lib.einsum('bjki,kjba->ia', eris_vooo, theta)

        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        t2new[:,:,p0:p1] += .5 * lib.einsum('ijkl,klab->ijab', woooo, tau)
        theta = tau = None

    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    t2new += lib.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= lib.einsum('ki,kjab->ijab', ft_ij, t2)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
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


def _add_vovv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap):
    time1 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2*3 + nocc**2*nvir
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-t2.size)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)
    nvir_pair = nvir * (nvir+1) // 2
    def load_eri(eri, p0, p1, buf):
        if p0 < p1:
            buf[:p1-p0] = eri[p0:p1]

    wVOov = fswap.create_dataset('wVOov', (nvir,nocc,nocc,nvir), 'f8')
    fwooVV = numpy.zeros((nocc,nocc,nvir,nvir))

    buf = numpy.empty((blksize,nocc,nvir_pair))
    eris_vovv = None
    with lib.call_in_background(load_eri) as prefetch:
        load_eri(eris.vovv, 0, blksize, buf)
        for p0, p1 in lib.prange(0, nvir, blksize):
            eris_vovv, buf = buf[:p1-p0], numpy.empty_like(buf)
            prefetch(eris.vovv, p1, min(nvir, p1+blksize), buf)

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
                eris_vvvo = None

            #:fwooVV -= numpy.einsum('jc,ciba->jiba', t1[:,p0:p1], eris_vovv)
            lib.ddot(_cp(t1[:,p0:p1]), eris_vovv.reshape(p1-p0,-1), -1,
                     fwooVV.reshape(nocc,-1), 1)

            wVOov[p0:p1] += lib.einsum('biac,jc->bija', eris_vovv, t1)

            theta = t2[:,:,p0:p1].transpose(1,2,0,3) * 2
            theta -= t2[:,:,p0:p1].transpose(0,2,1,3)
            t1new += lib.einsum('icjb,cjba->ia', theta, eris_vovv)
            theta = None
            time1 = log.timer_debug1('vovv [%d:%d]'%(p0, p1), *time1)

    fswap['wVooV'] = fwooVV.transpose(2,1,0,3)
    return fswap['wVOov'], fswap['wVooV']

def _add_vvvv_tril(mycc, t1, t2, eris, out=None, with_ovvv=True):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    Using symmetry t2[ijab] = t2[jiba] and Ht2[ijab] = Ht2[jiba], compute the
    lower triangular part of  Ht2
    '''
    time0 = time.clock(), time.time()
    _dgemm = lib.numpy_helper._dgemm
    nocc, nvir = t1.shape
    t2new_tril = numpy.ndarray((nocc*(nocc+1)//2,nvir,nvir), buffer=out)
    t2new_tril[:] = 0

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
    def contract_rec_(t2new_tril, tau, eri, i0, i1, j0, j1):
        nao = tau.shape[-1]
        ic = i1 - i0
        jc = j1 - j0
        #: t2tril[:,j0:j1] += numpy.einsum('xcd,cdab->xab', tau[:,i0:i1], eri)
        _dgemm('N', 'N', nocc*(nocc+1)//2, jc*nao, ic*nao,
               tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
               t2new_tril.reshape(-1,nao*nao), 1, 1, i0*nao, 0, j0*nao)

        #: t2tril[:,i0:i1] += numpy.einsum('xcd,abcd->xab', tau[:,j0:j1], eri)
        _dgemm('N', 'T', nocc*(nocc+1)//2, ic*nao, jc*nao,
               tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
               t2new_tril.reshape(-1,nao*nao), 1, 1, j0*nao, 0, i0*nao)

    def contract_tril_(t2new_tril, tau, eri, a0, a):
        nvir = tau.shape[-1]
        #: t2new[i,:i+1, a] += numpy.einsum('xcd,cdb->xb', tau[:,a0:a+1], eri)
        _dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (a+1-a0)*nvir,
               tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
               t2new_tril.reshape(-1,nvir*nvir), 1, 1, a0*nvir, 0, a*nvir)

        #: t2new[i,:i+1,a0:a] += numpy.einsum('xd,abd->xab', tau[:,a], eri[:a])
        if a > a0:
            _dgemm('N', 'T', nocc*(nocc+1)//2, (a-a0)*nvir, nvir,
                   tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
                   t2new_tril.reshape(-1,nvir*nvir), 1, 1, a*nvir, 0, a0*nvir)

    if mycc.direct:   # AO-direct CCSD
        mol = mycc.mol
        if hasattr(eris, 'mo_coeff'):
            mo = eris.mo_coeff
        else:
            mo = _mo_without_core(mycc, mycc.mo_coeff)
        nao, nmo = mo.shape
        aos = numpy.asarray(mo[:,nocc:].T, order='F')
        nocc2 = nocc*(nocc+1)//2
        outbuf = numpy.empty((nocc2,nao,nao))
        tau = numpy.ndarray((nocc2,nvir,nvir), buffer=outbuf)
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]
        tau = _ao2mo.nr_e2(tau.reshape(nocc2,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
        tau = tau.reshape(nocc2,nao,nao)
        time0 = logger.timer_debug1(mycc, 'vvvv-tau', *time0)

        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        outbuf[:] = 0
        ao_loc = mol.ao_loc_nr()
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        dmax = max(BLKMIN, numpy.sqrt(max_memory*.95e6/8/nao**2/2))
        sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
        dmax = max(x[2] for x in sh_ranges)
        eribuf = numpy.empty((dmax,dmax,nao,nao))
        loadbuf = numpy.empty((dmax,dmax,nao,nao))
        fint = gto.moleintor.getints4c

        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip]:
                eri = fint(intor, mol._atm, mol._bas, mol._env,
                           shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                           ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                tmp = numpy.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
                contract_rec_(outbuf, tau, tmp, i0, i1, j0, j1)
                time0 = logger.timer_debug1(mycc, 'AO-vvvv [%d:%d,%d:%d]' %
                                            (ish0,ish1,jsh0,jsh1), *time0)
            eri = fint(intor, mol._atm, mol._bas, mol._env,
                       shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                       ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            p1 = 0
            for i in range(i1-i0):
                p0, p1 = p1, p1 + i+1
                tmp = lib.unpack_tril(eri[p0:p1], out=loadbuf)
                contract_tril_(outbuf, tau, tmp, i0, i0+i)
            time0 = logger.timer_debug1(mycc, 'AO-vvvv [%d:%d,%d:%d]' %
                                        (ish0,ish1,ish0,ish1), *time0)
        eribuf = loadbuf = eri = tmp = None

        _ao2mo.nr_e2(outbuf, mo, (nocc,nmo,nocc,nmo), 's1', 's1', out=t2new_tril)

        if with_ovvv:
            #: tmp = numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
            #: t2new -= tmp + tmp.transpose(1,0,3,2)
            tmp = _ao2mo.nr_e2(outbuf, mo, (nocc,nmo,0,nocc), 's1', 's1', out=tau)
            t2new_tril -= lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1).reshape(nocc2,nvir,nvir)
            tmp = _ao2mo.nr_e2(outbuf, mo, (0,nocc,nocc,nmo), 's1', 's1', out=tau)
            #: t2new_tril -= numpy.einsum('xkb,ka->xab', tmp.reshape(-1,nocc,nvir), t1)
            tmp = lib.transpose(tmp.reshape(nocc2,nocc,nvir), axes=(0,2,1), out=outbuf)
            tmp = lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1, 1,
                           numpy.ndarray((nocc2*nvir,nvir), buffer=tau), 0)
            tmp = lib.transpose(tmp.reshape(nocc2,nvir,nvir), axes=(0,2,1), out=outbuf)
            t2new_tril -= tmp.reshape(nocc2,nvir,nvir)

    else:
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]
        time0 = logger.timer_debug1(mycc, 'vvvv-tau', *time0)
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        blksize = int(min(nvir, max(BLKMIN, max_memory*.95e6/8/(nvir**3*2))))

        def block_contract(buf, a0, a1):
            for a in range(a0, a1):
                contract_tril_(t2new_tril, tau, buf[a-a0], 0, a)

        with lib.call_in_background(block_contract) as bcontract:
            p1 = 0
            outbuf = numpy.empty((blksize,nvir,nvir,nvir))
            outbuf1 = numpy.empty_like(outbuf)
            for a0, a1 in lib.prange(0, nvir, blksize):
                for a in range(a0, a1):
                    p0, p1 = p1, p1 + a+1
                    lib.unpack_tril(eris.vvvv[p0:p1], out=outbuf[a-a0])
                bcontract(outbuf, a0, a1)
                outbuf, outbuf1 = outbuf1, outbuf
                time0 = logger.timer_debug1(mycc, 'vvvv [%d:%d]'%(a0,a1), *time0)
    return t2new_tril

def _add_vvvv_full(mycc, t1, t2, eris, out=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    without using symmetry in t2 or Ht2
    '''
    time0 = time.clock(), time.time()
    nocc, nvir = t1.shape
    tau = numpy.einsum('ia,jb->ijab', t1, t1)
    tau += t2

    if mycc.direct:  # AO direct CCSD
        mol = mycc.mol
        if hasattr(eris, 'mo_coeff'):
            mo = eris.mo_coeff
        else:
            mo = _mo_without_core(mycc, mycc.mo_coeff)
        nao, nmo = mo.shape
        aos = numpy.asarray(mo[:,nocc:].T, order='F')
        tau = _ao2mo.nr_e2(tau.reshape(nocc**2,nvir,nvir), aos, (0,nao,0,nao), 's1', 's1')
        tau = tau.reshape(nocc,nocc,nao,nao)
        time0 = logger.timer_debug1(mycc, 'vvvv-tau mo2ao', *time0)

        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        Ht2 = numpy.zeros((nocc,nocc,nao,nao))
        ao_loc = mol.ao_loc_nr()
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        dmax = max(BLKMIN, numpy.sqrt(max_memory*.9e6/8/nao**2/2))
        sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
        dmax = max(x[2] for x in sh_ranges)
        eribuf = numpy.empty((dmax,dmax,nao,nao))
        loadbuf = numpy.empty((dmax,dmax,nao,nao))
        fint = gto.moleintor.getints4c

        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                eri = fint(intor, mol._atm, mol._bas, mol._env,
                           shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                           ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                tmp = numpy.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
                Ht2[:,:,j0:j1] += lib.einsum('ijef,efab->ijab', tau[:,:,i0:i1], tmp)
                if ish0 > jsh0:
                    Ht2[:,:,i0:i1] += lib.einsum('ijef,abef->ijab', tau[:,:,j0:j1], tmp)
                time0 = logger.timer_debug1(mycc, 'AO-vvvv [%d:%d,%d:%d]' %
                                            (ish0,ish1,jsh0,jsh1), *time0)
        eribuf = loadbuf = eri = tmp = None
        Ht2 = _ao2mo.nr_e2(Ht2.reshape(nocc**2,-1), mo, (nocc,nmo,nocc,nmo), 's1', 's1')
    else:
        Ht2 = numpy.zeros_like(t2)
        for i in range(nvir):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(numpy.asarray(eris.vvvv[i0:i0+i+1]))
            Ht2[:,:, i] += lib.einsum('ijef,ebf->ijb', tau[:,:,:i+1], vvv)
            if i > 0:
                Ht2[:,:,:i] += lib.einsum('ijf,abf->ijab', tau[:,:,i], vvv[:i])
    return Ht2.reshape(nocc,nocc,nvir,nvir)

def _unpack_t2_tril(t2tril, nocc, nvir, out=None):
    t2 = numpy.ndarray((nocc,nocc,nvir,nvir), buffer=out)
    idx,idy = numpy.tril_indices(nocc)
    t2[idx,idy] = t2tril
    t2[idy,idx] = t2tril.transpose(0,2,1)
    return t2

def _unpack_4fold(c2vec, nocc, nvir, anti_symm=True):
    t2 = numpy.zeros((nocc**2,nvir**2), dtype=c2vec.dtype)
    if nocc > 1 and nvir > 1:
        t2tril = c2vec.reshape(nocc*(nocc-1)//2,nvir*(nvir-1)//2)
        otril = numpy.tril_indices(nocc, k=-1)
        vtril = numpy.tril_indices(nvir, k=-1)
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[0]*nvir+vtril[1])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[1]*nvir+vtril[0])
        if anti_symm:  # anti-symmetry when exchanging two particles
            t2tril = -t2tril
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[1]*nvir+vtril[0])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[0]*nvir+vtril[1])
    return t2.reshape(nocc,nocc,nvir,nvir)

def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=True):
    t2new_tril = mycc._add_vvvv_tril(t1, t2, eris, None, with_ovvv)
    nocc, nvir = t1.shape
    t2new = _unpack_t2_tril(t2new_tril, nocc, nvir, out)
    return t2new


def get_nocc(mycc):
    if mycc._nocc is not None:
        return mycc._nocc
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        nocc = numpy.count_nonzero(mycc.mo_occ > 0) - mycc.frozen
        assert(nocc > 0)
        return nocc
    else:
        occ_idx = mycc.mo_occ > 0
        occ_idx[list(mycc.frozen)] = False
        return numpy.count_nonzero(occ_idx)

def get_nmo(mycc):
    if mycc._nmo is not None:
        return mycc._nmo
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        return len(mycc.mo_occ) - mycc.frozen
    else:
        return len(mycc.mo_occ) - len(mycc.frozen)

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
    t2 = lib.unpack_tril(vector[nov:])
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


def energy(mycc, t1, t2, eris):
    '''CCSD correlation energy'''
    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir))))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_voov = _cp(eris.voov[p0:p1])
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        e += 2 * numpy.einsum('ijab,aijb', tau, eris_voov)
        e -=     numpy.einsum('jiab,aijb', tau, eris_voov)
    return e


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
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    logger.info(cc, 'Set %s as a scanner', cc.__class__)
    class CCSD_Scanner(cc.__class__, lib.SinglePointScanner):
        def __init__(self, cc):
            self.__dict__.update(cc.__dict__)
            self._scf = cc._scf.as_scanner()
        def __call__(self, mol):
            mf_scanner = self._scf
            mf_scanner(mol)
            self.mol = mol
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
            self.kernel(self.t1, self.t2)[0]
            return self.e_tot
    return CCSD_Scanner(cc)


class CCSD(lib.StreamObject):
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
        direct : bool
            AO-direct CCSD. Default is False.
        frozen : int or list
            If integer is given, the inner-most orbitals are frozen from CC
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in CC calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> mycc = cc.CCSD(mf).set(frozen = 2).run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> mycc.set(frozen = [0,1,16,17,18]).run()

    Saved results

        converged : bool
            CCSD converged or not
        e_corr : float
            CCSD correlation correction
        e_tot : float
            Total CCSD energy (HF + correlation)
        t1, t2 : 
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
        l1, l2 : 
            Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf import gto
        if isinstance(mf, gto.Mole):
            raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.10.
In the new API, the first argument of CC class is HF objects.  Please see
http://sunqm.net/pyscf/code-rule.html#api-rules for the details of API conventions''')
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_cycle = 50
        self.conv_tol = 1e-7
        self.conv_tol_normt = 1e-5
        self.diis_space = 6
        self.diis_file = None
        self.diis_start_cycle = 0
# FIXME: Should we avoid DIIS starting early?
        self.diis_start_energy_diff = 1e9
        self.direct = False
        self.cc2 = False

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = False
        self.converged_lambda = False
        self.emp2 = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self._nocc = None
        self._nmo = None
        self.chkfile = None

        self._keys = set(self.__dict__.keys())

    @property
    def ecc(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

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

    get_nocc = get_nocc
    get_nmo = get_nmo

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('CC2 = %g', self.cc2)
        log.info('CCSD nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
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
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return self.init_amps(eris)[1:]
    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir))))
        self.emp2 = 0
        for p0, p1 in lib.prange(0, nvir, blksize):
            eris_voov = _cp(eris.voov[p0:p1])
            t2[:,:,p0:p1] = (eris_voov.transpose(1,2,0,3)
                             / lib.direct_sum('ia,jb->ijab', eia[:,p0:p1], eia))
            self.emp2 += 2 * numpy.einsum('ijab,aijb', t2[:,:,p0:p1], eris_voov)
            self.emp2 -=     numpy.einsum('jiab,aijb', t2[:,:,p0:p1], eris_voov)

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy
    _add_vvvv = _add_vvvv
    _add_vvvv_tril = _add_vvvv_tril
    _add_vvvv_full = _add_vvvv_full
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)
    def ccsd(self, t1=None, t2=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CCSD converged')
        else:
            logger.note(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    as_scanner = as_scanner


    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
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

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
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
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        # Pseudo code how eris are implemented:
        # nocc = self.nocc
        # nmo = self.nmo
        # nvir = nmo - nocc
        # eris = lambda:None
        # eri1 = ao2mo.incore.full(self._scf._eri, mo_coeff)
        # eri1 = ao2mo.restore(1, eri1, nmo)
        # eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        # eris.vooo = eri1[nocc:,:nocc,:nocc,:nocc].copy()
        # eris.voov = eri1[nocc:,:nocc,:nocc,nocc:].copy()
        # eris.vvoo = eri1[nocc:,nocc:,:nocc,:nocc].copy()
        # vovv = eri1[nocc:,:nocc,nocc:,nocc:].copy()
        # eris.vovv = lib.pack_tril(vovv.reshape(-1,nvir,nvir))
        # eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
        # eris.fock = numpy.diag(self._scf.mo_energy)
        # return eris

        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)

    def diis(self, t1, t2, istep, normt, de, adiis):
        return self.diis_(t1, t2, istep, normt, de, adiis)
    def diis_(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
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

    def dump_chk(self, t1_t2=None, frozen=None, mo_coeff=None, mo_occ=None):
        if t1_t2 is None:
            t1, t2 = self.t1, self.t2
        else:
            t1, t2 = t1_t2
        if frozen is None: frozen = self.frozen
        cc_chk = {'e_corr': self.e_corr,
                  't1': t1,
                  't2': t2,
                  'frozen': frozen}

        if mo_coeff is not None: cc_chk['mo_coeff'] = mo_coeff
        if mo_occ is not None: cc_chk['mo_occ'] = mo_occ
        if self._nmo is not None: cc_chk['_nmo'] = self._nmo
        if self._nocc is not None: cc_chk['_nocc'] = self._nocc

        if self.chkfile is not None:
            chkfile = self.chkfile
        else:
            chkfile = self._scf.chkfile
        lib.chkfile.save(chkfile, 'ccsd', cc_chk)

CC = CCSD


class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self):
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.oooo = None
        self.vooo = None
        self.vvoo = None
        self.voov = None
        self.vovo = None
        self.vovv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)
# Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))
        self.nocc = mycc.nocc
        return self

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    #:eri1 = ao2mo.restore(1, eri1, nmo)
    #:eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    #:eris.vooo = eri1[nocc:,:nocc,:nocc,:nocc].copy()
    #:eris.voov = eri1[nocc:,:nocc,:nocc,nocc:].copy()
    #:eris.vvoo = eri1[nocc:,nocc:,:nocc,:nocc].copy()
    #:vovv = eri1[nocc:,:nocc,nocc:,nocc:].copy()
    #:eris.vovv = lib.pack_tril(vovv.reshape(-1,nvir,nvir))
    #:eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = numpy.empty((nocc,nocc,nocc,nocc))
    eris.vooo = numpy.empty((nvir,nocc,nocc,nocc))
    eris.voov = numpy.empty((nvir,nocc,nocc,nvir))
    eris.vovv = numpy.empty((nvir,nocc,nvir_pair))
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
    eris.vvoo = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.vooo[i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.voov[i-nocc] = buf[:nocc,:nocc,nocc:]
        lib.pack_tril(_cp(buf[:nocc,nocc:,nocc:]), out=eris.vovv[i-nocc])
        dij = i - nocc + 1
        lib.pack_tril(_cp(buf[nocc:i+1,nocc:,nocc:]),
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.vvoo = eris.feri1.create_dataset('vvoo', (nvir,nvir,nocc,nocc), 'f8')
    eris.vooo = eris.feri1.create_dataset('vooo', (nvir,nocc,nocc,nocc), 'f8')
    eris.voov = eris.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), 'f8')
    chunks = (nvir, min(8,nocc), nvpair)
    eris.vovv = eris.feri1.create_dataset('vovv', (nvir,nocc,nvpair), 'f8', chunks=chunks)

    nvir_pair = nvir*(nvir+1)//2
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        oovv[p0:p1] = eri[:,:,nocc:,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.vooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.voov[p0:p1] = eri[:,:,:nocc,nocc:]
        vv = _cp(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.vovv[p0:p1] = lib.pack_tril(vv).reshape(p1-p0,nocc,nvir_pair)

    cput1 = time.clock(), time.time()
    if not mycc.direct:
        max_memory = max(2000, mycc.max_memory-lib.current_memory()[0])
        eris.feri2 = lib.H5TmpFile()
        ao2mo.full(mol, orbv, eris.feri2, max_memory=max_memory, verbose=log)
        eris.vvvv = eris.feri2['eri_mo']
        cput1 = log.timer_debug1('transforming vvvv', *cput1)

    fswap = lib.H5TmpFile()
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    max_memory = max(2000, mycc.max_memory-lib.current_memory()[0])
    int2e = mol._add_suffix('int2e')
    ao2mo.outcore.half_e1(mol, (mo_coeff,mo_coeff[:,:nocc]), fswap, int2e,
                          's4', 1, max_memory, verbose=log)

    ao_loc = mol.ao_loc_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
    blksize = max(1, min(nmo*nocc, blksize))
    fload = ao2mo.outcore._load_from_h5g
    def prefetch(p0, p1, rowmax, buf):
        p0, p1 = p1, min(rowmax, p1+blksize)
        if p0 < p1:
            fload(fswap['0'], p0*nocc, p1*nocc, buf)

    buf = numpy.empty((blksize*nocc,nao_pair))
    buf_prefetch = numpy.empty_like(buf)
    outbuf = numpy.empty((blksize*nocc,nmo**2))
    with lib.call_in_background(prefetch) as bprefetch:
        fload(fswap['0'], 0, min(nocc,blksize)*nocc, buf_prefetch)
        for p0, p1 in lib.prange(0, nocc, blksize):
            nrow = (p1 - p0) * nocc
            buf, buf_prefetch = buf_prefetch, buf
            bprefetch(p0, p1, nocc, buf_prefetch)
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_occ_frac(p0, p1, dat)

        fload(fswap['0'], nocc**2, min(nmo,nocc+blksize)*nocc, buf_prefetch)
        for p0, p1 in lib.prange(0, nvir, blksize):
            nrow = (p1 - p0) * nocc
            buf, buf_prefetch = buf_prefetch, buf
            bprefetch(nocc+p0, nocc+p1, nmo, buf_prefetch)
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_vir_frac(p0, p1, dat)

    cput1 = log.timer_debug1('transforming oppp', *cput1)
    eris.vvoo[:] = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_df_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    oooo = numpy.zeros((nocc*nocc,nocc*nocc))
    vooo = numpy.zeros((nvir*nocc,nocc*nocc))
    vvoo = numpy.zeros((nvir*nvir,nocc*nocc))
    voov = numpy.zeros((nvir*nocc,nocc*nvir))
    vovv = numpy.zeros((nvir*nocc,nvpair))
    vvvv = numpy.zeros((nvir_pair,nvpair))

    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    for eri1 in mycc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        Loo = Lpq[:,:nocc,:nocc].reshape(-1,nocc**2)
        Lov = Lpq[:,:nocc,nocc:].reshape(-1,nocc*nvir)
        Lvo = Lpq[:,nocc:,:nocc].reshape(-1,nocc*nvir)
        Lvv = Lpq[:,nocc:,nocc:].reshape(-1,nvir**2)
        lib.ddot(Loo.T, Loo, 1, oooo, 1)
        lib.ddot(Lvo.T, Loo, 1, vooo, 1)
        lib.ddot(Lvv.T, Loo, 1, vvoo, 1)
        lib.ddot(Lvo.T, Lov, 1, voov, 1)
        Lvv = lib.pack_tril(Lvv.reshape(-1,nvir,nvir))
        lib.ddot(Lvo.T, Lvv, 1, vovv, 1)
        lib.ddot(Lvv.T, Lvv, 1, vvvv, 1)

    eris.feri1 = lib.H5TmpFile()
    eris.feri1['oooo'] = oooo.reshape(nocc,nocc,nocc,nocc)
    eris.feri1['vooo'] = vooo.reshape(nvir,nocc,nocc,nocc)
    eris.feri1['vvoo'] = vvoo.reshape(nvir,nvir,nocc,nocc)
    eris.feri1['voov'] = voov.reshape(nvir,nocc,nocc,nvir)
    eris.feri1['vovv'] = vovv.reshape(nvir,nocc,nvpair)
    eris.feri1['vvvv'] = vvvv.reshape(nvir_pair,nvpair)
    eris.oooo = eris.feri1['oooo']
    eris.vooo = eris.feri1['vooo']
    eris.vvoo = eris.feri1['vvoo']
    eris.voov = eris.feri1['voov']
    eris.vovv = eris.feri1['vovv']
    eris.vvvv = eris.feri1['vvvv']
    log.timer('CCSD integral transformation', *cput0)
    return eris

def get_moidx(cc):
    moidx = numpy.ones(cc.mo_occ.size, dtype=numpy.bool)
    if isinstance(cc.frozen, (int, numpy.integer)):
        moidx[:cc.frozen] = False
    elif len(cc.frozen) > 0:
        moidx[list(cc.frozen)] = False
    return moidx
def _mo_without_core(cc, mo):
    return mo[:,get_moidx(cc)]

def _fp(nocc, nvir):
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

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    numpy.random.seed(12)
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = numpy.random.random((nmo,nmo))
    mf.mo_energy = numpy.arange(0., nmo)
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = numpy.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(numpy.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = CCSD(mf)
    eris = mycc.ao2mo()
    a = numpy.random.random((nmo,nmo)) * .1
    eris.fock += a + a.T
    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    r1, r2 = vector_to_amplitudes(amplitudes_to_vector(t1, t2), nocc+nvir, nocc)
    print(abs(t1-r1).max())
    print(abs(t2-r2).max())

    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - 66540.100267798145)

    mycc.max_memory = 0
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - 66540.100267798145)

    t2tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
    t2tril = _add_vvvv_tril(mycc, t1, t2, eris, t2tril)
    print(lib.finger(t2tril) - 13306.139402693696)

    t2tril = _add_vvvv_tril(mycc, t1, t2, eris)
    print(lib.finger(t2tril) - 13306.139402693696)
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (1. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = {'H': 'sto3g',
                 'O': 'cc-pvdz',}
    mol.charge = 1
    mol.build(0, 0)
    mycc.mol = mol
    mycc.direct = True
    t2tril = _add_vvvv_tril(mycc, t1, t2, eris, with_ovvv=True)
    print(lib.finger(t2tril) - 680.07199094501584)
    t2tril = _add_vvvv_tril(mycc, t1, t2, eris, with_ovvv=False)
    print(lib.finger(t2tril) - 446.56702664171348)

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
