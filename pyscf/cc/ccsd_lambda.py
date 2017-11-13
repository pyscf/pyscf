#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd

einsum = lib.einsum
# t2,l2 as ijab

def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    nocc, nvir = t1.shape
    saved = make_intermediates(mycc, t1, t2, eris)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, saved)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir*(nvir+1)//2
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    class _Saved:
        pass
    saved = _Saved()
    saved.ftmp = lib.H5TmpFile()
    saved.woooo = saved.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    saved.wvooo = saved.ftmp.create_dataset('wvooo', (nvir,nocc,nocc,nocc), 'f8')
    saved.wVOov = saved.ftmp.create_dataset('wVOov', (nvir,nocc,nocc,nvir), 'f8')
    saved.wvOOv = saved.ftmp.create_dataset('wvOOv', (nvir,nocc,nocc,nvir), 'f8')
    saved.wvvov = saved.ftmp.create_dataset('wvvov', (nvir,nvir,nocc,nvir), 'f8')

# As we don't have l2 in memory, hold tau temporarily in memory
    w1 = fvv - numpy.einsum('ja,jb->ba', fov, t1)
    w2 = foo + numpy.einsum('ib,jb->ij', fov, t1)
    w3 = numpy.einsum('kc,jkbc->bj', fov, t2) * 2 + fov.T
    w3 -= numpy.einsum('kc,kjbc->bj', fov, t2)
    w3 += reduce(numpy.dot, (t1.T, fov, t1.T))
    w4 = fov.copy()

    time1 = time.clock(), time.time()
    unit = nocc*nvir**2*6
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(ccsd.BLKMIN, int((max_memory*.95e6/8-nocc**4-nvir*nocc**3)/unit)))
    log.debug1('ccsd lambda make_intermediates: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)//blksize))

    fswap = lib.H5TmpFile()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        eris_vovv = _cp(eris.vovv[p0:p1])
        eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,nvir_pair))
        eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)
        fswap['vvov/%d'%istep] = eris_vovv.transpose(2,3,1,0)

    woooo = 0
    wvooo = numpy.zeros((nvir,nocc,nocc,nocc))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_vovv = _cp(eris.vovv[p0:p1])
        eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,nvir_pair))
        eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)
        eris_vvov = numpy.empty(((p1-p0),nvir,nocc,nvir))
        for istep, (q0, q1) in enumerate(lib.prange(0, nvir, blksize)):
            eris_vvov[:,:,:,q0:q1] = fswap['vvov/%d'%istep][p0:p1]

        w1 += numpy.einsum('cjba,jc->ba', eris_vovv, t1[:,p0:p1]*2)
        w1[:,p0:p1] -= numpy.einsum('ajbc,jc->ba', eris_vovv, t1)
        theta = t2[:,:,:,p0:p1] * 2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        w3 += einsum('jkcd,dkcb->bj', theta, eris_vovv)
        theta = None
        wVOov = einsum('bjcd,kd->bjkc', eris_vovv, t1)
        wvOOv = einsum('cbjd,kd->cjkb', eris_vvov,-t1)
        g2vovv = eris_vvov.transpose(0,2,1,3) * 2 - eris_vvov.transpose(0,2,3,1)
        for i0, i1 in lib.prange(0, nocc, blksize):
            tau = t2[:,i0:i1] + numpy.einsum('ia,jb->ijab', t1, t1[i0:i1])
            wvooo[p0:p1,i0:i1] += einsum('cibd,jkbd->ckij', g2vovv, tau)
        g2vovv = tau = None

        # Watch out memory usage here, due to the t2 transpose
        wvvov  = einsum('ajbd,jkcd->abkc', eris_vovv, t2) * -1.5
        wvvov += eris_vvov.transpose(0,3,2,1) * 2
        wvvov -= eris_vvov

        g2vvov = eris_vvov * 2 - eris_vovv.transpose(0,2,1,3)
        for i0, i1 in lib.prange(0, nocc, blksize):
            theta = t2[i0:i1] * 2 - t2[i0:i1].transpose(0,1,3,2)
            vackb = einsum('acjd,kjbd->ackb', g2vvov, theta)
            wvvov[:,:,i0:i1] += vackb.transpose(0,3,2,1)
            wvvov[:,:,i0:i1] -= vackb * .5
        g2vvov = eris_vovv = eris_vvov = theta = None

        eris_vooo = _cp(eris.vooo[p0:p1])
        w2 += numpy.einsum('bkij,kb->ij', eris_vooo, t1[:,p0:p1]) * 2
        w2 -= numpy.einsum('bikj,kb->ij', eris_vooo, t1[:,p0:p1])
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        w3 -= einsum('clkj,klcb->bj', eris_vooo, theta)

        tmp = einsum('lc,cjik->ijkl', t1[:,p0:p1], eris_vooo)
        woooo += tmp
        woooo += tmp.transpose(1,0,3,2)
        theta = tmp = None

        wvOOv += einsum('bljk,lc->bjkc', eris_vooo, t1)
        wVOov -= einsum('bjkl,lc->bjkc', eris_vooo, t1)
        wvooo[p0:p1] += eris_vooo.transpose(0,3,2,1) * 2
        wvooo[p0:p1] -= eris_vooo
        wvooo -= einsum('klbc,bilj->ckij', t2[:,:,p0:p1], eris_vooo*1.5)

        g2vooo = eris_vooo * 2 - eris_vooo.transpose(0,3,2,1)
        theta = t2[:,:,:,p0:p1]*2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        vcjik = einsum('jlcb,blki->cjik', theta, g2vooo)
        wvooo += vcjik.transpose(0,3,2,1)
        wvooo -= vcjik*.5
        theta = g2vooo = None

        eris_voov = _cp(eris.voov[p0:p1])
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo += einsum('cijd,klcd->ijkl', eris_voov, tau)
        tau = None

        g2voov = eris_voov*2 - eris_voov.transpose(0,2,1,3)
        tmpw4 = numpy.einsum('ckld,ld->kc', g2voov, t1)
        w1 -= einsum('ckja,kjcb->ba', g2voov, t2[:,:,p0:p1])
        w1[:,p0:p1] -= numpy.einsum('ja,jb->ba', tmpw4, t1)
        w2 += einsum('jkbc,bikc->ij', t2[:,:,p0:p1], g2voov)
        w2 += numpy.einsum('ib,jb->ij', tmpw4, t1[:,p0:p1])
        w3 += reduce(numpy.dot, (t1.T, tmpw4, t1[:,p0:p1].T))
        w4[:,p0:p1] += tmpw4

        wvOOv += einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
        wVOov -= einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)

        VOov  = einsum('bjld,klcd->bjkc', g2voov, t2)
        VOov -= einsum('bjld,kldc->bjkc', eris_voov, t2)
        VOov += eris_voov
        vOOv = einsum('bljd,kldc->bjkc', eris_voov, t2)
        vOOv -= _cp(eris.vvoo[p0:p1]).transpose(0,3,2,1)
        wVOov += VOov
        wvOOv += vOOv
        saved.wVOov[p0:p1] = wVOov
        saved.wvOOv[p0:p1] = wvOOv
        wOVov = wOvOv = None

        ov1 = vOOv*2 + VOov
        ov2 = VOov*2 + vOOv
        vOOv = VOov = None
        wvooo -= einsum('jb,bikc->ckij', t1[:,p0:p1], ov1)
        wvooo += einsum('kb,bijc->ckij', t1[:,p0:p1], ov2)
        w3 += numpy.einsum('ckjb,kc->bj', ov2, t1[:,p0:p1])

        wvvov += einsum('ajkc,jb->abkc', ov1, t1)
        wvvov -= einsum('ajkb,jc->abkc', ov2, t1)

        eris_vooo = _cp(eris.vooo[p0:p1])
        g2vooo = eris_vooo * 2 - eris_vooo.transpose(0,2,1,3)
        tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        wvvov += einsum('alki,klbc->abic', g2vooo, tau)
        saved.wvvov[p0:p1] = wvvov
        wvvov = ov1 = ov2 = None

    woooo += _cp(eris.oooo).transpose(0,2,1,3)
    saved.woooo[:] = woooo
    saved.wvooo[:] = wvooo
    woooo = wvooo = None

    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)

    fswap = None

    saved.w1 = w1
    saved.w2 = w2
    saved.w3 = w3
    saved.w4 = w4
    saved.ftmp.flush()
    return saved


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris=None, saved=None):
    if saved is None: saved = make_intermediates(mycc, t1, t2, eris)
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    fov = eris.fock[:nocc,nocc:]

    theta = t2*2 - t2.transpose(0,1,3,2)
    mba = einsum('klca,klcb->ba', l2, theta)
    mij = einsum('ikcd,jkcd->ij', l2, theta)
    theta = None
    mba1 = numpy.einsum('jc,jb->bc', l1, t1) + mba
    mij1 = numpy.einsum('kb,jb->kj', l1, t1) + mij
    mia1 = t1 + numpy.einsum('kc,jkbc->jb', l1, t2) * 2
    mia1 -= numpy.einsum('kc,jkcb->jb', l1, t2)
    mia1 -= reduce(numpy.dot, (t1, l1.T, t1))
    mia1 -= numpy.einsum('bd,jd->jb', mba, t1)
    mia1 -= numpy.einsum('lj,lb->jb', mij, t1)

    tmp = mycc.add_wvvVV(numpy.zeros_like(l1), l2, eris)
    l2new = ccsd._unpack_t2_tril(tmp, nocc, nvir)
    l2new *= .5  # *.5 because of l2+l2.transpose(1,0,3,2) in the end
    l1new  = numpy.einsum('ijab,jb->ia', l2new, t1) * 4
    l1new -= numpy.einsum('jiab,jb->ia', l2new, t1) * 2
    tmp = tmp1 = None

    l1new += fov
    l1new += numpy.einsum('ib,ba->ia', l1, saved.w1)
    l1new -= numpy.einsum('ja,ij->ia', l1, saved.w2)
    l1new -= numpy.einsum('ik,ka->ia', mij, saved.w4)
    l1new -= numpy.einsum('ca,ic->ia', mba, saved.w4)
    l1new += numpy.einsum('ijab,bj->ia', l2, saved.w3) * 2
    l1new -= numpy.einsum('ijba,bj->ia', l2, saved.w3)

    l2new += numpy.einsum('ia,jb->ijab', l1, saved.w4)
    l2new += einsum('jibc,ca->jiba', l2, saved.w1)
    l2new -= einsum('jk,kiba->jiba', saved.w2, l2)

    eris_vooo = _cp(eris.vooo)
    l1new -= numpy.einsum('aijk,kj->ia', eris_vooo, mij1) * 2
    l1new += numpy.einsum('ajik,kj->ia', eris_vooo, mij1)
    l2new -= einsum('bjki,ka->jiba', eris_vooo, l1)
    eris_vooo = None

    tau = _ccsd.make_tau(t2, t1, t1)
    l2tau = einsum('ijcd,klcd->ijkl', l2, tau)
    tau = None
    l2t1 = einsum('jidc,kc->ijkd', l2, t1)

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2*5
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    log.debug1('block size = %d, nocc = %d is divided into %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))

    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_vovv = _cp(eris.vovv[p0:p1])
        eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,-1))
        eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)

        l1new[:,p0:p1] += numpy.einsum('aibc,bc->ia', eris_vovv, mba1) * 2
        l1new -= numpy.einsum('bica,bc->ia', eris_vovv, mba1[p0:p1])
        l2new[:,:,p0:p1] += einsum('bjac,ic->jiba', eris_vovv, l1)
        m4 = einsum('ijkd,akdb->ijab', l2t1, eris_vovv)
        l2new[:,:,p0:p1] -= m4
        l1new[:,p0:p1] -= numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijab,ia->jb', m4, t1[:,p0:p1]) * 2
        l1new[:,p0:p1] += numpy.einsum('jiab,jb->ia', m4, t1)
        l1new += numpy.einsum('jiab,ia->jb', m4, t1[:,p0:p1])
        eris_vovv = m4buf = m4 = None

        eris_voov = _cp(eris.voov[p0:p1])
        l1new[:,p0:p1] += numpy.einsum('jb,aijb->ia', l1, eris_voov) * 2
        l2new[:,:,p0:p1] += eris_voov.transpose(1,2,0,3) * .5
        l2new[:,:,p0:p1] -= einsum('bjic,ca->jiba', eris_voov, mba1)
        l2new[:,:,p0:p1] -= einsum('bjka,ik->jiba', eris_voov, mij1)
        l1new[:,p0:p1] += numpy.einsum('aijb,jb->ia', eris_voov, mia1) * 2
        l1new -= numpy.einsum('bija,jb->ia', eris_voov, mia1[:,p0:p1])
        m4 = einsum('ijkl,aklb->ijab', l2tau, eris_voov)
        l2new[:,:,p0:p1] += m4 * .5
        l1new[:,p0:p1] += numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijba,jb->ia', m4, t1[:,p0:p1])

        eris_vvoo = _cp(eris.vvoo[p0:p1])
        l1new[:,p0:p1] -= numpy.einsum('jb,abji->ia', l1, eris_vvoo)
        saved_wvooo = _cp(saved.wvooo[p0:p1])
        l1new -= einsum('ckij,jkca->ia', saved_wvooo, l2[:,:,p0:p1])
        saved_wvovv = _cp(saved.wvvov[p0:p1])
        # Watch out memory usage here, due to the l2 transpose
        l1new[:,p0:p1] += einsum('abkc,kibc->ia', saved_wvovv, l2)
        saved_wvooo = saved_wvovv = None

        saved_wvOOv = _cp(saved.wvOOv[p0:p1])
        tmp_voov = _cp(saved.wVOov[p0:p1]) * 2
        tmp_voov += saved_wvOOv
        tmp = l2.transpose(0,2,1,3) - l2.transpose(0,3,1,2)*.5
        l2new[:,:,p0:p1] += einsum('iakc,bjkc->jiba', tmp, tmp_voov)
        tmp = tmp1 = tmp_ovov = None

        tmp = einsum('jkca,bikc->jiba', l2, saved_wvOOv)
        l2new[:,:,p0:p1] += tmp
        l2new[:,:,p0:p1] += tmp.transpose(1,0,2,3) * .5
        saved_wvOOv = tmp = None

    saved_woooo = _cp(saved.woooo)
    m3 = einsum('ijkl,klab->ijab', saved_woooo, l2)
    l2new += m3 * .5
    l1new += numpy.einsum('ijab,jb->ia', m3, t1) * 2
    l1new -= numpy.einsum('ijba,jb->ia', m3, t1)
    saved_woooo = m3 = None
    #time1 = log.timer_debug1('lambda pass [%d:%d]'%(p0, p1), *time1)

    mo_e = eris.fock.diagonal()
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    l1new /= eia
    l1new += l1

#    l2new = l2new + l2new.transpose(1,0,3,2)
#    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
#    l2new += l2
    ij = 0
    for i in range(nocc):
        if i > 0:
            l2new[i,:i] += l2new[:i,i].transpose(0,2,1)
            l2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            l2new[:i,i] = l2new[i,:i].transpose(0,2,1)
        l2new[i,i] = l2new[i,i] + l2new[i,i].T
        l2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd

    mol = gto.M()
    mf = scf.RHF(mol)

    mcc = ccsd.CCSD(mf)

    numpy.random.seed(12)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = numpy.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*2
    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = numpy.random.random((nocc,nvir))
    l2 = numpy.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    eris = lambda:None
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.vooo = eri0[nocc:,:nocc,:nocc,:nocc].copy()
    eris.vvoo = eri0[nocc:,nocc:,:nocc,:nocc].copy()
    eris.voov = eri0[nocc:,:nocc,:nocc,nocc:].copy()
    idx = numpy.tril_indices(nvir)
    eris.vovv = eri0[nocc:,:nocc,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
    eris.vvvv = ao2mo.restore(4,eri0[nocc:,nocc:,nocc:,nocc:],nvir)
    eris.fock = fock0

    saved = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_amps(mcc, t1, t2, l1, l2, eris, saved)
    print(lib.finger(l1new) - -6699.5335665027187)
    print(lib.finger(l2new) - -514.7001243502192 )
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())

    mcc.max_memory = 0
    saved = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_amps(mcc, t1, t2, l1, l2, eris, saved)
    print(lib.finger(l1new) - -6699.5335665027187)
    print(lib.finger(l2new) - -514.7001243502192 )
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()

    nmo = rhf.mo_energy.size
    fock0 = numpy.diag(rhf.mo_energy)
    nocc = mol.nelectron // 2
    nvir = nmo - nocc

    eris = mcc.ao2mo()
    conv, l1, l2 = kernel(mcc, eris, t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.0132626841292)
    print(numpy.linalg.norm(l2)-0.212575609057)

    from pyscf.cc import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(numpy.dot, (rhf.mo_coeff.T, rhf.get_hcore(), rhf.mo_coeff))
    eri = ao2mo.full(rhf._eri, rhf.mo_coeff)
    eri = ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = numpy.einsum('pq,pq', h1, dm1)
    e2 = numpy.einsum('pqrs,pqrs', eri, dm2) * .5
    print(e1+e2+mol.energy_nuc() - rhf.e_tot - ecc)
