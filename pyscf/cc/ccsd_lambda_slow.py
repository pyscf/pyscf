#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
import pyscf.ao2mo
import pyscf.cc.ccsd_slow as ccsd

# t2,l2 as ijab

def kernel(cc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if eris is None: eris = ccsd._ERIS(cc)
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    nocc, nvir = t1.shape
    saved = make_intermediates(cc, t1, t2, eris)

    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda l1,l2,*args: (l1,l2)
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(cc, t1, t2, l1, l2, eris, saved)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if cc.diis:
            l1, l2 = cc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('istep = %d  norm(lambda1,lambda2) = %.6g', istep, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    def make_g2(g):
        if g.shape[1] == g.shape[3]:
            return g * 2 - g.transpose(0,3,2,1)
        else:
            return g * 2 - g.transpose(2,1,0,3)

    tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    w1 =(fvv - numpy.einsum('ja,jb->ba', fov, t1)
         + numpy.einsum('jcba,jc->ba', make_g2(eris.ovvv), t1)
         - numpy.einsum('jakc,jkbc->ba', make_g2(eris.ovov), tau))
    w2 =(foo + numpy.einsum('ib,jb->ij', fov, t1)
         + numpy.einsum('ijkb,kb->ij', make_g2(eris.ooov), t1)
         + numpy.einsum('ibkc,jkbc->ij', make_g2(eris.ovov), tau))
    vOvVo =(eris.ovov.transpose(0,3,1,2)
            + numpy.einsum('ldjb,klcd->jcbk', make_g2(eris.ovov), t2)
            - numpy.einsum('ldjb,kldc->jcbk', eris.ovov, t2))
    vOvvO =(-eris.oovv.transpose(0,2,3,1)
            + numpy.einsum('lbjd,kldc->jcbk', eris.ovov, t2))
    tmp = numpy.einsum('kcld,ld->kc', make_g2(eris.ovov), t1)
    w3 = fov.T + numpy.einsum('kc,jkbc->bj', fov, t2*2-t2.transpose(1,0,2,3))
    w3 += reduce(numpy.dot, (t1.T, fov + tmp, t1.T))
    w3 -= numpy.einsum('kjlc,klbc->bj', make_g2(eris.ooov), t2)
    w3 += numpy.einsum('kdbc,jkcd->bj', make_g2(eris.ovvv), t2)
    w3 += numpy.einsum('kbcj,kc->bj', vOvVo*2+vOvvO, t1)
    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)
    w4 = fov + numpy.einsum('kcjb,kc->jb', make_g2(eris.ovov), t1)
    wOvVo =(vOvVo - numpy.einsum('jbld,lc,kd->jcbk', eris.ovov, t1, t1)
            - numpy.einsum('lkjb,lc->jcbk', eris.ooov, t1)
            + numpy.einsum('jbcd,kd->jcbk', eris.ovvv, t1))
    wOvvO = vOvvO + numpy.einsum('jdlb,lc,kd->jcbk', eris.ovov, t1, t1)
    wOvvO += numpy.einsum('jklb,lc->jcbk', eris.ooov, t1)
    wOvvO -= numpy.einsum('jdcb,kd->jcbk', eris.ovvv, t1)
    vkabc = numpy.einsum('jdca,kjbd->kabc', make_g2(eris.ovvv),
                         t2*2-t2.transpose(0,1,3,2))
    wovvv = numpy.einsum('jkla,jlbc->kabc', make_g2(eris.ooov), tau)
    wovvv += numpy.einsum('jcak,jb->kabc', vOvvO*2+vOvVo, t1)
    wovvv -= numpy.einsum('jbak,jc->kabc', vOvVo*2+vOvvO, t1)
    wovvv += eris.ovvv.transpose(0,2,1,3)*2 - eris.ovvv.transpose(0,2,3,1)
    wovvv += vkabc - vkabc.transpose(0,1,3,2) * .5
    wovvv -= numpy.einsum('jabd,kjdc->kabc', eris.ovvv, t2) * 1.5
    vicjk = numpy.einsum('iklb,jlcb->icjk', make_g2(eris.ooov),
                         t2*2-t2.transpose(0,1,3,2))
    wovoo = numpy.einsum('idcb,jkbd->icjk', make_g2(eris.ovvv), tau)
    wovoo -= numpy.einsum('icbk,jb->icjk', vOvvO*2+vOvVo, t1)
    wovoo += numpy.einsum('icbj,kb->icjk', vOvVo*2+vOvvO, t1)
    wovoo += eris.ooov.transpose(1,3,2,0)*2 - eris.ooov.transpose(1,3,0,2)
    wovoo += vicjk - vicjk.transpose(0,1,3,2)*.5
    wovoo -= numpy.einsum('ljib,klbc->icjk', eris.ooov, t2) * 1.5
    woooo =(eris.oooo.transpose(0,2,1,3)
            + numpy.einsum('icjd,klcd->ijkl', eris.ovov, tau)
            + numpy.einsum('jlic,kc->ijkl', eris.ooov, t1)
            + numpy.einsum('ikjc,lc->ijkl', eris.ooov, t1))

    class _Saved: pass
    saved = _Saved()
    saved.w1 = w1
    saved.w2 = w2
    saved.w3 = w3
    saved.w4 = w4
    saved.wOvVo = wOvVo
    saved.wOvvO = wOvvO
    saved.woooo = woooo
    saved.wovvv = wovvv
    saved.wovoo = wovoo
    return saved


# update L1, L2
def update_amps(cc, t1, t2, l1, l2, eris, saved):
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    fov = eris.fock[:nocc,nocc:]
    l1new = numpy.zeros_like(l1)
    l2new = numpy.zeros_like(l2)

    mba = numpy.einsum('klca,klcb->ba', l2, t2*2-t2.transpose(0,1,3,2))
    mij = numpy.einsum('kicd,kjcd->ij', l2, t2*2-t2.transpose(0,1,3,2))
    m3 = numpy.einsum('klab,ijkl->ijab', l2, saved.woooo)
    tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    tmp = numpy.einsum('ijcd,klcd->ijkl', l2, tau)
    m4 = numpy.einsum('kalb,ijkl->ijab', eris.ovov, tmp)
    tmp = numpy.einsum('ijcd,kd->ijck', l2, t1)
    m4 -= numpy.einsum('kbca,ijck->ijab', eris.ovvv, tmp)
    m4 -= numpy.einsum('kadb,jidk->ijab', eris.ovvv, tmp)
    m4 += numpy.einsum('ijcd,cadb->ijab', l2, eris.vvvv)

    l2new += numpy.einsum('ia,jb->ijab', l1, saved.w4)
    l2new +=-numpy.einsum('ka,ikjb->ijab', l1, eris.ooov)
    l2new += numpy.einsum('ic,jbca->ijab', l1, eris.ovvv)
    l2new += numpy.einsum('ijcb,ca->ijab', l2, saved.w1)
    l2new +=-numpy.einsum('ikab,jk->ijab', l2, saved.w2)
    tmp = numpy.einsum('jc,jb->bc', l1, t1) + mba
    l2new +=-numpy.einsum('icjb,ca->ijab', eris.ovov, tmp)
    tmp = numpy.einsum('kb,jb->kj', l1, t1) + mij
    l2new +=-numpy.einsum('kajb,ik->ijab', eris.ovov, tmp)
    l2new += numpy.einsum('kica,jcbk->ijab', l2-l2.transpose(0,1,3,2)*.5,
                          saved.wOvVo*2+saved.wOvvO)
    tmp = numpy.einsum('jkca,icbk->ijab', l2, saved.wOvvO)
    l2new += tmp + tmp.transpose(1,0,2,3) * .5
    l2new = l2new + l2new.transpose(1,0,3,2)
    l2new += m3 + m4
    l2new += eris.ovov.transpose(0,2,1,3)

    l1new += fov
    l1new += numpy.einsum('jb,iajb->ia', l1, eris.ovov) * 2
    l1new +=-numpy.einsum('jb,ijba->ia', l1, eris.oovv)
    l1new += numpy.einsum('ib,ba->ia', l1, saved.w1)
    l1new +=-numpy.einsum('ja,ij->ia', l1, saved.w2)
    l1new += numpy.einsum('ijab,bj->ia', l2, saved.w3) * 2
    l1new +=-numpy.einsum('ijba,bj->ia', l2, saved.w3)
    l1new +=-numpy.einsum('kjac,icjk->ia', l2, saved.wovoo)
    l1new += numpy.einsum('ikcb,kabc->ia', l2, saved.wovvv)
    l1new += numpy.einsum('ijab,jb->ia', m4, t1) * 2
    l1new +=-numpy.einsum('jiab,jb->ia', m4, t1)
    l1new += numpy.einsum('ijab,jb->ia', m3, t1) * 2
    l1new +=-numpy.einsum('jiab,jb->ia', m3, t1)
    tmp =(t1 + numpy.einsum('kc,kjcb->jb', l1, t2) * 2
          - numpy.einsum('kc,kjbc->jb', l1, t2)
          - numpy.einsum('kc,jc,kb->jb', l1, t1, t1)
          - numpy.einsum('bd,jd->jb', mba, t1)
          - numpy.einsum('lj,lb->jb', mij, t1))
    l1new += numpy.einsum('jbia,jb->ia', eris.ovov, tmp) * 2
    l1new +=-numpy.einsum('jaib,jb->ia', eris.ovov, tmp)
    tmp = numpy.einsum('jc,jb->bc', l1, t1) + mba
    l1new += numpy.einsum('iacb,bc->ia', eris.ovvv, tmp) * 2
    l1new +=-numpy.einsum('ibca,bc->ia', eris.ovvv, tmp)
    tmp = numpy.einsum('kb,jb->kj', l1, t1) + mij
    l1new +=-numpy.einsum('jkia,kj->ia', eris.ooov, tmp) * 2
    l1new += numpy.einsum('ikja,kj->ia', eris.ooov, tmp)
    l1new +=-numpy.einsum('ik,ka->ia', mij, saved.w4)
    l1new +=-numpy.einsum('ca,ic->ia', mba, saved.w4)

    mo_e = eris.fock.diagonal()
    eia = lib.direct_sum('i-j->ij', mo_e[:nocc], mo_e[nocc:])
    l1new /= eia
    l1new += l1

    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import ao2mo

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
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
    eris.fock = fock0

    saved = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_amps(mcc, t1, t2, l1, l2, eris, saved)
    print(abs(l1new).sum()-38172.7896467303)
    print(numpy.dot(l1new.flatten(), numpy.arange(35)) - 739312.005491083)
    print(numpy.dot(l1new.flatten(), numpy.sin(numpy.arange(35)))-7019.50937051188)
    print(numpy.dot(numpy.sin(l1new.flatten()), numpy.arange(35))-69.6652346635955)

    print(abs(l2new).sum()-72035.4931071527)
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())
    print(numpy.dot(l2new.flatten(), numpy.arange(35**2)) - 48427109.5409886)
    print(numpy.dot(l2new.flatten(), numpy.sin(numpy.arange(35**2)))-137.758016736487)
    print(numpy.dot(numpy.sin(l2new.flatten()), numpy.arange(35**2))-507.656936701192)


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

    eri0 = pyscf.ao2mo.restore(1, pyscf.ao2mo.full(rhf._eri, rhf.mo_coeff), nmo)
    eris = lambda:None
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
    eris.fock = fock0
    mcc.ao2mo = lambda *args: eris

    conv, l1, l2 = kernel(mcc, eris, t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.0132626841292)
    print(numpy.linalg.norm(l2)-0.212575609057)

    import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(numpy.dot, (rhf.mo_coeff.T, rhf.get_hcore(), rhf.mo_coeff))
    eri = pyscf.ao2mo.full(rhf._eri, rhf.mo_coeff)
    eri = pyscf.ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = numpy.einsum('pq,pq', h1, dm1)
    e2 = numpy.einsum('pqrs,pqrs', eri, dm2) * .5
    print(e1+e2+mol.energy_nuc() - rhf.e_tot - ecc)
