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
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc.addons import spatial2spin, spin2spatial

einsum = numpy.einsum
#einsum = lib.einsum

def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    eris = _eris_spatial2spin(mycc, eris)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if not isinstance(t1, numpy.ndarray):
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    imds = make_intermediates(mycc, t1, t2, eris)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1, t2)
    cput0 = log.timer('UCCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, imds)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1 = spin2spatial(l1, eris.orbspin)
            l2 = spin2spatial(l2, eris.orbspin)
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
            l1 = spatial2spin(l1, eris.orbspin)
            l2 = spatial2spin(l2, eris.orbspin)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('UCCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    l1 = spin2spatial(l1, eris.orbspin)
    l2 = spin2spatial(l2, eris.orbspin)
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2

    v1 = fvv - einsum('ja,jb->ba', fov, t1)
    ovvv = numpy.asarray(eris.ovvv)
    ovvv = ovvv - ovvv.transpose(0,3,2,1)
    v1-= einsum('jabc,jc->ba', ovvv, t1)
    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    v1+= einsum('jcka,jkbc->ba', ovov, tau) * .5

    v2 = foo + einsum('ib,jb->ij', fov, t1)
    ooov = numpy.asarray(eris.ooov)
    ooov = ooov - ooov.transpose(2,1,0,3)
    v2-= einsum('kjib,kb->ij', ooov, t1)
    v2+= einsum('ibkc,jkbc->ij', ovov, tau) * .5

    v3 = einsum('icjd,klcd->ijkl', ovov, tau)
    v4 = einsum('ldjb,klcd->jcbk', ovov, t2)
    v4+= numpy.asarray(eris.ovvo).transpose(0,2,1,3)
    v4-= numpy.einsum('jkcb->jcbk', numpy.asarray(eris.oovv))

    v5 = fvo + einsum('kc,jkbc->bj', fov, t2)
    tmp = fov - einsum('kdlc,ld->kc', ovov, t1)
    v5+= numpy.einsum('kc,kb,jc->bj', tmp, t1, t1)
    v5-= einsum('kjlc,klbc->bj', ooov, t2) * .5
    v5+= einsum('kdbc,jkcd->bj', ovvv, t2) * .5

    w3 = v5 + einsum('jcbk,jb->ck', v4, t1)
    w3 += numpy.einsum('cb,jb->cj', v1, t1)
    w3 -= numpy.einsum('jk,jb->bk', v2, t1)

    woooo = numpy.asarray(eris.oooo).transpose(0,2,1,3)
    woooo = (woooo - woooo.transpose(1,0,2,3)) * .5
    woooo+= v3 * .25
    woooo+= einsum('jlic,kc->ijkl', ooov, t1)

    wovvo = v4 - numpy.einsum('ldjb,lc,kd->jcbk', ovov, t1, t1)
    wovvo-= einsum('lkjb,lc->jcbk', ooov, t1)
    wovvo+= einsum('jbcd,kd->jcbk', ovvv, t1)

    woovo = einsum('idcb,kjbd->kjci', ovvv, tau) * .25
    woovo+= numpy.einsum('jikc->kjci', ooov.conj()) * .5
    woovo+= einsum('icbk,jb->kjci', v4, t1)

    wovvv = einsum('jkla,jlbc->kbca', ooov, tau) * .25
    wovvv-= numpy.einsum('kcab->kbca', ovvv.conj()) * .5
    wovvv+= einsum('jcak,jb->kbca', v4, t1)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), 'f8')
    imds.woovo = imds.ftmp.create_dataset('woovo', (nocc,nocc,nvir,nocc), 'f8')
    imds.wovvv = imds.ftmp.create_dataset('wovvv', (nocc,nvir,nvir,nvir), 'f8')
    imds.woooo[:] = woooo
    imds.wovvo[:] = wovvo
    imds.woovo[:] = woovo
    imds.wovvv[:] = wovvv
    imds.v1 = v1
    imds.v2 = v2
    imds.v3 = v3
    imds.v4 = v4
    imds.w3 = w3
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris, imds):
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    l1new = numpy.zeros_like(l1)
    l2new = numpy.zeros_like(l2)

    mba = einsum('klca,klcb->ba', l2, t2) * .5
    mij = einsum('kicd,kjcd->ij', l2, t2) * .5
    m3 = numpy.einsum('klab,ijkl->ijab', l2, numpy.asarray(imds.woooo))
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    tmp = numpy.einsum('ijcd,klcd->ijkl', l2, tau)
    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    m4 = numpy.einsum('kalb,ijkl->ijab', ovov, tmp) * .25
    ovvv = numpy.asarray(eris.ovvv)
    ovvv = ovvv - ovvv.transpose(0,3,2,1)
    tmp = numpy.einsum('ijcd,kd->ijck', l2, t1)
    m4 -= numpy.einsum('kbca,ijck->ijab', ovvv, tmp)
    vvvv = numpy.asarray(eris.vvvv)
    vvvv = vvvv - vvvv.transpose(0,3,2,1)
    m4 += numpy.einsum('ijcd,cadb->ijab', l2, vvvv) * .5
    vvvv = None

    ooov = numpy.asarray(eris.ooov)
    ooov = ooov - ooov.transpose(2,1,0,3)

    l2new += ovov.transpose(0,2,1,3)
    l2new += m3
    l2new += m4
    fov1 = fov + einsum('kcjb,kc->jb', ovov, t1)
    tmp = einsum('ia,jb->ijab', l1, fov1)
    tmp+= einsum('kica,jcbk->ijab', l2, numpy.asarray(imds.wovvo))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,ikjb->ijab', l1, ooov)
    tmp+= einsum('ijca,cb->ijab', l2, imds.v1)
    m1tmp = mba + einsum('ka,kc->ca', l1, t1)
    tmp+= einsum('ca,icjb->ijab', m1tmp, ovov)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ic,jbca->ijab', l1, ovvv)
    tmp+= einsum('kiab,jk->ijab', l2, imds.v2)
    m2tmp = mij + einsum('ic,kc->ik', l1, t1)
    tmp-= einsum('ik,kajb->ijab', m2tmp, ovov)
    l2new += tmp - tmp.transpose(1,0,2,3)

    l1new += fov
    ovvo = (numpy.asarray(eris.ovvo).transpose(0,2,1,3) -
            numpy.einsum('ijba->ibaj', numpy.asarray(eris.oovv)))
    l1new += einsum('jb,ibaj->ia', l1, ovvo)
    l1new += einsum('ib,ba->ia', l1, imds.v1)
    l1new -= einsum('ja,ij->ia', l1, imds.v2)
    l1new -= einsum('kjca,kjci->ia', l2, numpy.asarray(imds.woovo))
    l1new -= einsum('ikbc,kbca->ia', l2, numpy.asarray(imds.wovvv))
    l1new += einsum('ijab,jb->ia', m4, t1)
    l1new += einsum('ijab,jb->ia', m3, t1)
    l1new += einsum('jiba,bj->ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('kc,jc,kb->jb', l1, t1, t1)
          - einsum('bd,jd->jb', mba, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += numpy.einsum('jbia,jb->ia', ovov, tmp)
    tmp = numpy.einsum('jc,jb->bc', l1, t1) + mba
    l1new += numpy.einsum('iacb,bc->ia', ovvv, tmp)
    tmp = numpy.einsum('kb,jb->kj', l1, t1) + mij
    l1new -= numpy.einsum('jkia,kj->ia', ooov, tmp)
    tmp = fov - einsum('kbja,jb->ka', ovov, t1)
    l1new -= numpy.einsum('ik,ka->ia', mij, tmp)
    tmp = fov - einsum('ibjc,jb->ic', ovov, t1)
    l1new -= numpy.einsum('ca,ic->ia', mba, tmp)

    tmp = einsum('kacd,jkbd->jacb', ovvv, t2)
    l1new -= einsum('ijcb,jacb->ia', l2, tmp)
    tmp = einsum('lkic,jlbc->jkib', ooov, t2)
    l1new += einsum('kjab,jkib->ia', l2, tmp)

    mo_e = eris.fock.diagonal()
    eia = lib.direct_sum('i-j->ij', mo_e[:nocc], mo_e[nocc:])
    l1new /= eia
    l1new += l1
    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def _eris_spatial2spin(cc, eris):
    '''Convert to spin-orbital ERIs'''
    if not hasattr(eris, 'ovOV'):
        return eris

    class _ERIS: pass
    eris1 = _ERIS()
    eris1.orbspin = eris.orbspin

    nocc = cc.nocc
    occidxa = numpy.where(eris1.orbspin[:nocc] == 0)[0]
    occidxb = numpy.where(eris1.orbspin[:nocc] == 1)[0]
    viridxa = numpy.where(eris1.orbspin[nocc:] == 0)[0]
    viridxb = numpy.where(eris1.orbspin[nocc:] == 1)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nvir = nvira + nvirb

    nmo = nmoa + nmob
    eris1.fock = numpy.zeros((nmo,nmo))
    maska = eris1.orbspin==0
    maskb = eris1.orbspin==1
    eris1.fock[maska.reshape(-1,1) & maska] = eris.focka.ravel()
    eris1.fock[maskb.reshape(-1,1) & maskb] = eris.fockb.ravel()

    dtype = eris.mo_coeff[0].dtype
    eris1.feri = lib.H5TmpFile()
    eris1.oooo = eris1.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), dtype.char)
    eris1.ooov = eris1.feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), dtype.char)
    eris1.ovoo = eris1.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), dtype.char)
    eris1.oovo = eris1.feri.create_dataset('oovo', (nocc,nocc,nvir,nocc), dtype.char)
    eris1.ovov = eris1.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), dtype.char)
    eris1.oovv = eris1.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype.char)
    eris1.ovvo = eris1.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), dtype.char)
    eris1.ovvv = eris1.feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), dtype.char)
    eris1.vvvv = eris1.feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), dtype.char)

    def convert(out, inp, idx1, idx2, idx3, idx4):
        dim1 = inp.shape
        dim2 = out.shape
        assert(dim1 == (len(idx1),len(idx2),len(idx3),len(idx4)))
        locx = (dim2[1] * idx1.reshape(-1,1) + idx2).ravel()
        locy = (dim2[3] * idx3.reshape(-1,1) + idx4).ravel()
        lib.takebak_2d(out.reshape(dim2[0]*dim2[1],dim2[2]*dim2[3]),
                       inp.reshape(dim1[0]*dim1[1],dim1[2]*dim1[3]), locx, locy)
    def convertall(out, inp, idx1, idx2, idx3, idx4):
        convert(out, inp[0], idx1[0], idx2[0], idx3[0], idx4[0])
        convert(out, inp[1], idx1[0], idx2[0], idx3[1], idx4[1])
        convert(out, inp[2], idx1[1], idx2[1], idx3[0], idx4[0])
        convert(out, inp[3], idx1[1], idx2[1], idx3[1], idx4[1])

    oidx = (occidxa, occidxb)
    vidx = (viridxa, viridxb)
    tmp = numpy.zeros((nocc,nocc,nocc,nocc), dtype=dtype)
    oooo = numpy.asarray(eris.oooo)
    ooOO = numpy.asarray(eris.ooOO)
    OOOO = numpy.asarray(eris.OOOO)
    convertall(tmp, (oooo, ooOO, ooOO.transpose(2,3,0,1), OOOO),
               oidx, oidx, oidx, oidx)
    eris1.oooo[:] = tmp
    tmp = oooo = ooOO = OOOO = None

    tmp = numpy.zeros((nocc,nocc,nocc,nvir), dtype=dtype)
    ooov = numpy.asarray(eris.ooov)
    ooOV = numpy.asarray(eris.ooOV)
    ovOO = numpy.asarray(eris.ovOO)
    OOOV = numpy.asarray(eris.OOOV)
    convertall(tmp, (ooov, ooOV, ovOO.transpose(2,3,0,1), OOOV),
               oidx, oidx, oidx, vidx)
    eris1.ooov[:] = tmp
    tmp = None

    tmp = numpy.zeros((nocc,nvir,nocc,nocc), dtype=dtype)
    convertall(tmp, (ooov.transpose(2,3,0,1), ovOO,
                     ooOV.transpose(2,3,0,1), OOOV.transpose(2,3,0,1)),
               oidx, vidx, oidx, oidx)
    eris1.ovoo[:] = tmp

    tmp = numpy.zeros((nocc,nocc,nvir,nocc), dtype=dtype)
    convertall(tmp, (ooov.transpose(0,1,3,2), ooOV.transpose(0,1,3,2),
                     ovOO.transpose(2,3,1,0), OOOV.transpose(0,1,3,2)),
               oidx, oidx, vidx, oidx)
    eris1.oovo[:] = tmp
    tmp = ooov = ooOV = ovOO = OOOV = None

    tmp = numpy.zeros((nocc,nvir,nocc,nvir), dtype=dtype)
    ovov = numpy.asarray(eris.ovov)
    ovOV = numpy.asarray(eris.ovOV)
    OVOV = numpy.asarray(eris.OVOV)
    convertall(tmp, (ovov, ovOV, ovOV.transpose(2,3,0,1), OVOV),
               oidx, vidx, oidx, vidx)
    eris1.ovov[:] = tmp
    tmp = None

    tmp = numpy.zeros((nocc,nvir,nvir,nocc), dtype=dtype)
    convertall(tmp, (ovov.transpose(0,1,3,2), ovOV.transpose(0,1,3,2),
                     ovOV.transpose(2,3,1,0), OVOV.transpose(0,1,3,2)),
               oidx, vidx, vidx, oidx)
    eris1.ovvo[:] = tmp
    tmp = ovov = ovOV = OVOV = None

    tmp = numpy.zeros((nocc,nocc,nvir,nvir), dtype=dtype)
    oovv = numpy.asarray(eris.oovv)
    ooVV = numpy.asarray(eris.ooVV)
    OOvv = numpy.asarray(eris.OOvv)
    OOVV = numpy.asarray(eris.OOVV)
    convertall(tmp, (oovv, ooVV, OOvv, OOVV),
               oidx, oidx, vidx, vidx)
    eris1.oovv[:] = tmp
    tmp = oovv = ooVV = OOvv = OOVV = None

    tmp = numpy.zeros((nocc,nvir,nvir,nvir), dtype=dtype)
    ovvv = lib.unpack_tril(numpy.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    ovVV = lib.unpack_tril(numpy.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    OVvv = lib.unpack_tril(numpy.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    OVVV = lib.unpack_tril(numpy.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    convertall(tmp, (ovvv, ovVV, OVvv, OVVV),
               oidx, vidx, vidx, vidx)
    eris1.ovvv[:] = tmp
    tmp = ovvv = ovVV = OVvv = OVVV = None

    tmp = numpy.zeros((nvir,nvir,nvir,nvir), dtype=dtype)
    vvvv = ao2mo.restore(1, numpy.asarray(eris.vvvv), nvira).reshape(nvira,nvira,nvira,nvira)
    VVVV = ao2mo.restore(1, numpy.asarray(eris.VVVV), nvirb).reshape(nvirb,nvirb,nvirb,nvirb)
    vvVV1 = lib.unpack_tril(numpy.asarray(eris.vvVV))
    vvVV = numpy.zeros((nvira,nvira,nvirb,nvirb),dtype=vvVV1.dtype)
    idx,idy = numpy.tril_indices(nvira)
    vvVV[idx,idy] = vvVV1
    vvVV[idy,idx] = vvVV1
    convertall(tmp, (vvvv, vvVV, vvVV.transpose(2,3,0,1), VVVV),
               vidx, vidx, vidx, vidx)
    eris1.vvvv[:] = tmp
    tmp = vvvv = vvVV = VVVV = vvVV1 = None

    return eris1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import uccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run()
    mycc = uccsd.UCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    conv, l1, l2 = kernel(mycc, eris, mycc.t1, mycc.t2, tol=1e-8)
