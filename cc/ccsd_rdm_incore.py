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
from pyscf import ao2mo
from pyscf.cc import _ccsd

#
# JCP, 95, 2623
# JCP, 95, 2639
#

def gamma1_intermediates(mycc, t1, t2, l1, l2):
    nocc, nvir = t1.shape
    doo =-numpy.einsum('ja,ia->ij', l1, t1)
    dvv = numpy.einsum('ia,ib->ab', l1, t1)
    dvo = l1.T
    xtv = numpy.einsum('ie,me->im', t1, l1)
    dov = t1 - numpy.einsum('im,ma->ia', xtv, t1)
    #:doo -= numpy.einsum('jkab,ikab->ij', l2, theta)
    #:dvv += numpy.einsum('jica,jicb->ab', l2, theta)
    #:xt1  = numpy.einsum('mnef,inef->mi', l2, make_theta(t2))
    #:xt2  = numpy.einsum('mnaf,mnef->ea', l2, make_theta(t2))
    #:dov += numpy.einsum('imae,me->ia', make_theta(t2), l1)
    #:dov -= numpy.einsum('ma,ie,me->ia', t1, t1, l1)
    #:dov -= numpy.einsum('mi,ma->ia', xt1, t1)
    #:dov -= numpy.einsum('ie,ae->ia', t1, xt2)
    theta = make_theta(t2)
    doo -= lib.dot(theta.reshape(nocc,-1), l2.reshape(nocc,-1).T)
    dov += numpy.einsum('imae,me->ia', theta, l1)
    xt1 = lib.dot(l2.reshape(nocc,-1), theta.reshape(nocc,-1).T)
    dov -= numpy.einsum('mi,ma->ia', xt1, t1)
    xt2 = lib.dot(theta.reshape(-1,nvir).T, l2.reshape(-1,nvir))
    dov -= numpy.einsum('ie,ae->ia', t1, xt2)
    dvv += lib.dot(l2.reshape(-1,nvir).T, theta.reshape(-1,nvir))
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(mycc, t1, t2, l1, l2):
    return gamma2_incore(mycc, t1, t2, l1, l2)

def gamma2_incore(mycc, t1, t2, l1, l2):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir

    time1 = time.clock(), time.time()
    #:theta = make_theta(t2)
    #:mOvOv = numpy.einsum('ikca,jkcb->jbia', l2, t2)
    #:mOVov = -numpy.einsum('ikca,jkbc->jbia', l2, t2)
    #:mOVov += numpy.einsum('ikac,jkbc->jbia', l2, theta)
    l2a = numpy.empty((nocc,nvir,nocc,nvir))
    t2a = numpy.empty((nocc,nvir,nocc,nvir))
    for i in range(nocc):
        l2a[i] = l2[i].transpose(2,0,1)
        t2a[i] = t2[i].transpose(2,0,1)
    mOvOv = lib.dot(t2a.reshape(-1,nov), l2a.reshape(-1,nov).T).reshape(nocc,nvir,nocc,nvir)
    for i in range(nocc):
        t2a[i] = t2[i].transpose(1,0,2)
    mOVov = lib.dot(t2a.reshape(-1,nov), l2a.reshape(-1,nov).T, -1).reshape(nocc,nvir,nocc,nvir)
    theta = t2a
    for i in range(nocc):
        l2a[i] = l2[i].transpose(1,0,2)
        theta[i] *= 2
        theta[i] -= t2[i].transpose(2,0,1)
    lib.dot(theta.reshape(-1,nov), l2a.reshape(nov,-1).T, 1, mOVov.reshape(nov,-1), 1)
    theta = l2a = t2a = None
    moo =(numpy.einsum('jdld->jl', mOvOv) * 2
        + numpy.einsum('jdld->jl', mOVov))
    mvv =(numpy.einsum('lbld->bd', mOvOv) * 2
        + numpy.einsum('lbld->bd', mOVov))
    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2
        - numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1) + moo*.5

    gooov = numpy.zeros((nocc,nocc,nocc,nvir))
    tau = _ccsd.make_tau(t2, t1, t1)
    #:goooo = numpy.einsum('ijab,klab->klij', l2, tau)*.5
    goooo = lib.dot(tau.reshape(-1,nvir**2), l2.reshape(-1,nvir**2).T, .5)
    goooo = goooo.reshape(-1,nocc,nocc,nocc)
    doooo = _cp(make_theta(goooo).transpose(0,2,1,3))

    #:gooov -= numpy.einsum('ib,kjab->jkia', l1, tau)
    #:gooov -= numpy.einsum('kjab,ib->jkia', l2, t1)
    #:gooov += numpy.einsum('jkil,la->jkia', goooo, t1*2)
    gooov = lib.dot(_cp(tau.reshape(-1,nvir)), l1.T, -1)
    lib.dot(_cp(l2.reshape(-1,nvir)), t1.T, -1, gooov, 1)
    gooov = gooov.reshape(nocc,nocc,nvir,nocc)
    tmp = numpy.einsum('ji,ka->jkia', moo*-.5, t1)
    tmp += gooov.transpose(1,0,3,2)
    gooov, tmp = tmp, None
    lib.dot(goooo.reshape(-1,nocc), t1, 2, gooov.reshape(-1,nvir), 1)

    goovv = numpy.einsum('ia,jb->ijab', mia, t1)
    for i in range(nocc):
        goovv[i] += .5 * l2 [i]
        goovv[i] += .5 * tau[i]
    #:goovv -= numpy.einsum('jk,kiba->jiba', mij, tau)
    lib.dot(mij, tau.reshape(nocc,-1), -1, goovv.reshape(nocc,-1), 1)
    #:goovv -= numpy.einsum('cb,ijac->ijab', mab, t2)
    #:goovv -= numpy.einsum('bd,ijad->ijab', mvv*.5, tau)
    lib.dot(t2.reshape(-1,nvir), mab, -1, goovv.reshape(-1,nvir), 1)
    lib.dot(tau.reshape(-1,nvir), mvv.T, -.5, goovv.reshape(-1,nvir), 1)
    tau = None

    #:gooov += numpy.einsum('jaic,kc->jkia', mOvOv, t1)
    #:gooov -= numpy.einsum('kaic,jc->jkia', mOVov, t1)
    tmp = lib.dot(mOvOv.reshape(-1,nvir), t1.T).reshape(nocc,-1,nocc,nocc)
    gooov += tmp.transpose(0,3,2,1)
    lib.dot(t1, mOVov.reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1), 0)
    gooov -= tmp.reshape(nocc,nocc,nvir,nocc).transpose(0,1,3,2)
    dooov = gooov.transpose(0,2,1,3)*2 - gooov.transpose(1,2,0,3)
    gooov = None
    #:tmp = numpy.einsum('ikac,jc->jaik', l2, t1)
    #:gOvVo -= numpy.einsum('jaik,kb->jabi', tmp, t1)
    #:gOvvO = numpy.einsum('jaki,kb->jabi', tmp, t1) + mOvOv.transpose(0,3,1,2)
    tmp = tmp.reshape(nocc,nocc,nocc,nvir)
    lib.dot(t1, l2.reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1))
    gOvVo = numpy.einsum('ia,jb->jabi', l1, t1)
    gOvvO = numpy.empty((nocc,nvir,nvir,nocc))
    for i in range(nocc):
        gOvVo[i] -= lib.dot(_cp(tmp[i].transpose(0,2,1).reshape(-1,nocc)),
                            t1).reshape(nocc,nvir,-1).transpose(1,2,0)
        gOvVo[i] += mOVov[i].transpose(2,0,1)
        gOvvO[i] = lib.dot(tmp[i].reshape(nocc,-1).T,
                           t1).reshape(nocc,nvir,-1).transpose(1,2,0)
        gOvvO[i] += mOvOv[i].transpose(2,0,1)
    tmp = None

    dovvo = numpy.empty((nocc,nvir,nvir,nocc))
    doovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        tmp = gOvVo[i] * 2 + gOvvO[i]
        dovvo[i] = tmp.transpose(1,0,2)
        tmp = gOvvO[i] * -2 - gOvVo[i]
        doovv[i] = tmp.transpose(2,0,1)
    gOvvO = gOvVo = None

    tau2 = _ccsd.make_tau(t2, t1, t1)
    #:goovv += numpy.einsum('ijkl,klab->ijab', goooo[:,:,j0:j1], tau2)
    lib.dot(goooo.reshape(nocc*nocc,-1),
            tau2.reshape(-1,nvir**2), 1, goovv.reshape(-1,nvir**2), 1)
    tau2 += numpy.einsum('ia,jb->ijab', t1, t1)
    tau2p = tau2.reshape(nocc,nvir,nocc,nvir)
    for i in range(nocc):
        tau2p[i] = tau2[i].transpose(2,0,1)
    tau2, tau2p = tau2p.reshape(nov,-1), None
    #:goovv += numpy.einsum('ibld,jlda->ijab', mOvOv, tau2) * .5
    #:goovv -= numpy.einsum('iald,jldb->ijab', mOVov, tau2) * .5
    tmp = lib.dot(mOvOv.reshape(-1,nov), tau2.T, .5).reshape(nocc,nvir,-1,nvir)
    for i in range(nocc):
        tmp[i] = goovv[i].transpose(1,0,2) + tmp[i].transpose(2,1,0)
    goovv, tmp = tmp, None
    lib.dot(mOVov.reshape(-1,nov), tau2.T, -.5, goovv.reshape(nov,-1), 1)

    #:goovv += numpy.einsum('iald,jlbd->ijab', mOVov*2+mOvOv, t2) * .5
    t2a, tau2 = tau2.reshape(nocc,nvir,nocc,nvir), None
    for i in range(nocc):
        t2a[i] = t2[i].transpose(1,0,2)
    tmp = mOVov*2
    tmp += mOvOv
    lib.dot(tmp.reshape(-1,nov), t2a.reshape(nov,-1), .5, goovv.reshape(nov,-1), 1)
    t2a = tmp = None
    for i in range(nocc):
        goovv[i] = goovv[i] * 2 - goovv[i].transpose(2,1,0)
    dovov = goovv
    goooo = goovv = None

    #:gvovv += numpy.einsum('aick,kb->aibc', pvOVo, t1)
    mOVov = lib.transpose(mOVov.reshape(nov,-1))
    gvovv = lib.dot(mOVov.reshape(nocc,-1).T, t1).reshape(nvir,nocc,nvir,nvir)
    mOVov = None
    tmp = numpy.einsum('ja,jb->ab', l1, t1)
    #:gvovv += numpy.einsum('ab,ic->aibc', tmp, t1)
    #:gvovv += numpy.einsum('ba,ic->aibc', mvv, t1*.5)
    for i in range(nvir):
        gvovv[i] += numpy.einsum('b,ic->icb', tmp[i], t1)
        gvovv[i] += numpy.einsum('b,ic->icb', mvv[:,i]*.5, t1)
        gvovv[i] = gvovv[i].transpose(0,2,1)

    #:gvovv += numpy.einsum('ja,jibc->aibc', l1, t2)
    #:gvovv += numpy.einsum('jibc,ja->aibc', l2, t1)
    #:gvovv -= numpy.einsum('aibk,kc->aibc', pvOvO, t1)
    mOvOv = lib.transpose(mOvOv.reshape(nov,-1))
    lib.dot(mOvOv.reshape(nocc,-1).T, t1, -1, gvovv.reshape(-1,nvir), 1)
    mOvOv = None
    lib.dot(l1.T, t2.reshape(nocc,-1), 1, gvovv.reshape(nvir,-1), 1)
    lib.dot(t1.T, l2.reshape(nocc,-1), 1, gvovv.reshape(nvir,-1), 1)

    tmp = numpy.empty((nocc,nvir,nvir))
    for i in range(nvir):
        #:gvovv*2 - gvovv.transpose(0,1,3,2)
        gvovv[i] = _ccsd.make_021(gvovv[i], gvovv[i], 2, -1, out=tmp)

    #:gvvvv = numpy.einsum('ijab,ijcd->abcd', l2, t2)*.5
    #:jabc = numpy.einsum('ijab,ic->jabc', l2, t1) * .5
    #:gvvvv += numpy.einsum('jabc,jd->abcd', jabc, t1)
    #:gvovv -= numpy.einsum('adbc,id->aibc', gvvvv, t1*2)
    tau = _ccsd.make_tau(t2, t1, t1)
    theta = make_theta(tau)
    tau = None
    l2tmp = lib.pack_tril(l2.reshape(-1,nvir,nvir))
    gtmp = lib.dot(l2tmp.T, theta.reshape(nocc**2,-1), .5).reshape(-1,nvir,nvir)
    l2tmp = theta = None
    nvir_pair = nvir * (nvir+1) //2
    tmp = numpy.empty((nvir,nvir,nvir))
    tmp1 = numpy.empty((nvir,nvir,nvir))
    tmptril = numpy.empty((nvir,nvir_pair))
    diag_idx = numpy.arange(nvir)
    diag_idx = diag_idx*(diag_idx+1)//2 + diag_idx

    dvvvv = numpy.empty((nvir_pair,nvir_pair))
    dovvv = numpy.empty((nocc,nvir,nvir,nvir))
# dvvov = (gvovv*2 - gvovv.transpose(0,1,3,2)).transpose(0,2,1,3)
# dovvv = dvvov.transpose(2,3,0,1)
    p0 = 0
    for i in range(nvir):
        tmp[:i+1] = gtmp[p0:p0+i+1]
        for j in range(i+1, nvir):
            tmp[j] = gtmp[j*(j+1)//2+i].T
        lib.dot(t1, tmp.reshape(nvir,-1), -2, gvovv[i].reshape(nocc,-1), 1)
        dovvv[:,:,i] = gvovv[i].transpose(0,2,1)

        #:gvvvv[i] = (tmp*2-tmp.transpose(0,2,1)).transpose(1,0,2)
        #:gvvvv = .5*(gvvvv+gvvvv.transpose(0,1,3,2))
        #:dvvvv = .5*(gvvvv+gvvvv.transpose(1,0,3,2))
        tmp1[:] = tmp.transpose(1,0,2)
        _ccsd.precontract(tmp1, diag_fac=2, out=tmptril)
        dvvvv[p0:p0+i] += tmptril[:i]
        dvvvv[p0:p0+i] *= .25
        dvvvv[i*(i+1)//2+i] = tmptril[i] * .5
        for j in range(i+1, nvir):
            dvvvv[j*(j+1)//2+i] = tmptril[j]
        p0 += i + 1
    gtmp = tmp = tmp1 = tmptril = gvovv = None
    dvvov = dovvv.transpose(2,3,0,1)
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

def make_rdm1(mycc, t1, t2, l1, l2, d1=None):
    if d1 is None:
        doo, dov, dvo, dvv = gamma1_intermediates(mycc, t1, t2, l1, l2)
    else:
        doo, dov, dvo, dvv = d1
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm1 = numpy.empty((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[:nocc,nocc:] = dov + dvo.T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].T
    dm1[nocc:,nocc:] = dvv + dvv.T
    for i in range(nocc):
        dm1[i,i] += 2
    return dm1

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, d1=None, d2=None):
    if d1 is None:
        doo, dov, dvo, dvv = gamma1_intermediates(mycc, t1, t2, l1, l2)
    else:
        doo, dov, dvo, dvv = d1
    if d2 is None:
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
                gamma2_intermediates(mycc, t1, t2, l1, l2)
    else:
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo))

    dm2[:nocc,nocc:,:nocc,nocc:] = \
            (dovov                   +dovov.transpose(2,3,0,1))
    dm2[nocc:,:nocc,nocc:,:nocc] = \
            (dovov.transpose(1,0,3,2)+dovov.transpose(3,2,1,0))

    dm2[:nocc,:nocc,nocc:,nocc:] = \
            (doovv.transpose(0,1,3,2)+doovv.transpose(1,0,2,3))
    dm2[nocc:,nocc:,:nocc,:nocc] = \
            (doovv.transpose(3,2,0,1)+doovv.transpose(2,3,1,0))
    dm2[:nocc,nocc:,nocc:,:nocc] = \
            (dovvo                   +dovvo.transpose(3,2,1,0))
    dm2[nocc:,:nocc,:nocc,nocc:] = \
            (dovvo.transpose(2,3,0,1)+dovvo.transpose(1,0,3,2))

    dm2[nocc:,nocc:,nocc:,nocc:] = ao2mo.restore(1, dvvvv, nvir)
    dm2[nocc:,nocc:,nocc:,nocc:] *= 4

    dm2[:nocc,:nocc,:nocc,:nocc] =(doooo+doooo.transpose(1,0,3,2)) * 2

    dm2[nocc:,nocc:,:nocc,nocc:] = dvvov
    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,nocc:,:nocc] = dvvov.transpose(1,0,3,2)
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2)

    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2)
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0)

    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[:nocc,nocc:] = dov + dvo.T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].T
    dm1[nocc:,nocc:] = dvv + dvv.T
    for i in range(nocc):
        dm2[i,i,:,:] += dm1 * 2
        dm2[:,:,i,i] += dm1 * 2
        dm2[:,i,i,:] -= dm1
        dm2[i,:,:,i] -= dm1

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2

    return dm2

def _cp(a):
    return numpy.array(a, copy=False, order='C')

def make_theta(t2):
    return _ccsd.make_0132(t2, t2, 2, -1)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import ao2mo

    mol = gto.M()
    mf = scf.RHF(mol)

    mcc = ccsd.CCSD(mf)

    numpy.random.seed(2)
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
    h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))

    eris = lambda:None
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
    eris.fock = fock0

    doo, dov, dvo, dvv = gamma1_intermediates(mcc, t1, t2, l1, l2)
    print((numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc]))*2+20166.329861034799)
    print((numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:]))*2-58078.964019246778)
    print((numpy.einsum('ia,ia', dov, fock0[:nocc,nocc:]))*2+74994.356886784764)
    print((numpy.einsum('ai,ai', dvo, fock0[nocc:,:nocc]))*2-34.010188025702391)

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            gamma2_intermediates(mcc, t1, t2, l1, l2)

    dvvvv = ao2mo.restore(1, dvvvv, nvir)
    print('doooo',numpy.einsum('kilj,kilj', doooo, eris.oooo)*2-15939.9007625418)
    print('dvvvv',numpy.einsum('acbd,acbd', dvvvv, eris.vvvv)*2-37581.823919588 )
    print('dooov',numpy.einsum('jkia,jkia', dooov, eris.ooov)*2-128470.009687716)
    print('dovvv',numpy.einsum('icba,icba', dovvv, eris.ovvv)*2+166794.225195056)
    print('dovov',numpy.einsum('iajb,iajb', dovov, eris.ovov)*2+719279.812916893)
    print('dovvo',numpy.einsum('jbai,jbia', dovvo, eris.ovov)*2
                 +numpy.einsum('jiab,jiba', doovv, eris.oovv)*2+53634.0012286654)

    dm1 = make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = make_rdm2(mcc, t1, t2, l1, l2)
    e2 =(numpy.einsum('ijkl,ijkl', doooo, eris.oooo)*2
        +numpy.einsum('acbd,acbd', dvvvv, eris.vvvv)*2
        +numpy.einsum('jkia,jkia', dooov, eris.ooov)*2
        +numpy.einsum('icba,icba', dovvv, eris.ovvv)*2
        +numpy.einsum('iajb,iajb', dovov, eris.ovov)*2
        +numpy.einsum('jbai,jbia', dovvo, eris.ovov)*2
        +numpy.einsum('ijab,ijab', doovv, eris.oovv)*2
        +numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc])*2
        +numpy.einsum('ia,ia', dov, fock0[:nocc,nocc:])*2
        +numpy.einsum('ai,ai', dvo, fock0[nocc:,:nocc])*2
        +numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:])*2
        +fock0[:nocc].trace()*2
        -numpy.einsum('kkpq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace()*2
        +numpy.einsum('pkkq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace())
    print(e2+794721.197459942)
    print(numpy.einsum('pqrs,pqrs', dm2, eri0)*.5 +
          numpy.einsum('pq,pq', dm1, h1) - e2)

    print(numpy.allclose(dm2, dm2.transpose(1,0,3,2)))
    print(numpy.allclose(dm2, dm2.transpose(2,3,0,1)))

    d1 = numpy.einsum('kkpq->pq', dm2) / 9
    print(numpy.allclose(d1, dm1))
