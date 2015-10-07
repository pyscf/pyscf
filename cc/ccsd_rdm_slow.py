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

# dE = (goo * foo + gvv * fvv + doooo*eri_oooo + ...) * 2
def gamma1_intermediates(cc, t1, t2, l1, l2):
    theta = t2*2 - t2.transpose(0,1,3,2)
    goo = -numpy.einsum('ja,ia->ij', l1, t1)
    goo -= numpy.einsum('jkab,ikab->ij', l2, theta)
    gvv = numpy.einsum('ia,ib->ab', l1, t1)
    gvv += numpy.einsum('ijac,ijbc->ab', l2, theta)
    return goo, gvv

# gamma2 intermediates in Physist's notation
def gamma2_intermediates(cc, t1, t2, l1, l2):
    tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    tau2 = t2 + numpy.einsum('ia,jb->ijab', t1, t1*2)
    theta = t2*2 - t2.transpose(0,1,3,2)

    mOvOv = numpy.einsum('ikca,jkcb->jbia', l2, t2)
    mOVov = numpy.einsum('ikac,jkbc->jbia', l2, theta)
    mOVov -= numpy.einsum('ikca,jkbc->jbia', l2, t2)
    moo =(numpy.einsum('jdld->jl', mOvOv) * 2
        + numpy.einsum('jdld->jl', mOVov))
    mvv =(numpy.einsum('lbld->bd', mOvOv) * 2
        + numpy.einsum('lbld->bd', mOVov))

    gvvvv = numpy.einsum('ijab,ijcd->abcd', l2*.5, tau)

    goooo = numpy.einsum('ijab,klab->klij', l2, tau)*.5

    goovv = .5 * l2 + .5 * tau
    tmp = numpy.einsum('kc,ikac->ia', l1, theta)
    goovv += numpy.einsum('ia,jb->ijab', tmp, t1)
    tmp = numpy.einsum('kc,kb->cb', l1, t1)
    goovv -= numpy.einsum('cb,ijac->ijab', tmp, t2)
    tmp = numpy.einsum('kc,jc->kj', l1, t1)
    goovv -= numpy.einsum('kj,ikab->ijab', tmp, tau)
    goovv -= numpy.einsum('jl,ilab->ijab', moo*.5, tau)
    goovv -= numpy.einsum('bd,ijad->ijab', mvv*.5, tau)
    goovv += numpy.einsum('ibld,ljad->ijab', mOvOv, tau2) * .5
    goovv -= numpy.einsum('iald,ljbd->ijab', mOVov, tau2) * .5
    goovv += numpy.einsum('iald,ljdb->ijab', mOVov*2+mOvOv, t2) * .5
    goovv += numpy.einsum('ijkl,klab->ijab', goooo, tau)

    gooov = numpy.einsum('ib,kjab->jkia', -l1, tau)
    gooov += numpy.einsum('jkil,la->jkia', goooo, t1*2)
    gooov += numpy.einsum('ji,ka->jkia', moo*-.5, t1)
    gooov += numpy.einsum('jaic,kc->jkia', mOvOv, t1)
    gooov -= numpy.einsum('kaic,jc->jkia', mOVov, t1)
    gooov -= numpy.einsum('jkba,ib->jkia', l2, t1)

    gvovv = numpy.einsum('ja,jibc->aibc', l1, tau)
    gvovv -= numpy.einsum('adbc,id->aibc', gvvvv, t1*2)
    gvovv += numpy.einsum('ba,ic->aibc', mvv, t1*.5)
    gvovv -= numpy.einsum('ibka,kc->aibc', mOvOv, t1)
    gvovv += numpy.einsum('icka,kb->aibc', mOVov, t1)
    gvovv += numpy.einsum('jibc,ja->aibc', l2, t1)

    gOvVo = numpy.einsum('ia,jb->jabi', l1, t1) + mOVov.transpose(0,3,1,2)
    tmp = numpy.einsum('ikac,jc->jaik', l2, t1)
    gOvVo -= numpy.einsum('jaik,kb->jabi', tmp, t1)
    gOvvO = mOvOv.transpose(0,3,1,2) + numpy.einsum('jaki,kb->jabi', tmp, t1)

    doovv = goovv*2 - goovv.transpose(0,1,3,2)
    dvvvv = gvvvv*2 - gvvvv.transpose(0,1,3,2)
    doooo = goooo*2 - goooo.transpose(0,1,3,2)
    dovov = -2*gOvvO.transpose(0,1,3,2) - gOvVo.transpose(0,1,3,2)
    dovvo = gOvVo*2 + gOvvO
    dvovv = gvovv*2 - gvovv.transpose(0,1,3,2)
    dooov = gooov*2 - gooov.transpose(1,0,2,3)
    return doovv, dvvvv, doooo, dovov, dovvo, dvovv, dooov

def make_rdm1(cc, t1, t2, l1, l2, d1=None):
    if d1 is None:
        doo, dvv = gamma1_intermediates(cc, t1, t2, l1, l2)
    else:
        doo, dvv = d1
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[nocc:,nocc:] = dvv + dvv.T
    for i in range(nocc):
        dm1[i,i] += 2
    return dm1

# rdm2 in Chemist's notation
def make_rdm2(cc, t1, t2, l1, l2, d1=None, d2=None):
    if d1 is None:
        doo, dvv = gamma1_intermediates(cc, t1, t2, l1, l2)
    else:
        doo, dvv = d1
    if d2 is None:
        doovv, dvvvv, doooo, dovov, dovvo, dvovv, dooov = \
                gamma2_intermediates(cc, t1, t2, l1, l2)
    else:
        doovv, dvvvv, doooo, dovov, dovvo, dvovv, dooov = d2
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo))

    dm2[:nocc,nocc:,:nocc,nocc:] = \
            (doovv.transpose(0,2,1,3)+doovv.transpose(1,3,0,2))
    dm2[nocc:,:nocc,nocc:,:nocc] = \
            (doovv.transpose(2,0,3,1)+doovv.transpose(3,1,2,0))

    dm2[:nocc,:nocc,nocc:,nocc:] = \
            (dovov.transpose(0,2,3,1)+dovov.transpose(2,0,1,3))
    dm2[nocc:,nocc:,:nocc,:nocc] = \
            (dovov.transpose(3,1,0,2)+dovov.transpose(1,3,2,0))
    dm2[:nocc,nocc:,nocc:,:nocc] = \
            (dovvo.transpose(0,2,1,3)+dovvo.transpose(3,1,2,0))
    dm2[nocc:,:nocc,:nocc,nocc:] = \
            (dovvo.transpose(1,3,0,2)+dovvo.transpose(2,0,3,1))

    dm2[nocc:,nocc:,nocc:,nocc:] = \
            (dvvvv.transpose(0,2,1,3)+dvvvv.transpose(2,0,3,1)) * 2

    dm2[:nocc,:nocc,:nocc,:nocc] = \
            (doooo.transpose(0,2,1,3)+doooo.transpose(2,0,3,1)) * 2

    dm2[nocc:,nocc:,:nocc,nocc:] = dvovv.transpose(0,2,1,3)
    dm2[:nocc,nocc:,nocc:,nocc:] = dvovv.transpose(1,3,0,2)
    dm2[nocc:,nocc:,nocc:,:nocc] = dvovv.transpose(2,0,3,1)
    dm2[nocc:,:nocc,nocc:,nocc:] = dvovv.transpose(3,1,2,0)

    dm2[:nocc,:nocc,:nocc,nocc:] = dooov.transpose(0,2,1,3)
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(1,3,0,2)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(2,0,3,1)
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,1,2,0)

    doo = doo + doo.T
    dvv = dvv + dvv.T
    for i in range(nocc):
        dm2[i,i,:nocc,:nocc] += doo * 2
        dm2[:nocc,:nocc,i,i] += doo * 2
        dm2[i,i,nocc:,nocc:] += dvv * 2
        dm2[nocc:,nocc:,i,i] += dvv * 2
        dm2[:nocc,i,i,:nocc] -= doo
        dm2[i,:nocc,:nocc,i] -= doo
        dm2[nocc:,i,i,nocc:] -= dvv
        dm2[i,nocc:,nocc:,i] -= dvv

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2

    return dm2


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

    goo, gvv = gamma1_intermediates(mcc, t1, t2, l1, l2)
    print((numpy.einsum('ij,ij', goo, fock0[:nocc,:nocc]))*2+20166.3298610348)
    print((numpy.einsum('ab,ab', gvv, fock0[nocc:,nocc:]))*2-58078.9640192468)

    doovv, dvvvv, doooo, dovov, dovvo, dvovv, dooov = \
            gamma2_intermediates(mcc, t1, t2, l1, l2)

    print('doooo',numpy.einsum('klij,kilj', doooo, eris.oooo)*2-15939.9007625418)
    print('dvvvv',numpy.einsum('abcd,acbd', dvvvv, eris.vvvv)*2-37581.823919588 )
    print('dooov',numpy.einsum('jika,jkia', dooov, eris.ooov)*2-128470.009687716)
    print('dvovv',numpy.einsum('aibc,icab', dvovv, eris.ovvv)*2+166794.225195056)
    print('doovv',numpy.einsum('ijab,iajb', doovv, eris.ovov)*2+719279.812916893)
    print('dovvo',numpy.einsum('jabi,jbia', dovvo, eris.ovov)*2
                 +numpy.einsum('jaib,jiba', dovov, eris.oovv)*2+53634.0012286654)

    dm1 = make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = make_rdm2(mcc, t1, t2, l1, l2)
    e2 =(numpy.einsum('klij,kilj', doooo, eris.oooo)*2
        +numpy.einsum('abcd,acbd', dvvvv, eris.vvvv)*2
        +numpy.einsum('jika,jkia', dooov, eris.ooov)*2
        +numpy.einsum('aibc,icab', dvovv, eris.ovvv)*2
        +numpy.einsum('ijab,iajb', doovv, eris.ovov)*2
        +numpy.einsum('jabi,jbia', dovvo, eris.ovov)*2
        +numpy.einsum('jaib,jiba', dovov, eris.oovv)*2
        +numpy.einsum('ij,ij', goo, fock0[:nocc,:nocc])*2
        +numpy.einsum('ab,ab', gvv, fock0[nocc:,nocc:])*2
        +fock0[:nocc].trace()*2
        -numpy.einsum('kkpq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace()*2
        +numpy.einsum('pkkq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace())
    print(e2+719760.850761183)
    print(numpy.einsum('pqrs,pqrs', dm2, eri0)*.5 +
           numpy.einsum('pq,pq', dm1, h1) - e2)

    print(numpy.allclose(dm2, dm2.transpose(1,0,3,2)))
    print(numpy.allclose(dm2, dm2.transpose(2,3,0,1)))

    d1 = numpy.einsum('kkpq->pq', dm2) / 9
    print(numpy.allclose(d1, dm1))
