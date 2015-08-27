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
from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda

# dE = (goo * foo + gvv * fvv + doooo*eri_oooo + ...) * 2
def gamma1_intermediates(cc, t1, t2, l1, l2, blksize=ccsd.BLKMIN):
    nocc, nvir = t1.shape
    goo = -numpy.einsum('ja,ia->ij', l1, t1)
    gvv = numpy.einsum('ia,ib->ab', l1, t1)
    #:goo -= numpy.einsum('jkab,ikab->ij', l2, theta)
    #:gvv += numpy.einsum('jica,jicb->ab', l2, theta)
    for p0, p1 in ccsd.prange(0, nocc, blksize):
        theta = ccsd_lambda.make_theta(t2[p0:p1])
        goo[p0:p1] -= lib.dot(theta.reshape(p1-p0,-1), l2.reshape(nocc,-1).T)
        gvv += lib.dot(l2[p0:p1].reshape(-1,nvir).T, theta.reshape(-1,nvir))
    return goo, gvv

# gamma2 intermediates in Physist's notation
def gamma2_intermediates(cc, t1, t2, l1, l2, blksize=ccsd.BLKMIN):
    tmpfile = tempfile.NamedTemporaryFile()
    gamma2_outcore(cc, t1, t2, l1, l2, tmpfile.name, blksize)
    with h5py.File(tmpfile.name, 'r') as f:
        return (f['doovv'].value, f['dvvvv'].value, f['doooo'].value,
                f['dovov'].value, f['dovvo'].value, f['dvovv'].value,
                f['dooov'].value)

def gamma2_outcore(cc, t1, t2, l1, l2, filename, blksize=ccsd.BLKMIN):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    fsave = h5py.File(filename, 'w')
    fsave.create_dataset('doovv', (nocc,nocc,nvir,nvir), 'f8')
    fsave.create_dataset('dvvvv', (nvir,nvir,nvir,nvir), 'f8')
    fsave.create_dataset('doooo', (nocc,nocc,nocc,nocc), 'f8')
    fsave.create_dataset('dovov', (nocc,nvir,nocc,nvir), 'f8')
    fsave.create_dataset('dovvo', (nocc,nvir,nvir,nocc), 'f8')
    fsave.create_dataset('dvovv', (nvir,nocc,nvir,nvir), 'f8')
    fsave.create_dataset('dooov', (nocc,nocc,nocc,nvir), 'f8')

    _tmpfile = tempfile.NamedTemporaryFile()
    fswap = h5py.File(_tmpfile.name)
    mOvOv = fswap.create_dataset('mOvOv', (nocc,nvir,nocc,nvir), 'f8')
    mOVov = fswap.create_dataset('mOVov', (nocc,nvir,nocc,nvir), 'f8')

    moo = numpy.empty((nocc,nocc))
    mvv = numpy.zeros((nvir,nvir))
    for istep, (p0, p1) in enumerate(ccsd.prange(0, nocc, blksize)):
        #:pOvOv = numpy.einsum('ikca,jkcb->ijba', l2, t2[p0:p1])
        #:pOVov = -numpy.einsum('ikca,jkbc->ijba', l2, t2[p0:p1])
        #:pOVov += numpy.einsum('ikac,jkbc->ijba', l2, theta)
        pOvOv = numpy.empty((nocc,(p1-p0)*nvir,nvir))
        pOVov = numpy.empty((nocc,(p1-p0)*nvir,nvir))
        t2a = ccsd._cp(t2[p0:p1].transpose(0,3,1,2).reshape(-1,nov))
        t2b = ccsd._cp(t2[p0:p1].transpose(0,2,1,3).reshape(-1,nov))
        theta = ccsd._cp(ccsd_lambda.make_theta(t2[p0:p1]).transpose(0,2,1,3).reshape(-1,nov))
        for j in range(nocc):
            pOvOv[j] = lib.dot(t2a, l2[j].reshape(nov,-1))
            pOVov[j] = lib.dot(t2b, l2[j].reshape(nov,-1), -1)
            lib.dot(theta, ccsd._cp(l2[j].transpose(0,2,1).reshape(nov,-1)), 1, pOVov[j], 1)
        pOvOv = pOvOv.reshape(nocc,p1-p0,nvir,nvir)
        pOVov = pOVov.reshape(nocc,p1-p0,nvir,nvir)
        theta = t2a = t2b = None
        mOvOv[p0:p1] = pOvOv.transpose(1,2,0,3)
        mOVov[p0:p1] = pOVov.transpose(1,2,0,3)
        fswap['mvOvO/%d'%istep] = pOvOv.transpose(3,1,2,0)
        fswap['mvOVo/%d'%istep] = pOVov.transpose(3,1,2,0)
        moo[p0:p1] =(numpy.einsum('ljdd->jl', pOvOv) * 2
                   + numpy.einsum('ljdd->jl', pOVov))
        mvv +=(numpy.einsum('llbd->bd', pOvOv[p0:p1]) * 2
             + numpy.einsum('llbd->bd', pOVov[p0:p1]))
        pOvOv = pOVov = None
    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2
        - numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1) + moo*.5

    gooov = numpy.zeros((nocc,nocc,nocc,nvir))
    for p0, p1 in ccsd.prange(0, nocc, blksize):
        tau = t2[p0:p1] + numpy.einsum('ia,jb->ijab', t1[p0:p1], t1)
        #:goooo = numpy.einsum('ijab,klab->klij', l2, tau)*.5
        goooo = lib.dot(tau.reshape(-1,nvir**2), l2.reshape(-1,nvir**2).T, .5)
        goooo = goooo.reshape(-1,nocc,nocc,nocc)
        fsave['doooo'][p0:p1] = ccsd_lambda.make_theta(goooo)

        #:gooov[p0:p1] -= numpy.einsum('ib,jkba->jkia', l1, tau)
        #:gooov[p0:p1] -= numpy.einsum('jkba,ib->jkia', l2[p0:p1], t1)
        #:gooov[p0:p1] += numpy.einsum('jkil,la->jkia', goooo, t1*2)
        gooov[p0:p1] -= lib.dot(ccsd._cp(tau.transpose(0,1,3,2).reshape(-1,nvir)),
                                l1.T).reshape(-1,nocc,nvir,nocc).transpose(0,1,3,2)
        gooov[p0:p1] -= lib.dot(ccsd._cp(l2[p0:p1].transpose(0,1,3,2).reshape(-1,nvir)),
                                t1.T).reshape(-1,nocc,nvir,nocc).transpose(0,1,3,2)
        lib.dot(goooo.reshape(-1,nocc), t1, 2, gooov[p0:p1].reshape(-1,nvir), 1)
        gooov[p0:p1] += numpy.einsum('ji,ka->jkia', moo[p0:p1]*-.5, t1)

        goovv = .5 * l2[p0:p1] + .5 * tau
        goovv += numpy.einsum('ia,jb->ijab', mia[p0:p1], t1)
        #:goovv -= numpy.einsum('jk,ikab->ijab', mij, tau)
        for j in range(p1-p0):
            lib.dot(mij, tau[j].reshape(nocc,-1), -1, goovv[j].reshape(nocc,-1), 1)
        #:goovv -= numpy.einsum('cb,ijac->ijab', mab, t2[p0:p1])
        #:goovv -= numpy.einsum('bd,ijad->ijab', mvv*.5, tau)
        lib.dot(t2[p0:p1].reshape(-1,nvir), mab, -1, goovv.reshape(-1,nvir), 1)
        lib.dot(tau.reshape(-1,nvir), mvv.T, -.5, goovv.reshape(-1,nvir), 1)
        tau = None

        pOvOv = ccsd._cp(mOvOv[p0:p1])
        pOVov = ccsd._cp(mOVov[p0:p1])
        #:gooov[p0:p1,:] += numpy.einsum('jaic,kc->jkia', pOvOv, t1)
        #:gooov[:,p0:p1] -= numpy.einsum('kaic,jc->jkia', pOVov, t1)
        tmp = lib.dot(t1, pOvOv.reshape(-1,nvir).T).reshape(nocc,-1,nvir,nocc)
        gooov[p0:p1,:] += tmp.transpose(1,0,3,2)
        lib.dot(t1, pOVov.reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1), 0)
        gooov[:,p0:p1] -= tmp.transpose(0,1,3,2)
        gOvVo = numpy.einsum('ia,jb->jabi', l1, t1[p0:p1])
        gOvVo += pOVov.transpose(0,3,1,2)
        #:tmp = numpy.einsum('ikac,jc->ikaj', l2, t1[p0:p1])
        #:gOvVo -= numpy.einsum('ikaj,kb->jabi', tmp, t1)
        #:gOvvO = numpy.einsum('kiaj,kb->jabi', tmp, t1)
        tmp = lib.dot(l2.reshape(-1,nvir), t1[p0:p1].T).reshape(nocc,nocc,nvir,-1)
        gOvVo -= lib.dot(ccsd._cp(tmp.transpose(3,2,0,1).reshape(-1,nocc)),
                         t1).reshape(-1,nvir,nocc,nvir).transpose(0,1,3,2)
        gOvvO = lib.dot(t1.T, tmp.reshape(nocc,-1)).reshape(nvir,nocc,nvir,-1)
        gOvvO = gOvvO.transpose(3,2,0,1) + pOvOv.transpose(0,3,1,2)
        fsave['dovvo'][p0:p1] = gOvVo*2 + gOvvO
        gOvvO *= -2
        gOvvO -= gOvVo
        fsave['dovov'][p0:p1] = gOvvO.transpose(0,1,3,2)
        tmp = gOvvO = gOvVo = None

        for j0, j1 in ccsd.prange(0, nocc, blksize):
            tau2 = t2[j0:j1] + numpy.einsum('ia,jb->ijab', t1[j0:j1], t1)
            #:goovv += numpy.einsum('ijkl,klab->ijab', goooo[:,:,j0:j1], tau2)
            lib.dot(goooo[:,:,j0:j1].copy().reshape((p1-p0)*nocc,-1),
                    tau2.reshape(-1,nvir**2), 1, goovv.reshape(-1,nvir**2), 1)
            tau2 += numpy.einsum('ia,jb->ijab', t1[j0:j1], t1)
            tau2 = ccsd._cp(tau2.transpose(0,3,1,2).reshape(-1,nov))
            #:goovv += numpy.einsum('ibld,ljad->ijab', pOvOv[:,:,j0:j1], tau2) * .5
            #:goovv -= numpy.einsum('iald,ljbd->ijab', pOVov[:,:,j0:j1], tau2) * .5
            goovv += lib.dot(pOvOv[:,:,j0:j1].copy().reshape((p1-p0)*nvir,-1),
                             tau2, .5).reshape(-1,nvir,nocc,nvir).transpose(0,2,3,1)
            goovv += lib.dot(pOVov[:,:,j0:j1].copy().reshape((p1-p0)*nvir,-1),
                             tau2, -.5).reshape(-1,nvir,nocc,nvir).transpose(0,2,1,3)
            tau2 = None
        #:goovv += numpy.einsum('iald,jlbd->ijab', pOVov*2+pOvOv, t2) * .5
        pOVov *= 2
        pOVov += pOvOv
        for j in range(nocc):
            tmp = lib.dot(pOVov.reshape(-1,nov),
                          ccsd._cp(t2[j].transpose(0,2,1).reshape(-1,nvir)), .5)
            goovv[:,j] += tmp.reshape(-1,nvir,nvir)
            tmp = None
        fsave['doovv'][p0:p1] = ccsd_lambda.make_theta(goovv)
        goooo = goovv = pOvOv = pOVov = None

    fsave['dooov'][:] = gooov*2 - gooov.transpose(1,0,2,3)
    gooov = None

    for p0, p1 in ccsd.prange(0, nvir, blksize):
        l2tmp = l2[:,:,p0:p1] * .5
        #:gvvvv = numpy.einsum('ijab,ijcd->abcd', l2tmp, t2)
        #:jabc = numpy.einsum('ijab,ic->jabc', l2tmp, t1)
        #:gvvvv += numpy.einsum('jabc,jd->abcd', jabc, t1)
        gvvvv = lib.dot(l2tmp.reshape(nocc**2,-1).T, t2.reshape(nocc**2,-1))
        jabc = lib.dot(l2tmp.reshape(nocc,-1).T, t1)
        lib.dot(jabc.reshape(nocc,-1).T, t1, 1, gvvvv.reshape(-1,nvir), 1)
        gvvvv = gvvvv.reshape(-1,nvir,nvir,nvir)
        l2tmp = jabc = None
        fsave['dvvvv'][p0:p1] = ccsd_lambda.make_theta(gvvvv)

        #:gvovv = numpy.einsum('ja,jibc->aibc', l1[:,p0:p1], t2)
        #:gvovv += numpy.einsum('jibc,ja->aibc', l2, t1[:,p0:p1])
        gvovv = lib.dot(l1[:,p0:p1].copy().T, t2.reshape(nocc,-1))
        lib.dot(t1[:,p0:p1].copy().T, l2.reshape(nocc,-1), 1, gvovv, 1)
        gvovv = gvovv.reshape(-1,nocc,nvir,nvir)
        tmp = numpy.einsum('ja,jb->ab', l1[:,p0:p1], t1)
        gvovv += numpy.einsum('ab,ic->aibc', tmp, t1)
        gvovv += numpy.einsum('ba,ic->aibc', mvv[:,p0:p1]*.5, t1)
        #:gvovv -= numpy.einsum('adbc,id->aibc', gvvvv, t1*2)
        for j in range(p1-p0):
            lib.dot(t1, gvvvv[j].reshape(nvir,-1), -2,
                    gvovv[j].reshape(nocc,-1), 1)

        pvOvO = numpy.empty((p1-p0,nocc,nvir,nocc))
        pvOVo = numpy.empty((p1-p0,nocc,nvir,nocc))
        for istep, (j0, j1) in enumerate(ccsd.prange(0, nocc, blksize)):
            pvOvO[:,j0:j1] = fswap['mvOvO/%d'%istep][p0:p1]
            pvOVo[:,j0:j1] = fswap['mvOVo/%d'%istep][p0:p1]
        gvovv -= numpy.einsum('aibk,kc->aibc', pvOvO, t1)
        gvovv += numpy.einsum('aick,kb->aibc', pvOVo, t1)
        fsave['dvovv'][p0:p1] = ccsd_lambda.make_theta(gvovv)
        gvvvv = gvovv = pvOvO = pvOVo = None

    fswap.close()
    _tmpfile = None
    fsave.close()

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
