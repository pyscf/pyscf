#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd
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
    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2
    blksize = max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit))
    for p0, p1 in prange(0, nocc, blksize):
        theta = make_theta(t2[p0:p1])
        doo[p0:p1] -= lib.dot(theta.reshape(p1-p0,-1), l2.reshape(nocc,-1).T)
        dov[p0:p1] += numpy.einsum('imae,me->ia', theta, l1)
        xt1 = lib.dot(l2.reshape(nocc,-1), theta.reshape(p1-p0,-1).T)
        dov[p0:p1] -= numpy.einsum('mi,ma->ia', xt1, t1)
        xt2 = lib.dot(theta.reshape(-1,nvir).T, l2[p0:p1].reshape(-1,nvir))
        dov -= numpy.einsum('ie,ae->ia', t1, xt2)
        dvv += lib.dot(l2[p0:p1].reshape(-1,nvir).T, theta.reshape(-1,nvir))
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(mycc, t1, t2, l1, l2):
    tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    with h5py.File(tmpfile.name, 'w') as f:
        gamma2_outcore(mycc, t1, t2, l1, l2, f)
        nocc, nvir = f['dovov'].shape[:2]
        nov = nocc * nvir
        dovvv = numpy.empty((nocc,nvir,nvir,nvir))
        ao2mo.outcore._load_from_h5g(f['dovvv'], 0, nov, dovvv.reshape(nov,-1))
        dvvov = None
        d2 = (f['dovov'].value, f['dvvvv'].value, f['doooo'].value,
              f['doovv'].value, f['dovvo'].value, dvvov, dovvv,
              f['dooov'].value)
        for key in f.keys():
            del(f[key])
        return d2

def gamma2_outcore(mycc, t1, t2, l1, l2, h5fobj):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir * (nvir+1) //2
    dovov = h5fobj.create_dataset('dovov', (nocc,nvir,nocc,nvir), 'f8')
    dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), 'f8')
    doooo = h5fobj.create_dataset('doooo', (nocc,nocc,nocc,nocc), 'f8')
    doovv = h5fobj.create_dataset('doovv', (nocc,nocc,nvir,nvir), 'f8')
    dovvo = h5fobj.create_dataset('dovvo', (nocc,nvir,nvir,nocc), 'f8')
    dooov = h5fobj.create_dataset('dooov', (nocc,nocc,nocc,nvir), 'f8')

    _tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = h5py.File(_tmpfile.name)
    mOvOv = fswap.create_dataset('mOvOv', (nocc,nvir,nocc,nvir), 'f8')
    mOVov = fswap.create_dataset('mOVov', (nocc,nvir,nocc,nvir), 'f8')

    moo = numpy.empty((nocc,nocc))
    mvv = numpy.zeros((nvir,nvir))

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2 * 5
    blksize = max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit))
    log.debug1('rdm intermediates pass 1: block size = %d, nocc = %d in %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))
    time1 = time.clock(), time.time()
    for istep, (p0, p1) in enumerate(prange(0, nocc, blksize)):
        #:theta = make_theta(t2[p0:p1])
        #:pOvOv = numpy.einsum('ikca,jkcb->jbia', l2, t2[p0:p1])
        #:pOVov = -numpy.einsum('ikca,jkbc->jbia', l2, t2[p0:p1])
        #:pOVov += numpy.einsum('ikac,jkbc->jbia', l2, theta)
        pOvOv = numpy.empty((nocc,p1-p0,nvir,nvir))
        pOVov = numpy.empty((nocc,p1-p0,nvir,nvir))
        t2a = numpy.empty((p1-p0,nvir,nocc,nvir))
        t2b = numpy.empty((p1-p0,nvir,nocc,nvir))
        theta = make_theta(t2[p0:p1])
        tmp = numpy.empty_like(t2a)
        for i in range(p1-p0):
            t2a[i] = t2[p0+i].transpose(2,0,1)
            t2b[i] = t2[p0+i].transpose(1,0,2)
            tmp[i] = theta[i].transpose(1,0,2)
        t2a = t2a.reshape(-1,nov)
        t2b = t2b.reshape(-1,nov)
        theta, tmp = tmp.reshape(-1,nov), None
        for i in range(nocc):
            pOvOv[i] = lib.dot(t2a, l2[i].reshape(nov,-1)).reshape(-1,nvir,nvir)
            pOVov[i] = lib.dot(t2b, l2[i].reshape(nov,-1), -1).reshape(-1,nvir,nvir)
            pOVov[i] += lib.dot(theta, _cp(l2[i].transpose(0,2,1).reshape(nov,-1))).reshape(-1,nvir,nvir)
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
        time1 = log.timer_debug1('rdm intermediates pass1 [%d:%d]'%(p0, p1), *time1)
    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2
        - numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1) + moo*.5

    gooov = numpy.einsum('ji,ka->jkia', moo*-.5, t1)
    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc**3 + nocc**2*nvir + nocc*nvir**2*6
    blksize = max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit))
    log.debug1('rdm intermediates pass 2: block size = %d, nocc = %d in %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))
    for p0, p1 in prange(0, nocc, blksize):
        tau = _ccsd.make_tau(t2[p0:p1], t1[p0:p1], t1)
        #:goooo = numpy.einsum('ijab,klab->klij', l2, tau)*.5
        goooo = lib.dot(tau.reshape(-1,nvir**2), l2.reshape(-1,nvir**2).T, .5)
        goooo = goooo.reshape(-1,nocc,nocc,nocc)
        h5fobj['doooo'][p0:p1] = make_theta(goooo).transpose(0,2,1,3)

        #:gooov[p0:p1] -= numpy.einsum('ib,jkba->jkia', l1, tau)
        #:gooov[p0:p1] -= numpy.einsum('jkba,ib->jkia', l2[p0:p1], t1)
        #:gooov[p0:p1] += numpy.einsum('jkil,la->jkia', goooo, t1*2)
        for i in range(p0,p1):
            gooov[i] -= lib.dot(_cp(tau[i-p0].transpose(0,2,1).reshape(-1,nvir)),
                                l1.T).reshape(nocc,nvir,nocc).transpose(0,2,1)
            gooov[i] -= lib.dot(_cp(l2[i].transpose(0,2,1).reshape(-1,nvir)),
                                t1.T).reshape(nocc,nvir,nocc).transpose(0,2,1)
        lib.dot(goooo.reshape(-1,nocc), t1, 2, gooov[p0:p1].reshape(-1,nvir), 1)

        #:goovv -= numpy.einsum('jk,ikab->ijab', mij, tau)
        goovv = numpy.einsum('ia,jb->ijab', mia[p0:p1], t1)
        for i in range(p1-p0):
            lib.dot(mij, tau[i].reshape(nocc,-1), -1, goovv[i].reshape(nocc,-1), 1)
            goovv[i] += .5 * l2[p0+i]
            goovv[i] += .5 * tau[i]
        #:goovv -= numpy.einsum('cb,ijac->ijab', mab, t2[p0:p1])
        #:goovv -= numpy.einsum('bd,ijad->ijab', mvv*.5, tau)
        lib.dot(t2[p0:p1].reshape(-1,nvir), mab, -1, goovv.reshape(-1,nvir), 1)
        lib.dot(tau.reshape(-1,nvir), mvv.T, -.5, goovv.reshape(-1,nvir), 1)
        tau = None
#==== mem usage nocc**3 + nocc*nvir**2

        pOvOv = _cp(mOvOv[p0:p1])
        pOVov = _cp(mOVov[p0:p1])
        #:gooov[p0:p1,:] += numpy.einsum('jaic,kc->jkia', pOvOv, t1)
        #:gooov[:,p0:p1] -= numpy.einsum('kaic,jc->jkia', pOVov, t1)
        tmp = lib.dot(pOvOv.reshape(-1,nvir), t1.T).reshape(p1-p0,-1,nocc,nocc)
        gooov[p0:p1,:] += tmp.transpose(0,3,2,1)
        lib.dot(t1, pOVov.reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1), 0)
        gooov[:,p0:p1] -= tmp.reshape(nocc,p1-p0,nvir,nocc).transpose(0,1,3,2)
        #:tmp = numpy.einsum('ikac,jc->jika', l2, t1[p0:p1])
        #:gOvVo -= numpy.einsum('jika,kb->jabi', tmp, t1)
        #:gOvvO = numpy.einsum('jkia,kb->jabi', tmp, t1) + pOvOv.transpose(0,3,1,2)
        tmp = tmp.reshape(-1,nocc,nocc,nvir)
        lib.dot(t1[p0:p1], l2.reshape(-1,nvir).T, 1, tmp.reshape(p1-p0,-1))
        gOvVo = numpy.einsum('ia,jb->jabi', l1, t1[p0:p1])
        gOvvO = numpy.empty((p1-p0,nvir,nvir,nocc))
        for i in range(p1-p0):
            gOvVo[i] -= lib.dot(_cp(tmp[i].transpose(0,2,1).reshape(-1,nocc)),
                                t1).reshape(nocc,nvir,-1).transpose(1,2,0)
            gOvVo[i] += pOVov[i].transpose(2,0,1)
            gOvvO[i] = lib.dot(tmp[i].reshape(nocc,-1).T,
                               t1).reshape(nocc,nvir,-1).transpose(1,2,0)
            gOvvO[i] += pOvOv[i].transpose(2,0,1)
        tmp = None
#==== mem usage nocc**3 + nocc*nvir**6
        dovvo[p0:p1] = (gOvVo*2 + gOvvO).transpose(0,2,1,3)
        gOvvO *= -2
        gOvvO -= gOvVo
        doovv[p0:p1] = gOvvO.transpose(0,3,1,2)
        gOvvO = gOvVo = None

        for j0, j1 in prange(0, nocc, blksize):
            tau2 = _ccsd.make_tau(t2[j0:j1], t1[j0:j1], t1)
            #:goovv += numpy.einsum('ijkl,klab->ijab', goooo[:,:,j0:j1], tau2)
            lib.dot(goooo[:,:,j0:j1].copy().reshape((p1-p0)*nocc,-1),
                    tau2.reshape(-1,nvir**2), 1, goovv.reshape(-1,nvir**2), 1)
            tau2 += numpy.einsum('ia,jb->ijab', t1[j0:j1], t1)
            tau2 = _cp(tau2.transpose(0,3,1,2).reshape(-1,nov))
            #:goovv[:,j0:j1] += numpy.einsum('ibld,jlda->ijab', pOvOv, tau2) * .5
            #:goovv[:,j0:j1] -= numpy.einsum('iald,jldb->ijab', pOVov, tau2) * .5
            goovv[:,j0:j1] += lib.dot(pOvOv.reshape(-1,nov), tau2.T,
                                      .5).reshape(p1-p0,nvir,-1,nvir).transpose(0,2,3,1)
            goovv[:,j0:j1] += lib.dot(pOVov.reshape(-1,nov), tau2.T,
                                      -.5).reshape(p1-p0,nvir,-1,nvir).transpose(0,2,1,3)
            tau2 = None
#==== mem usage nocc**3 + nocc*nvir**2*7
        #:goovv += numpy.einsum('iald,jlbd->ijab', pOVov*2+pOvOv, t2) * .5
        pOVov *= 2
        pOVov += pOvOv
        for j in range(nocc):
            tmp = lib.dot(pOVov.reshape(-1,nov),
                          _cp(t2[j].transpose(0,2,1).reshape(-1,nvir)), .5)
            goovv[:,j] += tmp.reshape(-1,nvir,nvir)
            tmp = None
        dovov[p0:p1] = make_theta(goovv).transpose(0,2,1,3)
        goooo = goovv = pOvOv = pOVov = None
        time1 = log.timer_debug1('rdm intermediates pass2 [%d:%d]'%(p0, p1), *time1)

    h5fobj['dooov'][:] = gooov.transpose(0,2,1,3)*2 - gooov.transpose(1,2,0,3)
    gooov = None

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = max(nocc**2*nvir*2+nocc*nvir**2*2, nvir**3*2+nocc*nvir**2)
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    iobuflen = int(256e6/8/blksize)
    log.debug1('rdm intermediates pass 3: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    h5fobj.create_group('dovvv')
    for istep, (p0, p1) in enumerate(prange(0, nvir, blksize)):
        pvOvO = numpy.empty((p1-p0,nocc,nvir,nocc))
        pvOVo = numpy.empty((p1-p0,nocc,nvir,nocc))
        ao2mo.outcore._load_from_h5g(fswap['mvOvO'], p0, p1, pvOvO)
        ao2mo.outcore._load_from_h5g(fswap['mvOVo'], p0, p1, pvOVo)
        #:gvovv -= numpy.einsum('aibk,kc->aibc', pvOvO, t1)
        #:gvovv += numpy.einsum('aick,kb->aibc', pvOVo, t1)
        gvovv = lib.dot(pvOVo.reshape(-1,nocc), t1).reshape(-1,nocc,nvir,nvir)
        for i in range(p1-p0):
            gvovv[i] = gvovv[i].transpose(0,2,1)
        lib.dot(pvOvO.reshape(-1,nocc), t1, -1, gvovv.reshape(-1,nvir), 1)
        pvOvO = pvOVo = None
#==== mem usage nocc**2*nvir*2 + nocc*nvir**2*2

        l2tmp = l2[:,:,p0:p1] * .5
        #:gvvvv = numpy.einsum('ijab,ijcd->abcd', l2tmp, t2)
        #:jabc = numpy.einsum('ijab,ic->jabc', l2tmp, t1)
        #:gvvvv += numpy.einsum('jabc,jd->abcd', jabc, t1)
        gvvvv = lib.dot(l2tmp.reshape(nocc**2,-1).T, t2.reshape(nocc**2,-1))
        jabc = lib.dot(l2tmp.reshape(nocc,-1).T, t1)
        lib.dot(jabc.reshape(nocc,-1).T, t1, 1, gvvvv.reshape(-1,nvir), 1)
        gvvvv = gvvvv.reshape(-1,nvir,nvir,nvir)
        l2tmp = jabc = None

        #:gvovv = numpy.einsum('ja,jibc->aibc', l1[:,p0:p1], t2)
        #:gvovv += numpy.einsum('jibc,ja->aibc', l2, t1[:,p0:p1])
        lib.dot(l1[:,p0:p1].copy().T, t2.reshape(nocc,-1), 1, gvovv.reshape(p1-p0,-1), 1)
        lib.dot(t1[:,p0:p1].copy().T, l2.reshape(nocc,-1), 1, gvovv.reshape(p1-p0,-1), 1)
        tmp = numpy.einsum('ja,jb->ab', l1[:,p0:p1], t1)
        gvovv += numpy.einsum('ab,ic->aibc', tmp, t1)
        gvovv += numpy.einsum('ba,ic->aibc', mvv[:,p0:p1]*.5, t1)
        #:gvovv -= numpy.einsum('adbc,id->aibc', gvvvv, t1*2)
        for j in range(p1-p0):
            lib.dot(t1, gvvvv[j].reshape(nvir,-1), -2,
                    gvovv[j].reshape(nocc,-1), 1)

# symmetrize dvvvv because it is symmetrized in ccsd_grad and make_rdm2 anyway
#:dvvvv = .5*(gvvvv+gvvvv.transpose(0,1,3,2))
#:dvvvv = .5*(dvvvv+dvvvv.transpose(1,0,3,2))
# now dvvvv == dvvvv.transpose(2,3,0,1) == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
        tmp = numpy.empty((nvir,nvir,nvir))
        tmp1 = numpy.empty((nvir,nvir,nvir))
        tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
        for i in range(p1-p0):
            make_theta(gvvvv[i:i+1], out=tmp)
            tmp1[:] = tmp.transpose(1,0,2)
            _ccsd.precontract(tmp1, diag_fac=2, out=tmpvvvv[i])
        # tril of (dvvvv[p0:p1,p0:p1]+dvvvv[p0:p1,p0:p1].T)
        for i in range(p0, p1):
            for j in range(p0, i):
                tmpvvvv[i-p0,j] += tmpvvvv[j-p0,i]
            tmpvvvv[i-p0,i] *= 2
        for i in range(p0, p1):
            off = i * (i+1) // 2
            if p0 > 0:
                tmpvvvv[i-p0,:p0] += dvvvv[off:off+p0]
            dvvvv[off:off+i+1] = tmpvvvv[i-p0,:i+1] * .25
        for i in range(p1, nvir):
            off = i * (i+1) // 2
            dvvvv[off+p0:off+p1] = tmpvvvv[:,i]
        tmp = tmp1 = tmpvvvv = None
#==== mem usage nvir**3 + nocc*nvir**2
        gvvov = make_theta(gvovv).transpose(0,2,1,3)
        ao2mo.outcore._transpose_to_h5g(h5fobj, 'dovvv/%d'%istep,
                                        gvvov.reshape(-1,nov), iobuflen)
        gvvvv = None
        gvovv = None
        time1 = log.timer_debug1('rdm intermediates pass3 [%d:%d]'%(p0, p1), *time1)

    del(fswap['mOvOv'])
    del(fswap['mOVov'])
    del(fswap['mvOvO'])
    del(fswap['mvOVo'])
    fswap.close()
    _tmpfile = None
    return (h5fobj['dovov'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['doovv'],
            h5fobj['dovvo'], None, h5fobj['dovvv'], h5fobj['dooov'])

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

    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0)
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

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _cp(a):
    return numpy.array(a, copy=False, order='C')

def make_theta(t2, out=None):
    return _ccsd.make_0132(t2, t2, 2, -1, out)


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
