#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd

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
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    doo -= lib.einsum('jkab,ikab->ij', l2, theta)
    dvv += lib.einsum('jica,jicb->ab', l2, theta)
    xt1  = lib.einsum('mnef,inef->mi', l2, theta)
    xt2  = lib.einsum('mnaf,mnef->ea', l2, theta)
    dov += numpy.einsum('imae,me->ia', theta, l1)
    dov -= numpy.einsum('mi,ma->ia', xt1, t1)
    dov -= numpy.einsum('ie,ae->ia', t1, xt2)
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(mycc, t1, t2, l1, l2):
    f = lib.H5TmpFile()
    gamma2_outcore(mycc, t1, t2, l1, l2, f)
    d2 = (f['dvovo'].value.transpose(1,0,3,2),
          f['dvvvv'].value,
          f['doooo'].value,
          f['dvvoo'].value.transpose(2,3,0,1),
          f['dvoov'].value.transpose(2,3,0,1),
          f['dvvov'].value,
          f['dvovv'].value.transpose(1,0,2,3),
          f['dvooo'].value.transpose(3,2,1,0))
    return d2

def gamma2_outcore(mycc, t1, t2, l1, l2, h5fobj):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir * (nvir+1) //2
    dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), 'f8')
    dvvov = h5fobj.create_dataset('dvvov', (nvir,nvir,nocc,nvir), 'f8')
    dvoov = h5fobj.create_dataset('dvoov', (nvir,nocc,nocc,nvir), 'f8')
    dvvoo = h5fobj.create_dataset('dvvoo', (nvir,nvir,nocc,nocc), 'f8')
    fswap = lib.H5TmpFile()

    time1 = time.clock(), time.time()
    pvOOv = lib.einsum('ikca,jkcb->aijb', l2, t2)
    moo = numpy.einsum('dljd->jl', pvOOv) * 2
    mvv = numpy.einsum('blld->db', pvOOv) * 2
    gvooo = lib.einsum('kc,cija->aikj', t1, pvOOv)
    fswap['mvOOv'] = pvOOv
    pvOOv = None

    pvoOV = -lib.einsum('ikca,jkbc->aijb', l2, t2)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    pvoOV += lib.einsum('ikac,jkbc->aijb', l2, theta)
    moo += numpy.einsum('dljd->jl', pvoOV)
    mvv += numpy.einsum('blld->db', pvoOV)
    gvooo -= lib.einsum('jc,cika->aikj', t1, pvoOV)
    fswap['mvoOV'] = pvoOV
    pvoOV = None

    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2
        - numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1) + moo*.5

    tau = numpy.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    goooo = lib.einsum('ijab,klab->klij', l2, tau)*.5
    h5fobj['doooo'] = goooo.transpose(0,2,1,3)*2 - goooo.transpose(0,3,1,2)

    gvooo += numpy.einsum('ji,ka->aikj', -.5*moo, t1)
    gvooo += lib.einsum('la,jkil->aikj', 2*t1, goooo)
    gvooo -= lib.einsum('ib,jkba->aikj', l1, tau)
    gvooo -= lib.einsum('jkba,ib->aikj', l2, t1)
    h5fobj['dvooo'] = gvooo.transpose(0,2,1,3)*2 - gvooo.transpose(0,3,1,2)
    tau = gvooo = None
    time1 = log.timer_debug1('rdm intermediates pass1', *time1)

    gvvoo = numpy.einsum('ia,jb->abij', mia, t1)
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc**2*nvir*6
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    log.debug1('rdm intermediates pass 2: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    for p0, p1 in lib.prange(0, nvir, blksize):
        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        tau += t2[:,:,p0:p1]
        gvvoo[p0:p1] -= lib.einsum('jk,ikab->abij', mij, tau)
        gvvoo[p0:p1] += .5 * l2[:,:,p0:p1].transpose(2,3,0,1)
        gvvoo[p0:p1] += .5 * tau.transpose(2,3,0,1)
        gvvoo[p0:p1] -= lib.einsum('cb,ijac->abij', mab, t2[:,:,p0:p1])
        gvvoo[p0:p1] -= lib.einsum('bd,ijad->abij', mvv*.5, tau)
        gvvoo[p0:p1] += lib.einsum('ijkl,klab->abij', goooo, tau)

        pvOOv = _cp(fswap['mvOOv'][p0:p1])
        pvoOV = _cp(fswap['mvoOV'][p0:p1])
        gvOOv = lib.einsum('kiac,jc,kb->aijb', l2[:,:,p0:p1], t1, t1)
        gvOOv += pvOOv
        gvoOV = numpy.einsum('ia,jb->aijb', l1[:,p0:p1], t1)
        gvoOV -= lib.einsum('ikac,jc,kb->aijb', l2[:,:,p0:p1], t1, t1)
        gvoOV += pvoOV
        dvoov[p0:p1] = ( 2*gvoOV + gvOOv)
        dvvoo[p0:p1] = (-2*gvOOv - gvoOV).transpose(0,3,2,1)
        gvOOv = gvoOV = None

        tau -= t2[:,:,p0:p1] * .5
        for q0, q1 in lib.prange(0, nvir, blksize):
            gvvoo[q0:q1,:] += lib.einsum('dlib,jlda->abij', pvOOv, tau[:,:,:,q0:q1])
            gvvoo[:,q0:q1] -= lib.einsum('dlia,jldb->abij', pvoOV, tau[:,:,:,q0:q1])
            tmp = pvoOV[:,:,:,q0:q1] + pvOOv[:,:,:,q0:q1]*.5
            gvvoo[q0:q1,:] += lib.einsum('dlia,jlbd->abij', tmp, t2[:,:,:,p0:p1])
        pvOOv = pvoOV = tau = None
        time1 = log.timer_debug1('rdm intermediates pass2 [%d:%d]'%(p0, p1), *time1)
    h5fobj['dvovo'] = gvvoo.transpose(0,2,1,3) * 2 - gvvoo.transpose(0,3,1,2)
    gvvoo = goooo = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = max(nocc**2*nvir*2+nocc*nvir**2*3, nvir**3*2+nocc*nvir**2)
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    iobuflen = int(256e6/8/blksize)
    log.debug1('rdm intermediates pass 3: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    dvovv = h5fobj.create_dataset('dvovv', (nvir,nocc,nvir,nvir), 'f8',
                                  chunks=(nvir,nocc,blksize,nvir))
    time1 = time.clock(), time.time()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        l2tmp = l2[:,:,p0:p1] * .5
        gvvvv = lib.einsum('ijab,ijcd->abcd', l2tmp, t2)
        jabc = lib.einsum('ijab,ic->jabc', l2tmp, t1)
        gvvvv += lib.einsum('jabc,jd->abcd', jabc, t1)
        l2tmp = jabc = None

# symmetrize dvvvv because it is symmetrized in ccsd_grad and make_rdm2 anyway
#:dvvvv = .5*(gvvvv+gvvvv.transpose(0,1,3,2))
#:dvvvv = .5*(dvvvv+dvvvv.transpose(1,0,3,2))
# now dvvvv == dvvvv.transpose(2,3,0,1) == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
        tmp = numpy.empty((nvir,nvir,nvir))
        tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
        for i in range(p1-p0):
            tmp[:] = gvvvv[i].transpose(1,0,2)*2 - gvvvv[i].transpose(2,0,1)
            lib.pack_tril(tmp+tmp.transpose(0,2,1), out=tmpvvvv[i])
        # tril of (dvvvv[p0:p1,p0:p1]+dvvvv[p0:p1,p0:p1].T)
        for i in range(p0, p1):
            for j in range(p0, i):
                tmpvvvv[i-p0,j] += tmpvvvv[j-p0,i]
            tmpvvvv[i-p0,i] *= 2
        for i in range(p1, nvir):
            off = i * (i+1) // 2
            dvvvv[off+p0:off+p1] = tmpvvvv[:,i]
        for i in range(p0, p1):
            off = i * (i+1) // 2
            if p0 > 0:
                tmpvvvv[i-p0,:p0] += dvvvv[off:off+p0]
            dvvvv[off:off+i+1] = tmpvvvv[i-p0,:i+1] * .25
        tmp = tmpvvvv = None

        gvovv = lib.einsum('adbc,id->aibc', gvvvv, t1*-2)
        gvvvv = None

        gvovv += lib.einsum('akic,kb->aibc', _cp(fswap['mvoOV'][p0:p1]), t1)
        gvovv -= lib.einsum('akib,kc->aibc', _cp(fswap['mvOOv'][p0:p1]), t1)

        gvovv += lib.einsum('ja,jibc->aibc', l1[:,p0:p1], t2)
        gvovv += lib.einsum('jibc,ja->aibc', l2, t1[:,p0:p1])
        gvovv += lib.einsum('ja,jb,ic->aibc', l1[:,p0:p1], t1, t1)
        gvovv += numpy.einsum('ba,ic->aibc', mvv[:,p0:p1]*.5, t1)

        gvvov = gvovv.transpose(0,2,1,3)*2 - gvovv.transpose(0,3,1,2)
        dvovv[:,:,p0:p1] = gvvov.transpose(3,2,0,1)
        dvvov[p0:p1] = gvvov
        gvovv = gvvov = None
        time1 = log.timer_debug1('rdm intermediates pass3 [%d:%d]'%(p0, p1), *time1)

    fswap = None
    return (h5fobj['dvovo'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['dvvoo'],
            h5fobj['dvoov'], h5fobj['dvvov'], h5fobj['dvovv'], h5fobj['dvooo'])

def make_rdm1(mycc, t1, t2, l1, l2, d1=None):
    if d1 is None: d1 = gamma1_intermediates(mycc, t1, t2, l1, l2)
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
    if d1 is None: d1 = gamma1_intermediates(mycc, t1, t2, l1, l2)
    if d2 is None: d2 = gamma2_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
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

def _cp(a):
    return numpy.array(a, copy=False, order='C')


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

    fdm2 = lib.H5TmpFile()
    dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo = \
            gamma2_outcore(mcc, t1, t2, l1, l2, fdm2)
    print('dvovo', lib.finger(_cp(dvovo)) -  5371.8073743106979)
    print('dvvvv', lib.finger(_cp(dvvvv)) - -25.374007033024839)
    print('doooo', lib.finger(_cp(doooo)) -  60.114594698129963)
    print('dvvoo', lib.finger(_cp(dvvoo)) -   5.718307760949088)
    print('dvoov', lib.finger(_cp(dvoov)) -  41.399881265666437)
    print('dvvov', lib.finger(_cp(dvvov)) - -1129.9636175187422)
    print('dvovv', lib.finger(_cp(dvovv)) - -1038.3437534763445)
    print('dvooo', lib.finger(_cp(dvooo)) -  979.27741472637604)
    fdm2 = None

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            gamma2_intermediates(mcc, t1, t2, l1, l2)
    print('dovov', lib.finger(_cp(dovov)) - -14384.907042073517)
    print('dvvvv', lib.finger(_cp(dvvvv)) - -25.374007033024839)
    print('doooo', lib.finger(_cp(doooo)) -  60.114594698129963)
    print('doovv', lib.finger(_cp(doovv)) - -79.176348067958401)
    print('dovvo', lib.finger(_cp(dovvo)) -  60.596864321502196)
    print('dvvov', lib.finger(_cp(dvvov)) - -1129.9636175187422)
    print('dovvv', lib.finger(_cp(dovvv)) - -421.90333700061342)
    print('dooov', lib.finger(_cp(dooov)) - -592.66863759586136)

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
