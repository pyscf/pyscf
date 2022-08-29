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
Restricted CCSD implementation for real integrals.  Permutation symmetry for
the 4-index integrals (ij|kl) = (ij|lk) = (ji|kl) are assumed.

Note MO integrals are treated in chemist's notation
'''


from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd

# t2,l2 as ijab
def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO,
           fintermediates=None, fupdate=None):
    if eris is None: eris = mycc.ao2mo()
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        normt = numpy.linalg.norm(mycc.amplitudes_to_vector(l1new, l2new) -
                                  mycc.amplitudes_to_vector(l1, l2))
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
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
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    #fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    class _IMDS: pass
    imds = _IMDS()
    #TODO: mycc.incore_complete
    imds.ftmp = lib.H5TmpFile()
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    imds.wvooo = imds.ftmp.create_dataset('wvooo', (nvir,nocc,nocc,nocc), 'f8')
    imds.wVOov = imds.ftmp.create_dataset('wVOov', (nvir,nocc,nocc,nvir), 'f8')
    imds.wvOOv = imds.ftmp.create_dataset('wvOOv', (nvir,nocc,nocc,nvir), 'f8')
    imds.wvvov = imds.ftmp.create_dataset('wvvov', (nvir,nvir,nocc,nvir), 'f8')

    w1 = fvv - numpy.einsum('ja,jb->ba', fov, t1)
    w2 = foo + numpy.einsum('ib,jb->ij', fov, t1)
    w3 = numpy.einsum('kc,jkbc->bj', fov, t2) * 2 + fov.T
    w3 -= numpy.einsum('kc,kjbc->bj', fov, t2)
    w3 += lib.einsum('kc,kb,jc->bj', fov, t1, t1)
    w4 = fov.copy()

    unit = nocc*nvir**2*6
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(ccsd.BLKMIN, int((max_memory*.95e6/8-nocc**4-nvir*nocc**3)/unit)))
    log.debug1('ccsd lambda make_intermediates: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)//blksize))

    fswap = lib.H5TmpFile()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        fswap['vvov/%d'%istep] = eris_ovvv.transpose(2,3,0,1)

    woooo = 0
    wvooo = numpy.zeros((nvir,nocc,nocc,nocc))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        eris_vvov = numpy.empty(((p1-p0),nvir,nocc,nvir))
        for istep, (q0, q1) in enumerate(lib.prange(0, nvir, blksize)):
            eris_vvov[:,:,:,q0:q1] = fswap['vvov/%d'%istep][p0:p1]

        w1 += numpy.einsum('jcba,jc->ba', eris_ovvv, t1[:,p0:p1]*2)
        w1[:,p0:p1] -= numpy.einsum('jabc,jc->ba', eris_ovvv, t1)
        theta = t2[:,:,:,p0:p1] * 2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        w3 += lib.einsum('jkcd,kdcb->bj', theta, eris_ovvv)
        theta = None
        wVOov = lib.einsum('jbcd,kd->bjkc', eris_ovvv, t1)
        wvOOv = lib.einsum('cbjd,kd->cjkb', eris_vvov,-t1)
        g2vovv = eris_vvov.transpose(0,2,1,3) * 2 - eris_vvov.transpose(0,2,3,1)
        for i0, i1 in lib.prange(0, nocc, blksize):
            tau = t2[:,i0:i1] + numpy.einsum('ia,jb->ijab', t1, t1[i0:i1])
            wvooo[p0:p1,i0:i1] += lib.einsum('cibd,jkbd->ckij', g2vovv, tau)
        g2vovv = tau = None

        # Watch out memory usage here, due to the t2 transpose
        wvvov  = lib.einsum('jabd,jkcd->abkc', eris_ovvv, t2) * -1.5
        wvvov += eris_vvov.transpose(0,3,2,1) * 2
        wvvov -= eris_vvov

        g2vvov = eris_vvov * 2 - eris_ovvv.transpose(1,2,0,3)
        for i0, i1 in lib.prange(0, nocc, blksize):
            theta = t2[i0:i1] * 2 - t2[i0:i1].transpose(0,1,3,2)
            vackb = lib.einsum('acjd,kjbd->ackb', g2vvov, theta)
            wvvov[:,:,i0:i1] += vackb.transpose(0,3,2,1)
            wvvov[:,:,i0:i1] -= vackb * .5
        g2vvov = eris_ovvv = eris_vvov = theta = None

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        w2 += numpy.einsum('kbij,kb->ij', eris_ovoo, t1[:,p0:p1]) * 2
        w2 -= numpy.einsum('ibkj,kb->ij', eris_ovoo, t1[:,p0:p1])
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        w3 -= lib.einsum('lckj,klcb->bj', eris_ovoo, theta)

        tmp = lib.einsum('lc,jcik->ijkl', t1[:,p0:p1], eris_ovoo)
        woooo += tmp
        woooo += tmp.transpose(1,0,3,2)
        theta = tmp = None

        wvOOv += lib.einsum('lbjk,lc->bjkc', eris_ovoo, t1)
        wVOov -= lib.einsum('jbkl,lc->bjkc', eris_ovoo, t1)
        wvooo[p0:p1] += eris_ovoo.transpose(1,3,2,0) * 2
        wvooo[p0:p1] -= eris_ovoo.transpose(1,0,2,3)
        wvooo -= lib.einsum('klbc,iblj->ckij', t2[:,:,p0:p1], eris_ovoo*1.5)

        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2,1,0,3)
        theta = t2[:,:,:,p0:p1]*2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        vcjik = lib.einsum('jlcb,lbki->cjki', theta, g2ovoo)
        wvooo += vcjik.transpose(0,3,2,1)
        wvooo -= vcjik*.5
        theta = g2ovoo = None

        eris_voov = _cp(eris.ovvo[:,p0:p1]).transpose(1,0,3,2)
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo += lib.einsum('cijd,klcd->ijkl', eris_voov, tau)
        tau = None

        g2voov = eris_voov*2 - eris_voov.transpose(0,2,1,3)
        tmpw4 = numpy.einsum('ckld,ld->kc', g2voov, t1)
        w1 -= lib.einsum('ckja,kjcb->ba', g2voov, t2[:,:,p0:p1])
        w1[:,p0:p1] -= numpy.einsum('ja,jb->ba', tmpw4, t1)
        w2 += lib.einsum('jkbc,bikc->ij', t2[:,:,p0:p1], g2voov)
        w2 += numpy.einsum('ib,jb->ij', tmpw4, t1[:,p0:p1])
        w3 += reduce(numpy.dot, (t1.T, tmpw4, t1[:,p0:p1].T))
        w4[:,p0:p1] += tmpw4

        wvOOv += lib.einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
        wVOov -= lib.einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)

        VOov  = lib.einsum('bjld,klcd->bjkc', g2voov, t2)
        VOov -= lib.einsum('bjld,kldc->bjkc', eris_voov, t2)
        VOov += eris_voov
        vOOv = lib.einsum('bljd,kldc->bjkc', eris_voov, t2)
        vOOv -= _cp(eris.oovv[:,:,p0:p1]).transpose(2,1,0,3)
        wVOov += VOov
        wvOOv += vOOv
        imds.wVOov[p0:p1] = wVOov
        imds.wvOOv[p0:p1] = wvOOv

        ov1 = vOOv*2 + VOov
        ov2 = VOov*2 + vOOv
        vOOv = VOov = None
        wvooo -= lib.einsum('jb,bikc->ckij', t1[:,p0:p1], ov1)
        wvooo += lib.einsum('kb,bijc->ckij', t1[:,p0:p1], ov2)
        w3 += numpy.einsum('ckjb,kc->bj', ov2, t1[:,p0:p1])

        wvvov += lib.einsum('ajkc,jb->abkc', ov1, t1)
        wvvov -= lib.einsum('ajkb,jc->abkc', ov2, t1)

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2,1,0,3)
        tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        wvvov += lib.einsum('laki,klbc->abic', g2ovoo, tau)
        imds.wvvov[p0:p1] = wvvov
        wvvov = ov1 = ov2 = g2ovoo = None

    woooo += _cp(eris.oooo).transpose(0,2,1,3)
    imds.woooo[:] = woooo
    imds.wvooo[:] = wvooo
    woooo = wvooo = None

    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)

    fswap = None

    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    theta = t2*2 - t2.transpose(0,1,3,2)
    mba = lib.einsum('klca,klcb->ba', l2, theta)
    mij = lib.einsum('ikcd,jkcd->ij', l2, theta)
    theta = None
    mba1 = numpy.einsum('jc,jb->bc', l1, t1) + mba
    mij1 = numpy.einsum('kb,jb->kj', l1, t1) + mij
    mia1 = t1 + numpy.einsum('kc,jkbc->jb', l1, t2) * 2
    mia1 -= numpy.einsum('kc,jkcb->jb', l1, t2)
    mia1 -= reduce(numpy.dot, (t1, l1.T, t1))
    mia1 -= numpy.einsum('bd,jd->jb', mba, t1)
    mia1 -= numpy.einsum('lj,lb->jb', mij, t1)

    l2new = mycc._add_vvvv(None, l2, eris, with_ovvv=False, t2sym='jiba')
    l1new  = numpy.einsum('ijab,jb->ia', l2new, t1) * 2
    l1new -= numpy.einsum('jiab,jb->ia', l2new, t1)
    l2new *= .5  # *.5 because of l2+l2.transpose(1,0,3,2) in the end
    tmp = None

    w1 = imds.w1 - numpy.diag(mo_e_v)
    w2 = imds.w2 - numpy.diag(mo_e_o)

    l1new += fov
    l1new += numpy.einsum('ib,ba->ia', l1, w1)
    l1new -= numpy.einsum('ja,ij->ia', l1, w2)
    l1new -= numpy.einsum('ik,ka->ia', mij, imds.w4)
    l1new -= numpy.einsum('ca,ic->ia', mba, imds.w4)
    l1new += numpy.einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= numpy.einsum('ijba,bj->ia', l2, imds.w3)

    l2new += numpy.einsum('ia,jb->ijab', l1, imds.w4)
    l2new += lib.einsum('jibc,ca->jiba', l2, w1)
    l2new -= lib.einsum('jk,kiba->jiba', w2, l2)

    eris_ovoo = _cp(eris.ovoo)
    l1new -= numpy.einsum('iajk,kj->ia', eris_ovoo, mij1) * 2
    l1new += numpy.einsum('jaik,kj->ia', eris_ovoo, mij1)
    l2new -= lib.einsum('jbki,ka->jiba', eris_ovoo, l1)
    eris_ovoo = None

    tau = _ccsd.make_tau(t2, t1, t1)
    l2tau = lib.einsum('ijcd,klcd->ijkl', l2, tau)
    tau = None
    l2t1 = lib.einsum('jidc,kc->ijkd', l2, t1)

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2*5
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    log.debug1('block size = %d, nocc = %d is divided into %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))

    l1new -= numpy.einsum('jb,jiab->ia', l1, _cp(eris.oovv))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        l1new[:,p0:p1] += numpy.einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
        l1new -= numpy.einsum('ibca,bc->ia', eris_ovvv, mba1[p0:p1])
        l2new[:,:,p0:p1] += lib.einsum('jbac,ic->jiba', eris_ovvv, l1)
        m4 = lib.einsum('ijkd,kadb->ijab', l2t1, eris_ovvv)
        l2new[:,:,p0:p1] -= m4
        l1new[:,p0:p1] -= numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijab,ia->jb', m4, t1[:,p0:p1]) * 2
        l1new[:,p0:p1] += numpy.einsum('jiab,jb->ia', m4, t1)
        l1new += numpy.einsum('jiab,ia->jb', m4, t1[:,p0:p1])
        eris_ovvv = m4 = None

        eris_voov = _cp(eris.ovvo[:,p0:p1].transpose(1,0,3,2))
        l1new[:,p0:p1] += numpy.einsum('jb,aijb->ia', l1, eris_voov) * 2
        l2new[:,:,p0:p1] += eris_voov.transpose(1,2,0,3) * .5
        l2new[:,:,p0:p1] -= lib.einsum('bjic,ca->jiba', eris_voov, mba1)
        l2new[:,:,p0:p1] -= lib.einsum('bjka,ik->jiba', eris_voov, mij1)
        l1new[:,p0:p1] += numpy.einsum('aijb,jb->ia', eris_voov, mia1) * 2
        l1new -= numpy.einsum('bija,jb->ia', eris_voov, mia1[:,p0:p1])
        m4 = lib.einsum('ijkl,aklb->ijab', l2tau, eris_voov)
        l2new[:,:,p0:p1] += m4 * .5
        l1new[:,p0:p1] += numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijba,jb->ia', m4, t1[:,p0:p1])

        saved_wvooo = _cp(imds.wvooo[p0:p1])
        l1new -= lib.einsum('ckij,jkca->ia', saved_wvooo, l2[:,:,p0:p1])
        saved_wvovv = _cp(imds.wvvov[p0:p1])
        # Watch out memory usage here, due to the l2 transpose
        l1new[:,p0:p1] += lib.einsum('abkc,kibc->ia', saved_wvovv, l2)
        saved_wvooo = saved_wvovv = None

        saved_wvOOv = _cp(imds.wvOOv[p0:p1])
        tmp_voov = _cp(imds.wVOov[p0:p1]) * 2
        tmp_voov += saved_wvOOv
        tmp = l2.transpose(0,2,1,3) - l2.transpose(0,3,1,2)*.5
        l2new[:,:,p0:p1] += lib.einsum('iakc,bjkc->jiba', tmp, tmp_voov)
        tmp = None

        tmp = lib.einsum('jkca,bikc->jiba', l2, saved_wvOOv)
        l2new[:,:,p0:p1] += tmp
        l2new[:,:,p0:p1] += tmp.transpose(1,0,2,3) * .5
        saved_wvOOv = tmp = None

    saved_woooo = _cp(imds.woooo)
    m3 = lib.einsum('ijkl,klab->ijab', saved_woooo, l2)
    l2new += m3 * .5
    l1new += numpy.einsum('ijab,jb->ia', m3, t1) * 2
    l1new -= numpy.einsum('ijba,jb->ia', m3, t1)
    saved_woooo = m3 = None
    #time1 = log.timer_debug1('lambda pass [%d:%d]'%(p0, p1), *time1)

    eia = lib.direct_sum('i-a->ia', mo_e_o, mo_e_v)
    l1new /= eia

#    l2new = l2new + l2new.transpose(1,0,3,2)
#    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
#    l2new += l2
    for i in range(nocc):
        if i > 0:
            l2new[i,:i] += l2new[:i,i].transpose(0,2,1)
            l2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            l2new[:i,i] = l2new[i,:i].transpose(0,2,1)
        l2new[i,i] = l2new[i,i] + l2new[i,i].T
        l2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
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

    eris = ccsd._ChemistsERIs()
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
    idx = numpy.tril_indices(nvir)
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
    eris.vvvv = ao2mo.restore(4,eri0[nocc:,nocc:,nocc:,nocc:],nvir)
    eris.fock = fock0
    eris.mo_energy = fock0.diagonal()

    imds = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_lambda(mcc, t1, t2, l1, l2, eris, imds)
    print(lib.finger(l1new) - -6699.5335665027187)
    print(lib.finger(l2new) - -514.7001243502192 )
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())

    mcc.max_memory = 0
    imds = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_lambda(mcc, t1, t2, l1, l2, eris, imds)
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
