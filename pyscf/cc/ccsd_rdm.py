#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    nocc, nvir = t1.shape
    doo =-numpy.einsum('ja,ia->ij', t1, l1)
    dvv = numpy.einsum('ia,ib->ab', t1, l1)
    xtv = numpy.einsum('ie,me->im', t1, l1)
    dvo = t1.T - numpy.einsum('im,ma->ai', xtv, t1)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    doo -= lib.einsum('jkab,ikab->ij', theta, l2)
    dvv += lib.einsum('jica,jicb->ab', theta, l2)
    xt1  = lib.einsum('mnef,inef->mi', l2, theta)
    xt2  = lib.einsum('mnaf,mnef->ea', l2, theta)
    dvo += numpy.einsum('imae,me->ai', theta, l1)
    dvo -= numpy.einsum('mi,ma->ai', xt1, t1)
    dvo -= numpy.einsum('ie,ae->ai', t1, xt2)
    dov = l1
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def _gamma2_intermediates(mycc, t1, t2, l1, l2, compress_vvvv=False):
    f = lib.H5TmpFile()
    _gamma2_outcore(mycc, t1, t2, l1, l2, f, compress_vvvv)
    d2 = (f['dovov'].value, f['dvvvv'].value, f['doooo'].value, f['doovv'].value,
          f['dovvo'].value, None,             f['dovvv'].value, f['dooov'].value)
    return d2

def _gamma2_outcore(mycc, t1, t2, l1, l2, h5fobj, compress_vvvv=False):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir * (nvir+1) //2
    dtype = numpy.result_type(t1, t2, l1, l2).char
    if compress_vvvv:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), dtype)
    else:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir,nvir,nvir,nvir), dtype)
    dovvo = h5fobj.create_dataset('dovvo', (nocc,nvir,nvir,nocc), dtype,
                                  chunks=(nocc,1,nvir,nocc))
    fswap = lib.H5TmpFile()

    time1 = time.clock(), time.time()
    pvOOv = lib.einsum('ikca,jkcb->aijb', l2, t2)
    moo = numpy.einsum('dljd->jl', pvOOv) * 2
    mvv = numpy.einsum('blld->db', pvOOv) * 2
    gooov = lib.einsum('kc,cija->jkia', t1, pvOOv)
    fswap['mvOOv'] = pvOOv
    pvOOv = None

    pvoOV = -lib.einsum('ikca,jkbc->aijb', l2, t2)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    pvoOV += lib.einsum('ikac,jkbc->aijb', l2, theta)
    moo += numpy.einsum('dljd->jl', pvoOV)
    mvv += numpy.einsum('blld->db', pvoOV)
    gooov -= lib.einsum('jc,cika->jkia', t1, pvoOV)
    fswap['mvoOV'] = pvoOV
    pvoOV = None

    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2
        - numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1) + moo*.5

    tau = numpy.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    goooo = lib.einsum('ijab,klab->ijkl', tau, l2)*.5
    h5fobj['doooo'] = (goooo.transpose(0,2,1,3)*2 -
                       goooo.transpose(0,3,1,2)).conj()

    gooov += numpy.einsum('ji,ka->jkia', -.5*moo, t1)
    gooov += lib.einsum('la,jkil->jkia', 2*t1, goooo)
    gooov -= lib.einsum('ib,jkba->jkia', l1, tau)
    gooov = gooov.conj()
    gooov -= lib.einsum('jkba,ib->jkia', l2, t1)
    h5fobj['dooov'] = gooov.transpose(0,2,1,3)*2 - gooov.transpose(1,2,0,3)
    tau = goovo = None
    time1 = log.timer_debug1('rdm intermediates pass1', *time1)

    goovv = numpy.einsum('ia,jb->ijab', mia.conj(), t1.conj())
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc**2*nvir*6
    blksize = min(nocc, nvir, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    doovv = h5fobj.create_dataset('doovv', (nocc,nocc,nvir,nvir), dtype,
                                  chunks=(nocc,nocc,1,nvir))

    log.debug1('rdm intermediates pass 2: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    for p0, p1 in lib.prange(0, nvir, blksize):
        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        tau += t2[:,:,p0:p1]
        tmpoovv  = lib.einsum('ijkl,klab->ijab', goooo, tau)
        tmpoovv -= lib.einsum('jk,ikab->ijab', mij, tau)
        tmpoovv -= lib.einsum('cb,ijac->ijab', mab, t2[:,:,p0:p1])
        tmpoovv -= lib.einsum('bd,ijad->ijab', mvv*.5, tau)
        tmpoovv += .5 * tau
        tmpoovv = tmpoovv.conj()
        tmpoovv += .5 * l2[:,:,p0:p1]
        goovv[:,:,p0:p1] += tmpoovv

        pvOOv = fswap['mvOOv'][p0:p1]
        pvoOV = fswap['mvoOV'][p0:p1]
        gOvvO = lib.einsum('kiac,jc,kb->iabj', l2[:,:,p0:p1], t1, t1)
        gOvvO += numpy.einsum('aijb->iabj', pvOOv)
        govVO = numpy.einsum('ia,jb->iabj', l1[:,p0:p1], t1)
        govVO -= lib.einsum('ikac,jc,kb->iabj', l2[:,:,p0:p1], t1, t1)
        govVO += numpy.einsum('aijb->iabj', pvoOV)
        dovvo[:,p0:p1] = 2*govVO + gOvvO
        doovv[:,:,p0:p1] = (-2*gOvvO - govVO).transpose(3,0,1,2).conj()
        gOvvO = govVO = None

        tau -= t2[:,:,p0:p1] * .5
        for q0, q1 in lib.prange(0, nvir, blksize):
            goovv[:,:,q0:q1,:] += lib.einsum('dlib,jlda->ijab', pvOOv, tau[:,:,:,q0:q1]).conj()
            goovv[:,:,:,q0:q1] -= lib.einsum('dlia,jldb->ijab', pvoOV, tau[:,:,:,q0:q1]).conj()
            tmp = pvoOV[:,:,:,q0:q1] + pvOOv[:,:,:,q0:q1]*.5
            goovv[:,:,q0:q1,:] += lib.einsum('dlia,jlbd->ijab', tmp, t2[:,:,:,p0:p1]).conj()
        pvOOv = pvoOV = tau = None
        time1 = log.timer_debug1('rdm intermediates pass2 [%d:%d]'%(p0, p1), *time1)
    h5fobj['dovov'] = goovv.transpose(0,2,1,3) * 2 - goovv.transpose(1,2,0,3)
    goovv = goooo = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = max(nocc**2*nvir*2+nocc*nvir**2*3,
               nvir**3*2+nocc*nvir**2*2+nocc**2*nvir*2)
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*.9e6/8/unit)))
    iobuflen = int(256e6/8/blksize)
    log.debug1('rdm intermediates pass 3: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    dovvv = h5fobj.create_dataset('dovvv', (nocc,nvir,nvir,nvir), dtype,
                                  chunks=(nocc,min(nocc,nvir),1,nvir))
    time1 = time.clock(), time.time()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        l2tmp = l2[:,:,p0:p1]
        gvvvv = lib.einsum('ijab,ijcd->abcd', l2tmp, t2)
        jabc = lib.einsum('ijab,ic->jabc', l2tmp, t1)
        gvvvv += lib.einsum('jabc,jd->abcd', jabc, t1)
        l2tmp = jabc = None

        if compress_vvvv:
# symmetrize dvvvv because it does not affect the results of ccsd_grad
# dvvvv = gvvvv.transpose(0,2,1,3)-gvvvv.transpose(0,3,1,2)*.5
# dvvvv = (dvvvv+dvvvv.transpose(0,1,3,2)) * .5
# dvvvv = (dvvvv+dvvvv.transpose(1,0,2,3)) * .5
# now dvvvv == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
            tmp = numpy.empty((nvir,nvir,nvir))
            tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
            for i in range(p1-p0):
                vvv = gvvvv[i].conj().transpose(1,0,2)
                tmp[:] = vvv - vvv.transpose(2,1,0)*.5
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
        else:
            for i in range(p0, p1):
                vvv = gvvvv[i-p0].conj().transpose(1,0,2)
                dvvvv[i] = vvv - vvv.transpose(2,1,0)*.5

        gvovv = lib.einsum('adbc,id->aibc', gvvvv, -t1)
        gvvvv = None

        gvovv += lib.einsum('akic,kb->aibc', fswap['mvoOV'][p0:p1], t1)
        gvovv -= lib.einsum('akib,kc->aibc', fswap['mvOOv'][p0:p1], t1)

        gvovv += lib.einsum('ja,jibc->aibc', l1[:,p0:p1], t2)
        gvovv += lib.einsum('ja,jb,ic->aibc', l1[:,p0:p1], t1, t1)
        gvovv += numpy.einsum('ba,ic->aibc', mvv[:,p0:p1]*.5, t1)
        gvovv = gvovv.conj()
        gvovv += lib.einsum('ja,jibc->aibc', t1[:,p0:p1], l2)

        dovvv[:,:,p0:p1] = gvovv.transpose(1,3,0,2)*2 - gvovv.transpose(1,2,0,3)
        gvvov = None
        time1 = log.timer_debug1('rdm intermediates pass3 [%d:%d]'%(p0, p1), *time1)

    fswap = None
    dvvov = None
    return (h5fobj['dovov'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['doovv'],
            h5fobj['dovvo'], dvvov          , h5fobj['dovvv'], h5fobj['dooov'])

def make_rdm1(mycc, t1, t2, l1, l2, ao_repr=False):
    '''
    Spin-traced one-particle density matrix in MO basis (the occupied-virtual
    blocks from the orbital response contribution are not included).

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm2(mycc, t1, t2, l1, l2):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    f = lib.H5TmpFile()
    d2 = _gamma2_outcore(mycc, t1, t2, l1, l2, f, False)
    return _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True)

def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False):
    '''dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir
    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T
    dm1[numpy.diag_indices(nocc)] += 2

    if with_frozen and not (mycc.frozen is 0 or mycc.frozen is None):
        nmo = mycc.mo_occ.size
        nocc = numpy.count_nonzero(mycc.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[numpy.diag_indices(nocc)] = 2
        moidx = numpy.where(mycc.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = lib.einsum('pi,ij,qj->pq', mo, dm1, mo.conj())
    return dm1

# Note vvvv part of 2pdm have been symmetrized.  It does not correspond to
# vvvv part of CI 2pdm
def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True):
    r'''
    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo), dtype=doovv.dtype)

    dovov = numpy.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[:nocc,nocc:,:nocc,nocc:]+= dovov.transpose(2,3,0,1)
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    doovv = numpy.asarray(doovv)
    dm2[:nocc,:nocc,nocc:,nocc:] = doovv
    dm2[:nocc,:nocc,nocc:,nocc:]+= doovv.transpose(1,0,3,2).conj()
    dm2[nocc:,nocc:,:nocc,:nocc] = dm2[:nocc,:nocc,nocc:,nocc:].transpose(2,3,0,1)
    doovv = None

    dovvo = numpy.asarray(dovvo)
    dm2[:nocc,nocc:,nocc:,:nocc] = dovvo
    dm2[:nocc,nocc:,nocc:,:nocc]+= dovvo.transpose(3,2,1,0).conj()
    dm2[nocc:,:nocc,:nocc,nocc:] = dm2[:nocc,nocc:,nocc:,:nocc].transpose(1,0,3,2).conj()
    dovvo = None

    if len(dvvvv.shape) == 2:
# To handle the case of compressed vvvv, which is used in nuclear gradients
        dvvvv = ao2mo.restore(1, dvvvv, nvir)
        dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
        dm2[nocc:,nocc:,nocc:,nocc:]*= 4
    else:
        dvvvv = numpy.asarray(dvvvv)
        dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
        dm2[nocc:,nocc:,nocc:,nocc:]+= dvvvv.transpose(1,0,3,2).conj()
        dm2[nocc:,nocc:,nocc:,nocc:]*= 2
    dvvvv = None

    doooo = numpy.asarray(doooo)
    dm2[:nocc,:nocc,:nocc,:nocc] = doooo
    dm2[:nocc,:nocc,:nocc,:nocc]+= doooo.transpose(1,0,3,2).conj()
    dm2[:nocc,:nocc,:nocc,:nocc]*= 2
    doooo = None

    dovvv = numpy.asarray(dovvv)
    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0).conj()
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2).conj()
    dovvv = None

    dooov = numpy.asarray(dooov)
    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2).conj()
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0).conj()

    if with_frozen and not (mycc.frozen is 0 or mycc.frozen is None):
        nmo, nmo0 = mycc.mo_occ.size, nmo
        nocc = numpy.count_nonzero(mycc.mo_occ > 0)
        rdm2 = numpy.zeros((nmo,nmo,nmo,nmo), dtype=dm2.dtype)
        moidx = numpy.where(mycc.get_frozen_mask())[0]
        idx = (moidx.reshape(-1,1) * nmo + moidx).ravel()
        lib.takebak_2d(rdm2.reshape(nmo**2,nmo**2),
                       dm2.reshape(nmo0**2,nmo0**2), idx, idx)
        dm2 = rdm2

    if with_dm1:
        dm1 = _make_rdm1(mycc, d1, with_frozen)
        dm1[numpy.diag_indices(nocc)] -= 2

        for i in range(nocc):
            dm2[i,i,:,:] += dm1 * 2
            dm2[:,:,i,i] += dm1 * 2
            dm2[:,i,i,:] -= dm1
            dm2[i,:,:,i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 4
                dm2[i,j,j,i] -= 2

    # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
    # above. Transposing it so that it be contracted with ERIs (in Chemist's
    # notation):
    #   E = einsum('pqrs,pqrs', eri, rdm2)
    return dm2.transpose(1,0,3,2)


if __name__ == '__main__':
    from functools import reduce
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

    doo, dov, dvo, dvv = _gamma1_intermediates(mcc, t1, t2, l1, l2)
    print((numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc]))*2+20166.329861034799)
    print((numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:]))*2-58078.964019246778)
    print((numpy.einsum('ai,ia', dvo, fock0[:nocc,nocc:]))*2+74994.356886784764)
    print((numpy.einsum('ia,ai', dov, fock0[nocc:,:nocc]))*2-34.010188025702391)

    fdm2 = lib.H5TmpFile()
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            _gamma2_outcore(mcc, t1, t2, l1, l2, fdm2, True)
    print('dovov', lib.finger(numpy.array(dovov)) - -14384.907042073517)
    print('dvvvv', lib.finger(numpy.array(dvvvv)) - -25.374007033024839)
    print('doooo', lib.finger(numpy.array(doooo)) -  60.114594698129963)
    print('doovv', lib.finger(numpy.array(doovv)) - -79.176348067958401)
    print('dovvo', lib.finger(numpy.array(dovvo)) -   9.864134457251815)
    print('dovvv', lib.finger(numpy.array(dovvv)) - -421.90333700061342)
    print('dooov', lib.finger(numpy.array(dooov)) - -592.66863759586136)
    fdm2 = None

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            _gamma2_intermediates(mcc, t1, t2, l1, l2)
    print('dovov', lib.finger(numpy.array(dovov)) - -14384.907042073517)
    print('dvvvv', lib.finger(numpy.array(dvvvv)) -  45.872344902116758)
    print('doooo', lib.finger(numpy.array(doooo)) -  60.114594698129963)
    print('doovv', lib.finger(numpy.array(doovv)) - -79.176348067958401)
    print('dovvo', lib.finger(numpy.array(dovvo)) -   9.864134457251815)
    print('dovvv', lib.finger(numpy.array(dovvv)) - -421.90333700061342)
    print('dooov', lib.finger(numpy.array(dooov)) - -592.66863759586136)

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
          numpy.einsum('pq,qp', dm1, h1) - e2)

    print(numpy.allclose(dm2, dm2.transpose(1,0,3,2)))
    print(numpy.allclose(dm2, dm2.transpose(2,3,0,1)))

    d1 = numpy.einsum('kkpq->qp', dm2) / 9
    print(numpy.allclose(d1, dm1))

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).run()

    mycc = ccsd.CCSD(mf)
    mycc.frozen = 2
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    nmo = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False).reshape([nmo]*4)
    hcore = mf.get_hcore()
    h1 = reduce(numpy.dot, (mf.mo_coeff.T, hcore, mf.mo_coeff))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mycc.e_tot)
