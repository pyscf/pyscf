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
from pyscf import gto
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm_incore as ccsd_rdm
from pyscf.scf import rhf_grad
from pyscf.scf import cphf

BLKSIZE = 192

def IX_intermediates(mycc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
# Note eris are in Chemist's notation
        eris = ccsd._ERIS(mycc)
    if d1 is None:
        d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    if d2 is None:
        d2 = ccsd_rdm.gamma2_incore(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2

    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir

# Note Ioo, Ivv are not hermitian
    Ioo = numpy.zeros((nocc,nocc))
    Ivv = numpy.zeros((nvir,nvir))
    Ivo = numpy.zeros((nvir,nocc))
    Xvo = numpy.zeros((nvir,nocc))

    eris_oooo = _cp(eris.oooo)
    eris_ooov = _cp(eris.ooov)
    d_oooo = _cp(doooo)
    d_oooo = _cp(d_oooo + d_oooo.transpose(1,0,2,3))
    #:Ioo += numpy.einsum('jmlk,imlk->ij', d_oooo, eris_oooo) * 2
    Ioo += lib.dot(eris_oooo.reshape(nocc,-1), d_oooo.reshape(nocc,-1).T, 2)
    d_oooo = _cp(d_oooo.transpose(0,2,3,1))
    #:Xvo += numpy.einsum('iljk,ljka->ai', d_oooo, eris_ooov) * 2
    Xvo += lib.dot(eris_ooov.reshape(-1,nvir).T, d_oooo.reshape(nocc,-1).T, 2)
    Xvo +=(numpy.einsum('kj,kjia->ai', doo, eris_ooov) * 4
         - numpy.einsum('kj,ikja->ai', doo+doo.T, eris_ooov))
    eris_oooo = eris_ooov = d_oooo = None

    d_ooov = _cp(dooov)
    eris_oooo = _cp(eris.oooo)
    eris_ooov = _cp(eris.ooov)
    #:Ivv += numpy.einsum('ijkb,ijka->ab', d_ooov, eris_ooov)
    #:Ivo += numpy.einsum('jlka,jlki->ai', d_ooov, eris_oooo)
    Ivv += lib.dot(eris_ooov.reshape(-1,nvir).T, d_ooov.reshape(-1,nvir))
    Ivo += lib.dot(d_ooov.reshape(-1,nvir).T, eris_oooo.reshape(-1,nocc))
    #:Ioo += numpy.einsum('klja,klia->ij', d_ooov, eris_ooov)
    #:Xvo += numpy.einsum('kjib,kjba->ai', d_ooov, eris.oovv)
    eris_oovv = _cp(eris.oovv)
    tmp = _cp(d_ooov.transpose(0,1,3,2).reshape(-1,nocc))
    tmpooov = _cp(eris_ooov.transpose(0,1,3,2))
    Ioo += lib.dot(tmpooov.reshape(-1,nocc).T, tmp)
    Xvo += lib.dot(eris_oovv.reshape(-1,nvir).T, tmp)
    eris_oooo = tmp = None

    d_ooov = d_ooov + d_ooov.transpose(1,0,2,3)
    eris_ovov = _cp(eris.ovov)
    #:Ioo += numpy.einsum('jlka,ilka->ij', d_ooov, eris_ooov)
    #:Xvo += numpy.einsum('ijkb,kbja->ai', d_ooov, eris.ovov)
    Ioo += lib.dot(eris_ooov.reshape(nocc,-1), d_ooov.reshape(nocc,-1).T)
    Xvo += lib.dot(eris_ovov.reshape(-1,nvir).T,
                   _cp(d_ooov.transpose(0,2,3,1).reshape(nocc,-1)).T)
    d_ooov = None

    #:Ioo += numpy.einsum('kjba,kiba->ij', d_oovv, eris.oovv)
    #:Ivv += numpy.einsum('ijcb,ijca->ab', d_oovv, eris.oovv)
    #:Ivo += numpy.einsum('kjba,kjib->ai', d_oovv, eris.ooov)
    d_oovv = _cp(doovv + doovv.transpose(1,0,3,2))
    for i in range(nocc):
        Ioo += lib.dot(eris_oovv[i].reshape(nocc, -1), d_oovv[i].reshape(nocc,-1).T)
    Ivv += lib.dot(eris_oovv.reshape(-1,nvir).T, d_oovv.reshape(-1,nvir))
    Ivo += lib.dot(d_oovv.reshape(-1,nvir).T, tmpooov.reshape(-1,nocc))
    d_oovv = _ccsd.precontract(d_oovv.reshape(-1,nvir,nvir)).reshape(nocc,nocc,-1)
    eris_ooov = tmpooov = None

    blksize = 4
    d_ovov = numpy.empty((nocc,nvir,nocc,nvir))
    for p0, p1 in prange(0, nocc, blksize):
        d_ovov[p0:p1] = _cp(dovov[p0:p1])
        d_ovvo = _cp(dovvo[p0:p1])
        for i in range(p0,p1):
            d_ovov[i] += d_ovvo[i-p0].transpose(0,2,1)
    d_ovvo = None
    #:d_ovov = d_ovov + d_ovov.transpose(2,3,0,1)
    lib.transpose_sum(d_ovov.reshape(nov,nov), inplace=True)
    #:Ivo += numpy.einsum('jbka,jbki->ai', d_ovov, eris.ovoo)
    Ivo += lib.dot(d_ovov.reshape(-1,nvir).T, _cp(eris.ovoo).reshape(-1,nocc))
    #:Ioo += numpy.einsum('jakb,iakb->ij', d_ovov, eris.ovov)
    #:Ivv += numpy.einsum('jcib,jcia->ab', d_ovov, eris.ovov)
    Ioo += lib.dot(eris_ovov.reshape(nocc,-1), d_ovov.reshape(nocc,-1).T)
    Ivv += lib.dot(eris_ovov.reshape(-1,nvir).T, d_ovov.reshape(-1,nvir))

    nvir_pair = nvir * (nvir+1) // 2
    bufe_ovvv = numpy.empty((blksize,nvir,nvir,nvir))
    bufc_ovvv = numpy.empty((blksize,nvir,nvir_pair))
    bufc_ovvv.data = bufe_ovvv.data
    c_vvvo = numpy.empty((nvir_pair,nvir,nocc))
    for p0, p1 in prange(0, nocc, blksize):
        d_ovvv = numpy.empty((p1-p0,nvir,nvir,nvir))
        #:Ivo += numpy.einsum('jadc,jidc->ai', d_ovvv, eris_oovv)
        for i in range(p1-p0):
            lib.dot(dovvv[p0+i].reshape(nvir,-1),
                    eris_oovv[p0+i].reshape(nocc,-1).T, 1, Ivo, 1)

        c_ovvv = bufc_ovvv[:p1-p0]
        # tril part of (d_ovvv + d_ovvv.transpose(0,1,3,2))
        _ccsd.precontract(dovvv[p0:p1].reshape(-1,nvir,nvir), out=c_ovvv)
        for i0, i1, in prange(0, nvir_pair, BLKSIZE):
            for j0, j1 in prange(0, nvir, BLKSIZE//(p1-p0)+1):
                c_vvvo[i0:i1,j0:j1,p0:p1] = c_ovvv[:,j0:j1,i0:i1].transpose(2,1,0)
        eris_ovx = _cp(eris.ovvv[p0:p1])
        #:Xvo += numpy.einsum('jibc,jabc->ai', d_oovv, eris_ovvv)
        #:Ivv += numpy.einsum('ibdc,iadc->ab', d_ovvv, eris_ovvv)
        for i in range(p1-p0):
            lib.dot(eris_ovx[i].reshape(nvir,-1),
                    d_oovv[p0+i].reshape(nocc,-1).T, 1, Xvo, 1)
            lib.dot(eris_ovx[i].reshape(nvir,-1),
                    c_ovvv[i].reshape(nvir,-1).T, 1, Ivv, 1)

        eris_ovvv = bufe_ovvv[:p1-p0]
        lib.unpack_tril(eris_ovx.reshape(-1,nvir_pair),
                        out=eris_ovvv.reshape(-1,nvir**2))
        eris_ovx = None
        #:Xvo += numpy.einsum('icjb,acjb->ai', d_ovov, eris_vvov)
        d_ovvo = _cp(d_ovov[p0:p1].transpose(0,1,3,2))
        lib.dot(eris_ovvv.reshape(-1,nvir).T, d_ovvo.reshape(-1,nocc), 1, Xvo, 1)

        e_ovvo, d_ovvo = d_ovvo, None
        for i in range(p1-p0):
            d_ovvv[i] = _ccsd.sum021(dovvv[p0+i])
            e_ovvo[i] = eris_ovov[p0+i].transpose(0,2,1)
        #:Ivo += numpy.einsum('jcab,jcib->ai', d_ovvv, eris_ovov)
        #:Ivv += numpy.einsum('icdb,icda->ab', d_ovvv, eris_ovvv)
        lib.dot(d_ovvv.reshape(-1,nvir).T,
                e_ovvo[:p1-p0].reshape(-1,nocc), 1, Ivo, 1)
        lib.dot(eris_ovvv.reshape(-1,nvir).T, d_ovvv.reshape(-1,nvir), 1, Ivv, 1)

        Xvo[:,p0:p1] +=(numpy.einsum('cb,iacb->ai', dvv, eris_ovvv) * 4
                      - numpy.einsum('cb,icba->ai', dvv+dvv.T, eris_ovvv))
    d_oovv = d_ovvv = bufc_ovvv = bufe_ovvv = None
    eris_ovov = eris_ovvv = eris_oovv = e_ovvo = None

    eris_ovvv = _cp(eris.ovvv)
    bufe_vvvo = numpy.empty((blksize*nvir,nvir,nocc))
    bufe_vvvv = numpy.empty((blksize*nvir,nvir,nvir))
    bufd_vvvv = numpy.empty((blksize*nvir,nvir,nvir))
    for p0, p1 in prange(0, nvir, blksize):
        off0 = p0*(p0+1)//2
        off1 = p1*(p1+1)//2
        d_vvvv = _cp(dvvvv[off0:off1]) * 4
        for i in range(p0, p1):
            d_vvvv[i*(i+1)//2+i-off0] *= .5
        d_vvvv = lib.unpack_tril(d_vvvv, out=bufd_vvvv[:off1-off0])
        eris_vvvv = lib.unpack_tril(eris.vvvv[off0:off1], out=bufe_vvvv[:off1-off0])
        #:Ivv += numpy.einsum('decb,deca->ab', d_vvvv, eris_vvvv) * 2
        #:Xvo += numpy.einsum('icdb,acdb->ai', d_ovvv, eris_vvvv)
        lib.dot(eris_vvvv.reshape(-1,nvir).T, d_vvvv.reshape(-1,nvir), 2, Ivv, 1)
        d_vvvo = _cp(c_vvvo[off0:off1])
        lib.dot(eris_vvvv.reshape(-1,nvir).T, d_vvvo.reshape(-1,nocc), 1, Xvo, 1)

        #:Ioo += numpy.einsum('abjc,abci->ij', d_vvov, eris_vvvo)
        #:Ivo += numpy.einsum('dbca,dbci->ai', d_vvvv, eris_vvvo) * 2
        eris_vvvo = bufe_vvvo[:off1-off0]
        for i0, i1 in prange(off0, off1, BLKSIZE):
            for j0, j1, in prange(0, nvir, BLKSIZE//nocc+1):
                eris_vvvo[i0-off0:i1-off0,j0:j1,:] = eris_ovvv[:,j0:j1,i0:i1].transpose(2,1,0)
        lib.dot(eris_vvvo.reshape(-1,nocc).T, d_vvvo.reshape(-1,nocc), 1, Ioo, 1)
        lib.dot(d_vvvv.reshape(-1,nvir).T, eris_vvvo.reshape(-1,nocc), 2, Ivo, 1)

    Ioo *= -1
    Ivv *= -1
    Ivo *= -1
    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


def response_dm1(mycc, t1, t2, l1, l2, eris=None, IX=None):
    if eris is None:
# Note eris are in Chemist's notation
        eris = ccsd._ERIS(mycc)
    if IX is None:
        Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        Ioo, Ivv, Ivo, Xvo = IX
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    def fvind(x):
        x = x.reshape(Xvo.shape)
        if eris is None:
            mo_coeff = mycc.mo_coeff
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            dm = (dm + dm.T) * 2
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, mycc._scf.get_veff(mol, dm),
                                   mo_coeff[:,:nocc]))
        else:
            eris_ovov = _cp(eris.ovov)
            v  = numpy.einsum('iajb,bj->ia', eris_ovov, x) * 4
            v -= numpy.einsum('ibja,bj->ia', eris_ovov, x)
            eris_ovov = None
            v -= numpy.einsum('jiab,bj->ia', _cp(eris.oovv), x)
        return v.T
    mo_energy = eris.fock.diagonal()
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[:nocc] = 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1


#
# Note: only works with canonical orbitals
# Non-canonical formula refers to JCP, 95, 2639
#
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           mf_grad=None, verbose=logger.INFO):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = ccsd._ERIS(mycc)
    if mf_grad is None:
        mf_grad = rhf_grad.Gradients(mycc._scf)

    log = logger.Logger(mycc.stdout, mycc.verbose)
    time0 = time.clock(), time.time()
    mol = mycc.mol
    if mycc.frozen is not 0:
        raise NotImplementedError('frozen orbital ccsd_grad')
    moidx = ccsd.get_moidx(mycc)
    mo_coeff = mycc.mo_coeff[:,moidx]  #FIXME: ensure mycc.mo_coeff is canonical orbital
    mo_energy = eris.fock.diagonal()
    nocc, nvir = t1.shape
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    log.debug('Build ccsd rdm1 intermediates')
    d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer('rdm1 intermediates', *time0)

    log.debug('Build ccsd rdm2 intermediates')
    d2 = ccsd_rdm.gamma2_incore(mycc, t1, t2, l1, l2)
    time1 = log.timer('rdm2 intermediates', *time1)
    log.debug('Build ccsd response_rdm1')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris, d1, d2)
    time1 = log.timer('response_rdm1 intermediates', *time1)

    dm1mo = response_dm1(mycc, t1, t2, l1, l2, eris, (Ioo, Ivv, Ivo, Xvo))
    dm1mo[:nocc,:nocc] = doo * 2
    dm1mo[nocc:,nocc:] = dvv * 2
    dm1ao = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    im1 = numpy.zeros_like(dm1mo)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T
    im1 = reduce(numpy.dot, (mo_coeff, im1, mo_coeff.T))
    time1 = log.timer('response_rdm1', *time1)

    log.debug('symmetrized rdm2 and MO->AO transformation')
    dm1_with_hf = dm1mo.copy()
    for i in range(nocc):
        dm1_with_hf[i,i] += 1
    dm2ao = _rdm2_mo2ao(mycc, d2, dm1_with_hf, mo_coeff)
    time1 = log.timer('MO->AO transformation', *time1)

    log.debug('h1 and JK1')
    h1 = mf_grad.get_hcore(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf4sij = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1ao+dm1ao.T), p1))
    time1 = log.timer('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    dm1ao += hf_dm1
    zeta += mf_grad.make_rdm1e(mycc.mo_energy, mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] =(numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
              + numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1]))
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        h1ao = mf_grad._grad_rinv(mol, ia)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        de[k] +=(numpy.einsum('xij,ij->x', h1ao, dm1ao)
               + numpy.einsum('xji,ij->x', h1ao, dm1ao))
# -s[1]*e \dot DM,  contribute to f1
        de[k] -=(numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
               + numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1]))
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf4sij[p0:p1]) * 2

# 2e AO integrals dot 2pdm
        eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                         shls_slice=(shl0,shl1,0,mol.nbas,0,mol.nbas,0,mol.nbas))
        eri1 = eri1.reshape(3,p1-p0,nao,-1)
        dm2buf = _load_block_tril(dm2ao, p0, p1)
        de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2

        eri1 = dm2buf = None
        log.debug('grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer('grad of atom %d'%ia, *time1)

    de += rhf_grad.grad_nuc(mol)
    log.note('CCSD gradinets')
    log.note('==============')
    log.note('           x                y                z')
    for k, ia in enumerate(atmlst):
        log.note('%d %s  %15.9f  %15.9f  %15.9f', ia, mol.atom_symbol(ia), *de[k])
    log.timer('CCSD gradients', *time0)
    return de

def _rdm2_mo2ao(mycc, d2, dm1, mo_coeff):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = time.clock(), time.time()
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nov = nocc * nvir
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    nvir_pair = nvir * (nvir+1) //2
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        fdrv = getattr(_ccsd.libcc, 'AO2MOnr_e2_drv')
        pao_loc = ctypes.POINTER(ctypes.c_void_p)()
        fdrv(_ccsd.libcc.AO2MOtranse2_nr_s1, _ccsd.libcc.CCmmm_transpose_sum,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    blksize = 4
    bufop = numpy.empty((nocc*nmo,nao_pair))
    bufvv = numpy.empty((nvir_pair,nao_pair))
    pool = numpy.empty((blksize,nmo,nmo,nmo))
    for p0, p1 in prange(0, nocc, blksize):
        buf1 = pool[:p1-p0]
        buf1[:,:nocc,:nocc,:nocc] = doooo[p0:p1]
        buf1[:,:nocc,:nocc,nocc:] = dooov[p0:p1]
        buf1[:,:nocc,nocc:,:nocc] = 0
        buf1[:,:nocc,nocc:,nocc:] = doovv[p0:p1]
        buf1[:,nocc:,:nocc,:nocc] = 0
        buf1[:,nocc:,:nocc,nocc:] = dovov[p0:p1]
        buf1[:,nocc:,nocc:,:nocc] = dovvo[p0:p1]
        buf1[:,nocc:,nocc:,nocc:] = dovvv[p0:p1]
        for i in range(p1-p0):
            buf1[i,p0+i,:,:] += dm1
            buf1[i,:,:,p0+i] -= dm1 * .5
        _trans(buf1.reshape(-1,nmo**2), (0,nmo,0,nmo), bufop[p0*nmo:p1*nmo])

    pool = pool.ravel()[:blksize*nvir**3].reshape((blksize*nvir,nvir,nvir))
    for p0, p1 in prange(0, nvir_pair, blksize*nvir):
        buf1 = lib.unpack_tril(_cp(dvvvv[p0:p1]), out=pool[:p1-p0])
        _trans(buf1, (nocc,nmo,nocc,nmo), bufvv[p0:p1])
    pool = buf1 = None

    # trans ij of d_ijkl
    bufaa = numpy.empty((nao_pair,nao_pair))
    tmpbuf1 = numpy.empty((BLKSIZE,nmo,nmo))
    tmpbuf2 = numpy.empty((BLKSIZE,nvir_pair))
    tmpbuf3 = numpy.empty((BLKSIZE,nvir,nvir))
    tmpbuf1[:,nocc:,:nocc] = 0
    for i0, i1 in prange(0, nao_pair, BLKSIZE):
        buf1 = tmpbuf1[:i1-i0].reshape(i1-i0,-1)
        for j0, j1 in prange(0, nocc*nmo, BLKSIZE):
            buf1[:,j0:j1] = bufop[j0:j1,i0:i1].T
        buf1 = tmpbuf1[:i1-i0]
        tmp  = tmpbuf2[:i1-i0]
        for j0, j1 in prange(0, nvir_pair, BLKSIZE):
            tmp[:,j0:j1] = bufvv[j0:j1,i0:i1].T
        buf1[:,nocc:,nocc:] = lib.unpack_tril(tmp, out=tmpbuf3[:i1-i0])
        _trans(buf1, (0,nmo,0,nmo), bufaa[i0:i1])
# dm2 + dm2.transpose(2,3,0,1)
        for j0, j1, in prange(0, i0, BLKSIZE):
            bufaa[i0:i1,j0:j1] += bufaa[j0:j1,i0:i1].T
            bufaa[j0:j1,i0:i1] = bufaa[i0:i1,j0:j1].T
        bufaa[i0:i1,i0:i1] = bufaa[i0:i1,i0:i1] + bufaa[i0:i1,i0:i1].T
    tmpbuf1 = tmpbuf2 = tmpbuf3 = bufop = bufvv = pool = None

    idx = numpy.arange(nao)
    idx = idx*(idx+1)//2 + idx
    bufaa[:,idx] *= .5

    return bufaa

#
# .
# . .
# ----+             -----------
# ----|-+       =>  -----------
# . . | | .
# . . | | . .
#
def _load_block_tril(dat, row0, row1, out=None):
    shape = dat.shape
    nd = int(numpy.sqrt(shape[0]*2))
    if out is None:
        out = numpy.empty((row1-row0,nd)+shape[1:])
    p0 = row0*(row0+1)//2
    for i in range(row0, row1):
        out[i-row0,:i+1] = _cp(dat[p0:p0+i+1])
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
        p0 += i + 1
    for i in range(row1, nd):
        i2 = i*(i+1)//2
        out[:,i] = dat[i2+row0:i2+row1]
    return out


def hf_get_jk_incore(eri, dm):
    ni, nj = eri.shape[:2]
    vj = numpy.empty((ni,nj))
    vk = numpy.empty((ni,nj))
    _ccsd.libcc.CCvhfs2kl(eri.ctypes.data_as(ctypes.c_void_p),
                          dm.ctypes.data_as(ctypes.c_void_p),
                          vj.ctypes.data_as(ctypes.c_void_p),
                          vk.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(ni), ctypes.c_int(nj))
    return vj, vk

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import grad

    mol = gto.M()
    mf = scf.RHF(mol)

    mycc = ccsd.CCSD(mf)

    numpy.random.seed(2)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = numpy.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*20
    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = numpy.random.random((nocc,nvir))
    l2 = numpy.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))
    eris = lambda:None
    idx = numpy.tril_indices(nvir)
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovo = eri0[:nocc,:nocc,nocc:,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:]
    eris.ovvv = eris.ovvv[:,:,idx[0],idx[1]].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:]
    eris.vvvv = eris.vvvv[idx[0],idx[1]][:,idx[0],idx[1]].copy()
    eris.fock = fock0

    print('-----------------------------------')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    numpy.random.seed(1)
    h1 = numpy.random.random((nmo,nmo))
    h1 = h1 + h1.T
    print(numpy.einsum('ij,ij', h1[:nocc,:nocc], Ioo) - 2613213.0346526774)
    print(numpy.einsum('ab,ab', h1[nocc:,nocc:], Ivv) - 6873038.9907923322)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Ivo) - 4353360.4241635408)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Xvo) - 203575.42337558540)
    dm1 = response_dm1(mycc, t1, t2, l1, l2, eris)
    print(numpy.einsum('pq,pq', h1[nocc:,:nocc], dm1[nocc:,:nocc])--486.638981725713393)

    print('-----------------------------------')
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
    print('gcc')
    print(g1)
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]

    lib.parameters.BOHR = 1
    r = 1.76#.748
    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % r,
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    ghf = grad.RHF(mf).grad()
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
    print('gcc')
    print(g1)
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]
