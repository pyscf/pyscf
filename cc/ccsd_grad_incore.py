#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import _ctypes
import tempfile
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import ccsd_rdm_incore as ccsd_rdm
import pyscf.grad
from pyscf.grad import cphf

BLKSIZE = 192

def IX_intermediates(mycc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
# Note eris are in Chemist's notation
        eris = ccsd._ERIS(mycc)
    if d1 is None:
        doo, dvv = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    else:
        doo, dvv = d1
    if d2 is None:
        d2 = ccsd_rdm.gamma2_incore(mycc, t1, t2, l1, l2)
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2

    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir

# Note Ioo, Ivv are not hermitian
    Ioo = numpy.zeros((nocc,nocc))
    Ivv = numpy.zeros((nvir,nvir))
    Ivo = numpy.zeros((nvir,nocc))
    Xvo = numpy.zeros((nvir,nocc))

    eris_oooo = ccsd._cp(eris.oooo)
    eris_ooov = ccsd._cp(eris.ooov)
    d_oooo = ccsd._cp(doooo)
    d_oooo = ccsd._cp(d_oooo + d_oooo.transpose(1,0,2,3))
    #:Ioo += numpy.einsum('jmlk,imlk->ij', d_oooo, eris_oooo) * 2
    Ioo += lib.dot(eris_oooo.reshape(nocc,-1), d_oooo.reshape(nocc,-1).T, 2)
    d_oooo = ccsd._cp(d_oooo.transpose(0,2,3,1))
    #:Xvo += numpy.einsum('iljk,ljka->ai', d_oooo, eris_ooov) * 2
    Xvo += lib.dot(eris_ooov.reshape(-1,nvir).T, d_oooo.reshape(nocc,-1).T, 2)
    Xvo +=(numpy.einsum('kj,kjia->ai', doo, eris_ooov) * 2
         + numpy.einsum('kj,kjia->ai', doo, eris_ooov) * 2
         - numpy.einsum('kj,kija->ai', doo, eris_ooov)
         - numpy.einsum('kj,ijka->ai', doo, eris_ooov))
    eris_oooo = eris_ooov = d_oooo = None

    d_ooov = ccsd._cp(dooov)
    eris_oooo = ccsd._cp(eris.oooo)
    eris_ooov = ccsd._cp(eris.ooov)
    #:Ivv += numpy.einsum('ijkb,ijka->ab', d_ooov, eris_ooov)
    #:Ivo += numpy.einsum('jlka,jlki->ai', d_ooov, eris_oooo)
    Ivv += lib.dot(eris_ooov.reshape(-1,nvir).T, d_ooov.reshape(-1,nvir))
    Ivo += lib.dot(d_ooov.reshape(-1,nvir).T, eris_oooo.reshape(-1,nocc))
    #:Ioo += numpy.einsum('klja,klia->ij', d_ooov, eris_ooov)
    #:Xvo += numpy.einsum('kjib,kjba->ai', d_ooov, eris.oovv)
    eris_oovv = ccsd._cp(eris.oovv)
    tmp = ccsd._cp(d_ooov.transpose(0,1,3,2).reshape(-1,nocc))
    tmpooov = ccsd._cp(eris_ooov.transpose(0,1,3,2))
    Ioo += lib.dot(tmpooov.reshape(-1,nocc).T, tmp)
    Xvo += lib.dot(eris_oovv.reshape(-1,nvir).T, tmp)
    eris_oooo = tmp = None

    d_ooov = d_ooov + d_ooov.transpose(1,0,2,3)
    eris_ovov = ccsd._cp(eris.ovov)
    #:Ioo += numpy.einsum('jlka,ilka->ij', d_ooov, eris_ooov)
    #:Xvo += numpy.einsum('ijkb,kbja->ai', d_ooov, eris.ovov)
    Ioo += lib.dot(eris_ooov.reshape(nocc,-1), d_ooov.reshape(nocc,-1).T)
    Xvo += lib.dot(eris_ovov.reshape(-1,nvir).T,
                   ccsd._cp(d_ooov.transpose(0,2,3,1).reshape(nocc,-1)).T)
    d_ooov = None

    #:Ioo += numpy.einsum('kjba,kiba->ij', d_oovv, eris.oovv)
    #:Ivv += numpy.einsum('ijcb,ijca->ab', d_oovv, eris.oovv)
    #:Ivo += numpy.einsum('kjba,kjib->ai', d_oovv, eris.ooov)
    d_oovv = ccsd._cp(doovv) + doovv.transpose(1,0,3,2)
    for i in range(nocc):
        Ioo += lib.dot(eris_oovv[i].reshape(nocc, -1), d_oovv[i].reshape(nocc,-1).T)
    Ivv += lib.dot(eris_oovv.reshape(-1,nvir).T, d_oovv.reshape(-1,nvir))
    Ivo += lib.dot(d_oovv.reshape(-1,nvir).T, tmpooov.reshape(-1,nocc))
    eris_ooov = tmpooov = None

    blksize = 12
    nvir_pair = nvir * (nvir+1) // 2
    eris_ovvv = numpy.empty((nocc,nvir,nvir,nvir))
    bufd_ovvv = numpy.empty((blksize,nvir,nvir,nvir))
    tmp = numpy.empty((blksize,nvir,nvir,nocc))
    d_vvov = ccsd._cp(dvvov).reshape(-1,nov)
    for p0, p1 in ccsd.prange(0, nocc, blksize):
        d_ovvv = bufd_ovvv[:p1-p0].reshape(-1,nvir**2)
        off = p0 * nvir
        for i0, i1 in ccsd.prange(0, (p1-p0)*nvir, BLKSIZE):
            for j0, j1, in ccsd.prange(0, nvir**2, BLKSIZE):
                d_ovvv[i0:i1,j0:j1] = d_vvov[j0:j1,off+i0:off+i1].T
        d_ovvv = d_ovvv.reshape(-1,nvir,nvir,nvir)
        ccsd.unpack_tril(ccsd._cp(eris.ovvv[p0:p1]).reshape(-1,nvir_pair),
                         eris_ovvv[p0:p1].reshape(-1,nvir**2))
        #:Ivo += numpy.einsum('jadc,jidc->ai', d_ovvv, eris_oovv)
        for i in range(p1-p0):
            lib.dot(d_ovvv[i].reshape(nvir,-1),
                    eris_oovv[p0+i].reshape(nocc,-1).T, 1, Ivo, 1)

        for i in range(p1-p0):
            d_ovvv[i] += d_ovvv[i].transpose(0,2,1)
            tmp[i] = eris_ovov[p0+i].transpose(0,2,1)
        #:Ivo += numpy.einsum('jcab,jcib->ai', d_ovvv, eris_ovov)
        lib.dot(d_ovvv.reshape(-1,nvir).T,
                tmp[:p1-p0].reshape(-1,nocc), 1, Ivo, 1)
        #:Ivv += numpy.einsum('icdb,icda->ab', d_ovvv, eris_ovvv)
        #:Xvo += numpy.einsum('jibc,jabc->ai', d_oovv, eris_ovvv)
        lib.dot(eris_ovvv[p0:p1].reshape(-1,nvir).T,
                bufd_ovvv[:p1-p0].reshape(-1,nvir), 1, Ivv, 1)
        for i in range(p0, p1):
            lib.dot(eris_ovvv[i].reshape(nvir,-1),
                    d_oovv[i].reshape(nocc,-1).T, 1, Xvo, 1)
    eris_oovv = d_oovv = d_vvov = d_ovvv = bufd_ovvv = tmp = None
    Xvo +=(numpy.einsum('cb,iacb->ai', dvv, eris_ovvv) * 2
         + numpy.einsum('cb,iacb->ai', dvv, eris_ovvv) * 2
         - numpy.einsum('cb,icab->ai', dvv, eris_ovvv)
         - numpy.einsum('cb,ibca->ai', dvv, eris_ovvv))

    d_ovov = numpy.empty((nocc,nvir,nocc,nvir))
    for p0, p1 in ccsd.prange(0, nocc, blksize):
        d_ovov[p0:p1] = ccsd._cp(dovov[p0:p1])
        d_ovvo = ccsd._cp(dovvo[p0:p1])
        for i in range(p0,p1):
            d_ovov[i] += d_ovvo[i-p0].transpose(0,2,1)
    d_ovvo = None
    d_ovov = lib.transpose_sum(d_ovov.reshape(nov,nov)).reshape(nocc,nvir,nocc,nvir)
    #:Ivo += numpy.einsum('jbka,jbki->ai', d_ovov, eris.ovoo)
    Ivo += lib.dot(d_ovov.reshape(-1,nvir).T,
                   ccsd._cp(eris.ovoo).reshape(-1,nocc))
    #:Ioo += numpy.einsum('jakb,iakb->ij', d_ovov, eris.ovov)
    #:Ivv += numpy.einsum('jcib,jcia->ab', d_ovov, eris.ovov)
    Ioo += lib.dot(eris_ovov.reshape(nocc,-1), d_ovov.reshape(nocc,-1).T)
    Ivv += lib.dot(eris_ovov.reshape(-1,nvir).T, d_ovov.reshape(-1,nvir))
    eris_ovov = None

    #:d_vvvv = ccsd._cp(d_vvvv + d_vvvv.transpose(0,1,3,2))
    blksize = 12
    bufd_vvvv = numpy.empty((blksize,nvir,nvir,nvir))
    bufe1vvvv = numpy.empty((blksize,nvir,nvir_pair))
    bufe2vvvv = numpy.empty((blksize,nvir,nvir,nvir))
    bufd_vvvo = numpy.empty((blksize,nvir,nvir,nocc))
    bufe_vvov = numpy.empty((blksize,nvir,nov))
    bufe_vvvo = numpy.empty((blksize,nvir,nvir,nocc))
    for p0, p1 in ccsd.prange(0, nvir, blksize):
        d_vvvv = bufd_vvvv[:p1-p0]
        _sum0132(ccsd._cp(dvvvv[p0:p1]), d_vvvv)
        tmp1 = bufe1vvvv[:p1-p0]
        _load_block_tril(eris.vvvv, p0, p1, out=tmp1)
        eris_vvvv = bufe2vvvv[:p1-p0]
        ccsd.unpack_tril(tmp1.reshape(-1,nvir_pair), eris_vvvv.reshape(-1,nvir**2))
        #:Ivv += numpy.einsum('decb,deca->ab', d_vvvv, eris_vvvv) * 2
        #:Xvo += numpy.einsum('icdb,acdb->ai', d_ovvv, eris_vvvv)
        lib.dot(eris_vvvv.reshape(-1,nvir).T, d_vvvv.reshape(-1,nvir), 2, Ivv, 1)
        d_vvov = ccsd._cp(dvvov[p0:p1])
        d_vvvo = bufd_vvvo[:p1-p0]
        for i in range(p1-p0):
            d_vvvo[i] = d_vvov[i].transpose(0,2,1)
        lib.dot(eris_vvvv.reshape(-1,nvir).T, d_vvvo.reshape(-1,nocc), 1, Xvo, 1)

        eris_vvov = bufe_vvov[:p1-p0].reshape(-1,nov)
        tmp_ovvv = eris_ovvv.reshape(nov,-1)
        off = p0 * nvir
        for i0, i1 in ccsd.prange(0, (p1-p0)*nvir, BLKSIZE):
            for j0, j1, in ccsd.prange(0, nov, BLKSIZE):
                eris_vvov[i0:i1,j0:j1] = tmp_ovvv[j0:j1,off+i0:off+i1].T
        #:Ivv += numpy.einsum('dcib,dcia->ab', d_vvov, eris_vvov)
        #:Xvo += numpy.einsum('icjb,acjb->ai', d_ovov, eris_vvov)
        lib.dot(eris_vvov.reshape(-1,nvir).T, d_vvov.reshape(-1,nvir), 1, Ivv, 1)
        Xvo[p0:p1] += lib.dot(eris_vvov.reshape(p1-p0,-1), d_ovov.reshape(nocc,-1).T)

        eris_vvov = eris_vvov.reshape(p1-p0,nvir,nocc,nvir)
        eris_vvvo = bufe_vvvo[:p1-p0]
        for i in range(p1-p0):
            eris_vvvo[i] = eris_vvov[i].transpose(0,2,1)
        #:Ioo += numpy.einsum('abjc,abci->ij', d_vvov, eris_vvvo)
        #:Ivo += numpy.einsum('dbca,dbci->ai', d_vvvv, eris_vvvo) * 2
        lib.dot(d_vvvv.reshape(-1,nvir).T, eris_vvvo.reshape(-1,nocc), 2, Ivo, 1)
        lib.dot(eris_vvvo.reshape(-1,nocc).T, d_vvvo.reshape(-1,nocc), 1, Ioo, 1)

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
            eris_ovov = ccsd._cp(eris.ovov)
            v  = numpy.einsum('iajb,bj->ia', eris_ovov, x) * 4
            v -= numpy.einsum('ibja,bj->ia', eris_ovov, x)
            eris_ovov = None
            v -= numpy.einsum('jiab,bj->ia', ccsd._cp(eris.oovv), x)
        return v.T
    mo_energy = eris.fock.diagonal()
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[:nocc] = 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1


# Only works with canonical orbitals
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           grad_hf=None, verbose=logger.INFO):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = ccsd._ERIS(mycc)
    if grad_hf is None:
        grad_hf = pyscf.grad.hf.RHF(mycc._scf)

    log = logger.Logger(mycc.stdout, mycc.verbose)
    time0 = time.clock(), time.time()
    mol = mycc.mol
    moidx = numpy.ones(mycc.mo_energy.size, dtype=numpy.bool)
    if isinstance(mycc.frozen, (int, numpy.integer)):
        raise NotImplementedError('frozen orbital ccsd_grad')
        moidx[:mycc.frozen] = False
    else:
        moidx[mycc.frozen] = False
    mo_coeff = mycc.mo_coeff[:,moidx]  #FIXME: ensure mycc.mo_coeff is canonical orbital
    mo_energy = mycc.mo_energy[moidx]
    nocc, nvir = t1.shape
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    log.debug('Build ccsd rdm1 intermediates')
    doo, dvv = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    time1 = log.timer('rdm1 intermediates', *time0)

    log.debug('Build ccsd rdm2 intermediates')
    d2 = ccsd_rdm.gamma2_incore(mycc, t1, t2, l1, l2)
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    time1 = log.timer('rdm2 intermediates', *time1)
    log.debug('Build ccsd response_rdm1')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris, (doo,dvv), d2)
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
    dm2ao = _rdm2_mo2ao(mycc, d2, dm1mo, mo_coeff)
    time1 = log.timer('MO->AO transformation', *time1)

#TODO: pass hf_grad object to compute h1 and s1
    log.debug('h1 and JK1')
    h1 = grad_hf.get_hcore(mol)
    s1 = grad_hf.get_ovlp(mol)
    zeta = numpy.empty((nmo,nmo))
    zeta[:nocc,:nocc] = (mo_energy[:nocc].reshape(-1,1) + mo_energy[:nocc]) * .5
    zeta[nocc:,nocc:] = (mo_energy[nocc:].reshape(-1,1) + mo_energy[nocc:]) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf4sij = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1ao+dm1ao.T), p1))
    time1 = log.timer('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    dm1ao += hf_dm1
    zeta += grad_hf.make_rdm1e(mycc.mo_energy, mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = grad_hf.aorange_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] =(numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
              + numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1]))
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        vrinv = grad_hf._grad_rinv(mol, ia)
        de[k] +=(numpy.einsum('xij,ij->x', h1[:,p0:p1], dm1ao[p0:p1]  )
               + numpy.einsum('xji,ij->x', h1[:,p0:p1], dm1ao[:,p0:p1]))
        de[k] +=(numpy.einsum('xij,ij->x', vrinv, dm1ao)
               + numpy.einsum('xji,ij->x', vrinv, dm1ao))
# -s[1]*e \dot DM,  contribute to f1
        de[k] -=(numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
               + numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1]))
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf4sij[p0:p1]) * 2

# 2e AO integrals dot 2pdm
        eri1 = gto.moleintor.getints('cint2e_ip1_sph', mol._atm, mol._bas,
                                     mol._env, numpy.arange(shl0,shl1), comp=3,
                                     aosym='s2kl').reshape(3,p1-p0,nao,-1)
        dm2buf = _load_block_tril(dm2ao, p0, p1)
        de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2

        for i in range(3):
            #:tmp = ccsd.unpack_tril(eri1[i].reshape(-1,nao_pair))
            #:vj = numpy.einsum('ijkl,kl->ij', tmp, hf_dm1)
            #:vk = numpy.einsum('ijkl,jk->il', tmp, hf_dm1)
            vj, vk = hf_get_jk_incore(eri1[i], hf_dm1)
            de[k,i] -=(numpy.einsum('ij,ij->', vj, hf_dm1[p0:p1])
                     - numpy.einsum('ij,ij->', vk, hf_dm1[p0:p1])*.5) * 2
        eri1 = dm2buf = None
        log.debug('grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer('grad of atom %d'%ia, *time1)

    log.note('CCSD gradinets')
    log.note('==============')
    log.note('           x                y                z')
    for k, ia in enumerate(atmlst):
        log.note('%d %s  %15.9f  %15.9f  %15.9f', ia, mol.atom_symbol(ia),
                 de[k,0], de[k,1], de[k,2])
    log.timer('CCSD gradients', *time0)
    return de

def shell_prange(mol, start, stop, blksize):
    nao = 0
    ib0 = start
    for ib in range(start, stop):
        now = (mol.bas_angular(ib)*2+1) * mol.bas_nctr(ib)
        nao += now
        if nao > blksize and nao > now:
            yield (ib0, ib, nao-now)
            ib0 = ib
            nao = now
    yield (ib0, stop, nao)

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
    def _trans(vin, i0, icount, j0, jcount, out=None):
        nrow = vin.shape[0]
        fdrv = getattr(ccsd.libcc, 'AO2MOnr_e2_drv')
        pao_loc = ctypes.POINTER(ctypes.c_void_p)()
        ftrans = ctypes.c_void_p(_ctypes.dlsym(ccsd.libcc._handle, 'AO2MOtranse2_nr_s1'))
        fmmm = ctypes.c_void_p(_ctypes.dlsym(ccsd.libcc._handle, 'CCmmm_transpose_sum'))
        fdrv(ftrans, fmmm,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             ctypes.c_int(i0), ctypes.c_int(icount),
             ctypes.c_int(j0), ctypes.c_int(jcount),
             pao_loc, ctypes.c_int(0))
        return out

    pool = numpy.empty(max(nvir_pair*nvir**2, nocc*nmo**3))
    buf1 = pool[:nocc*nmo**3].reshape((nocc,nmo,nmo,nmo))
    buf1[:,:nocc,:nocc,:nocc] = doooo
    buf1[:,:nocc,:nocc,nocc:] = dooov
    buf1[:,:nocc,nocc:,:nocc] = 0
    buf1[:,:nocc,nocc:,nocc:] = doovv
    buf1[:,nocc:,:nocc,:nocc] = 0
    buf1[:,nocc:,:nocc,nocc:] = dovov
    buf1[:,nocc:,nocc:,:nocc] = dovvo
    buf1[:,nocc:,nocc:,nocc:] = dovvv
    for i in range(nocc):
        buf1[i,i,:,:] += dm1
        buf1[i,:,:,i] -= dm1 * .5
    bufop = numpy.empty((nocc*nmo,nao_pair))
    _trans(buf1.reshape(-1,nmo**2), 0, nmo, 0, nmo, bufop)
    buf1 = pool[:nvir_pair*nvir**2].reshape((nvir_pair,nvir,nvir))
    p0 = 0
    for i in range(nvir):
        buf1[p0:p0+i+1]  = dvvvv[i,:i+1]
        buf1[p0:p0+i+1] += dvvvv[:i+1,i]
        buf1[p0:p0+i+1] *= .5
        p0 += i + 1
    bufvv = numpy.empty((nvir_pair,nao_pair))
    _trans(buf1, nocc, nvir, nocc, nvir, bufvv)
    buf1 = buf0 = None

    # trans ij of d_ijkl
    bufaa = numpy.empty((nao_pair,nao_pair))
    tmpbuf1 = numpy.empty((BLKSIZE,nmo,nmo))
    tmpbuf2 = numpy.empty((BLKSIZE,nvir_pair))
    tmpbuf3 = numpy.empty((BLKSIZE,nvir,nvir))
    tmpbuf1[:,nocc:,:nocc] = 0
    for i0, i1 in ccsd.prange(0, nao_pair, BLKSIZE):
        buf1 = tmpbuf1[:i1-i0].reshape(i1-i0,-1)
        for j0, j1 in ccsd.prange(0, nocc*nmo, BLKSIZE):
            buf1[:,j0:j1] = bufop[j0:j1,i0:i1].T
        buf1 = tmpbuf1[:i1-i0]
        tmp  = tmpbuf2[:i1-i0]
        for j0, j1 in ccsd.prange(0, nvir_pair, BLKSIZE):
            tmp[:,j0:j1] = bufvv[j0:j1,i0:i1].T
        tmp1 = tmpbuf3[:i1-i0]
        ccsd.unpack_tril(tmp, out=tmp1)
        buf1[:,nocc:,nocc:] = tmp1
        _trans(buf1, 0, nmo, 0, nmo, bufaa[i0:i1])
# dm2 + dm2.transpose(2,3,0,1)
        for j0, j1, in ccsd.prange(0, i0, BLKSIZE):
            bufaa[i0:i1,j0:j1] += bufaa[j0:j1,i0:i1].T
            bufaa[j0:j1,i0:i1] = bufaa[i0:i1,j0:j1].T
        bufaa[i0:i1,i0:i1] = bufaa[i0:i1,i0:i1] + bufaa[i0:i1,i0:i1].T
    tmpbuf1 = tmpbuf2 = tmpbuf3 = bufop = bufvv = pool = None

    idx = numpy.arange(nao)
    idx = idx*(idx+1)//2 + idx
    bufaa[:,idx] *= .5

    return bufaa

def _load_block_tril(dat, row0, row1, out=None):
    shape = dat.shape
    nd = int(numpy.sqrt(shape[0]*2))
    if out is None:
        out = numpy.empty((row1-row0,nd)+shape[1:])
    p0 = row0*(row0+1)//2
    for i in range(row0, row1):
        out[i-row0,:i+1] = ccsd._cp(dat[p0:p0+i+1])
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
        p0 += i + 1
    for i in range(row1, nd):
        i2 = i*(i+1)//2
        out[:,i] = dat[i2+row0:i2+row1]
    return out


def _sum0132(a, out=None):
    assert(a.flags.c_contiguous)
    if out is None:
        out = numpy.empty_like(a)
    ccsd.libcc.CCsum021(out.ctypes.data_as(ctypes.c_void_p),
                        a.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(a.shape[0]*a.shape[1]),
                        ctypes.c_int(a.shape[2]))
    return out

def hf_get_jk_incore(eri, dm):
    ni, nj = eri.shape[:2]
    vj = numpy.empty((ni,nj))
    vk = numpy.empty((ni,nj))
    ccsd.libcc.CCvhfs2kl(eri.ctypes.data_as(ctypes.c_void_p),
                         dm.ctypes.data_as(ctypes.c_void_p),
                         vj.ctypes.data_as(ctypes.c_void_p),
                         vk.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(ni), ctypes.c_int(nj))
    return vj, vk


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
    #dm1 = response_dm1(mycc, t1, t2, l1, l2, eris)
    #print(numpy.einsum('pq,pq', h1[nocc:,:nocc], dm1[nocc:,:nocc])--486.638981725713393)

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
    l1, l2 = mycc.solve_lambda()[1:]
    g1 = kernel(mycc, t1, t2, l1, l2, grad_hf=grad.hf.RHF(mf))
    print('gcc')
    print(g1 + grad.hf.grad_nuc(mol))
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
    ghf = grad.hf.RHF(mf).grad()
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()[1:]
    g1 = kernel(mycc, t1, t2, l1, l2, grad_hf=grad.hf.RHF(mf))
    print('gcc')
    print(g1 + grad.hf.grad_nuc(mol))
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]
