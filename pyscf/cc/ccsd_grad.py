#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD analytical nuclear gradients
'''

import time
import ctypes
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf import gto
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm
from pyscf.scf import rhf_grad
from pyscf.scf import cphf


def IX_intermediates(mycc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
        eris = mycc.ao2mo()
    if d1 is None:
        d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    if d2 is None:
        fd2intermediate = lib.H5TmpFile()
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
    dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo = d2

    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir * (nvir+1) //2
    with_frozen = not (mycc.frozen is None or mycc.frozen is 0)
    fswap = lib.H5TmpFile()

    d_vovo = _cp(dvovo) + _cp(dvoov).transpose(0,1,3,2)
    d_vovo = lib.transpose_sum(d_vovo.reshape(nov,nov)).reshape(nvir,nocc,nvir,nocc)
    fswap['d_voov'] = d_vovo.transpose(0,1,3,2)
    d_vovo = None

    d_vvoo = _cp(dvvoo)
    d_vvoo = d_vvoo + d_vvoo.transpose(1,0,3,2)
    fswap['d_vvoo'] = d_vvoo
    d_woo = d_vvoo[numpy.tril_indices(nvir)]
    for i in range(1, nvir):
        off = i*(i+1)//2
        d_woo[off:off+i] += d_vvoo[:i,i]
    fswap['d_woo'] = d_woo
    d_vvoo = None

    doo = doo + doo.T
    dvv = dvv + dvv.T

    if with_frozen:
        erif = _make_frozen_orbital_eris(mycc)
        OA, VA, OF, VF = index_frozen_active(mycc)
        n_OF = len(OF)
        n_VF = len(VF)
        nfrozen = n_OF + n_VF
        nactive = len(OA) + len(VA)
        Ifa = numpy.empty((nfrozen,nactive))

# Note Ioo, Ivv are not hermitian
    eris_oooo = _cp(eris.oooo)
    eris_vooo = _cp(_cp(eris.ovoo).transpose(1,0,2,3))
    d_oooo = _cp(doooo)
    d_oooo = _cp(d_oooo + d_oooo.transpose(1,0,2,3))
    Ioo = lib.einsum('imlk,jmlk->ij', eris_oooo, d_oooo) * 2
    Xvo = lib.einsum('aklj,iklj->ai', eris_vooo, d_oooo) * 2
    if with_frozen:
        Ifa[:,:nocc] = lib.einsum('aklj,iklj->ai', erif.fooo, d_oooo) * 2
    else:
        Xvo += numpy.einsum('kj,aikj->ai', doo, eris_vooo) * 2
        Xvo -= numpy.einsum('kj,ajik->ai', doo, eris_vooo)

    d_vooo = _cp(dvooo)
    Ivv = lib.einsum('bkij,akij->ab', d_vooo, eris_vooo)
    Ivo = lib.einsum('akjl,ikjl->ai', d_vooo, eris_oooo)
    if with_frozen:
        Ifa[:,nocc:] = lib.einsum('akij,pkij->pa', d_vooo, erif.fooo)
    eris_oooo = eris_vooo = d_oooo = d_vooo = None

    eris_oovv_tril = _cp(eris.oovv).reshape(nocc**2,nvir,nvir)
    eris_oovv_tril = lib.pack_tril(eris_oovv_tril).reshape(nocc,nocc,-1)
    fswap['vvoo'] = eris_oovv_tril.transpose(2,0,1)
    eris_oovv_tril = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2 * 4
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/unit)))
    log.debug1('IX_intermediates pass 1: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    fswap.create_dataset('eris_wvo', (nvir_pair,nvir,nocc), 'f8',
                         chunks=(nvir_pair,blksize,nocc))
    fswap.create_dataset('d_wvo', (nvir_pair,nvir,nocc), 'f8',
                         chunks=(nvir_pair,blksize,nocc))

    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        eris_vvoo = _cp(eris.oovv[:,:,p0:p1].transpose(2,3,0,1))

        d_vooo = _cp(dvooo[p0:p1])
        Ioo += lib.einsum('ajkl,iakl->ij', d_vooo, eris_ovoo)
        Xvo += lib.einsum('bikj,bakj->ai', d_vooo, eris_vvoo)
        if with_frozen:
            fvoo = erif.fvoo[:,p0:p1]
            Ifa[:,:nocc] += lib.einsum('bjkl,pbkl->pj', d_vooo, fvoo)

        d_vooo = d_vooo + d_vooo.transpose(0,1,3,2)
        eris_voov = _cp(eris.ovvo[:,p0:p1].transpose(1,0,3,2))
        Ioo += lib.einsum('aklj,kali->ij', d_vooo, eris_ovoo)
        Xvo += lib.einsum('bkji,bkja->ai', d_vooo, eris_voov)

        d_vvoo = _cp(fswap['d_vvoo'][p0:p1])
        Ivv += lib.einsum('cbij,caij->ab', d_vvoo, eris_vvoo)
        Ivo += lib.einsum('bakj,ibkj->ai', d_vvoo, eris_ovoo)
        if with_frozen:
            voof = erif.voof[p0:p1]
            Ifa[:,:nocc] += lib.einsum('aklj,aklp->pj', d_vooo, voof)
            Ifa[:,nocc:] += lib.einsum('cbij,pcij->pb', d_vvoo, fvoo)
            fvoo = None
        d_vvoo = None

        d_voov = _cp(fswap['d_voov'][p0:p1])
        Ivo += lib.einsum('bjka,jbki->ai', d_voov, eris_ovoo)
        Ioo += lib.einsum('akjb,akib->ij', d_voov, eris_voov)
        Ivv += lib.einsum('cjib,cjia->ab', d_voov, eris_voov)
        if with_frozen:
            Ifa[:,nocc:] += lib.einsum('cjkb,cjkp->pb', d_voov, voof)
            voof = None
        eris_vvoo = eris_ovoo = None

        # tril part of (d_vovv + d_vovv.transpose(0,1,3,2))
        d_vovv = _cp(dvovv[p0:p1])
        c_vovv = _ccsd.precontract(d_vovv.reshape(-1,nvir,nvir))
        c_vovv = c_vovv.reshape(p1-p0,nocc,nvir_pair)
        fswap['d_wvo'][:,p0:p1] = c_vovv.transpose(2,0,1)
        c_vovv = None

        eris_vox = _cp(eris.ovvv[:,p0:p1]).transpose(1,0,2)
        fswap['eris_wvo'][:,p0:p1] = eris_vox.reshape(p1-p0,nocc,nvir_pair).transpose(2,0,1)

        d_vovv = d_vovv + d_vovv.transpose(0,1,3,2)
        Ivo += lib.einsum('cjba,cjib->ai', d_vovv, eris_voov)
        eris_voov = None
        if with_frozen:
            vovf = erif.vovf[p0:p1]
            Ifa[:,:nocc] += lib.einsum('ckjb,ckbp->pj', d_voov, vovf)
            Ifa[:,nocc:] += lib.einsum('cjda,cjdp->pa', d_vovv, vovf)
            vovf = None

        eris_vovv = lib.unpack_tril(eris_vox.reshape(-1,nvir_pair))
        eris_vox = None
        eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)
        Ivv += lib.einsum('cidb,cida->ab', d_vovv, eris_vovv)
        Xvo += lib.einsum('bjic,bjca->ai', d_voov, eris_vovv)
        if not with_frozen:
            Xvo[p0:p1] += numpy.einsum('cb,aicb->ai', dvv, eris_vovv) * 2
            Xvo -= numpy.einsum('cb,ciba->ai', dvv[p0:p1], eris_vovv)
        d_vovv = d_voov = eris_vovv = None
    eris_oovv = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2 + nvir**3*2.5
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/unit)))
    log.debug1('IX_intermediates pass 2: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    for p0, p1 in lib.prange(0, nvir, blksize):
        off0 = p0*(p0+1)//2
        off1 = p1*(p1+1)//2
        d_vvvv = _cp(dvvvv[off0:off1]) * 4
        for i in range(p0, p1):
            d_vvvv[i*(i+1)//2+i-off0] *= .5
        d_vvvv = lib.unpack_tril(d_vvvv)
        eris_vvvv = lib.unpack_tril(_cp(eris.vvvv[off0:off1]))
        Ivv += lib.einsum('xcb,xca->ab', d_vvvv, eris_vvvv) * 2

        d_vvvo = _cp(fswap['d_wvo'][off0:off1])
        Xvo += lib.einsum('xci,xca->ai', d_vvvo, eris_vvvv)
        eris_vvoo = _cp(fswap['vvoo'][off0:off1])
        Ivo += lib.einsum('xaj,xji->ai', d_vvvo, eris_vvoo)
        eris_vvvv = None

        eris_vvvo = _cp(fswap['eris_wvo'][off0:off1])
        Ivo += lib.einsum('xca,xci->ai', d_vvvv, eris_vvvo) * 2
        if with_frozen:
            vvvf = erif.vvvf[off0:off1]
            Ifa[:,nocc:] += lib.einsum('xca,xcp->pa', d_vvvv, vvvf) * 2
        d_vvvv = None
        Ioo += lib.einsum('xcj,xci->ij', d_vvvo, eris_vvvo)
        Ivv += lib.einsum('xbi,xai->ab', d_vvvo, eris_vvvo)

        d_vvoo = _cp(fswap['d_woo'][off0:off1])
        Ioo += lib.einsum('xkj,xki->ij', d_vvoo, eris_vvoo)
        Xvo += lib.einsum('xji,xaj->ai', d_vvoo, eris_vvvo)
        if with_frozen:
            vvof = erif.vvof[off0:off1]
            Ifa[:,:nocc] += lib.einsum('xki,xkp->pi', d_vvoo, vvof)
            Ifa[:,nocc:] += lib.einsum('xaj,xjp->pa', d_vvvo, vvof)
            Ifa[:,:nocc] += lib.einsum('xcj,xcp->pj', d_vvvo, vvvf)
            vvvf = vvof = None
        eris_vvvo = d_vvvo = d_vvoo = eris_vvoo = None

    Ioo *= -1
    Ivv *= -1
    Ivo *= -1

    if with_frozen:
        Ico = Ifa[:n_OF,:nocc] * -1
        Ifv = Ifa[n_OF:,nocc:] * -1
        Ivc = Ifa[:n_OF,nocc:].T * -1
        Xfo = Ifa[n_OF:,:nocc]

        mo_coeff = mycc.mo_coeff
        nao, nmo = mo_coeff.shape
        nocc = len(OA) + len(OF)
        nvir = len(VA) + len(VF)

        Ioo1 = numpy.zeros((nocc,nocc))
        Ivo1 = numpy.zeros((nvir,nocc))
        Ivv1 = numpy.zeros((nvir,nvir))
        Ioo1[OA[:,None],OA] = Ioo
        Ioo1[OF[:,None],OA] = Ico
        Ivo1[VA[:,None],OA] = Ivo
        Ivo1[VA[:,None],OF] = Ivc
        Ivv1[VA[:,None],VA] = Ivv
        Ivv1[VF[:,None],VA] = Ifv

        mo_e_o = mycc._scf.mo_energy[mycc.mo_occ > 0]
        mo_e_v = mycc._scf.mo_energy[mycc.mo_occ ==0]
        dco = Ico / lib.direct_sum('i-j->ij', mo_e_o[OF], mo_e_o[OA])
        dfv = Ifv / lib.direct_sum('a-b->ab', mo_e_v[VF], mo_e_v[VA])
        dm1 = numpy.zeros((nmo,nmo))
        dm1[OA[:,None],OA] = doo
        dm1[OF[:,None],OA] = dco
        dm1[OA[:,None],OF] = dco.T
        dm1[VA[:,None]+nocc,VA+nocc] = dvv
        dm1[VF[:,None]+nocc,VA+nocc] = dfv
        dm1[VA[:,None]+nocc,VF+nocc] = dfv.T
        dm1 = reduce(numpy.dot, (mo_coeff, dm1, mo_coeff.T))
        vj, vk = mycc._scf.get_jk(mycc.mol, dm1)
        Xvo1 = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vj*2-vk, mo_coeff[:,:nocc]))
        Xvo1[VA[:,None],OA] += Xvo
        Xvo1[VF[:,None],OA] += Xfo
        Ioo, Ivo, Ivv, Xvo = Ioo1, Ivo1, Ivv1, Xvo1

    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


def response_dm1(mycc, t1, t2, l1, l2, eris=None, IX=None):
    if eris is None:
        eris = mycc.ao2mo()
    if IX is None:
        IX = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    Ioo, Ivv, Ivo, Xvo = IX
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    with_frozen = not (mycc.frozen is None or mycc.frozen is 0)
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc._scf.mo_occ
        def fvind(x):
            mo_coeff = mycc.mo_coeff
            x = x.reshape(Xvo.shape)
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, v, mo_coeff[:,:nocc]))
            return v * 2
    else:
        mo_energy = eris.fock.diagonal()
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = numpy.empty((nocc,nvir,nvir,nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1,0,2)
            ovvo[i]-= eris.oovv[i].transpose(2,1,0)
        def fvind(x):
            return numpy.einsum('iabj,bj->ai', ovvo, x.reshape(Xvo.shape))
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
           mf_grad=None, d1=None, d2=None, verbose=logger.INFO):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = mycc.ao2mo()
    if mf_grad is None:
        mf_grad = rhf_grad.Gradients(mycc._scf)

    log = logger.new_logger(mycc, verbose)
    time0 = time.clock(), time.time()
    if mycc.direct:
        raise NotImplementedError('AO-direct CCSD gradients')
    if abs(eris.fock - numpy.diag(eris.fock.diagonal())).max() > 1e-3:
        raise RuntimeError('CCSD gradients does not support NHF (non-canonical HF)')

    mol = mycc.mol
    with_frozen = not (mycc.frozen is None or mycc.frozen is 0)
    if with_frozen:
        mo_coeff = mycc.mo_coeff
        mo_energy = mycc._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = numpy.count_nonzero(mycc.mo_occ > 0)
        nvir = nmo - nocc
        mo_e_o = mo_energy[mycc.mo_occ > 0]
        mo_e_v = mo_energy[mycc.mo_occ ==0]
    else:
        mo_coeff = eris.mo_coeff
        mo_energy = eris.fock.diagonal()
        nao, nmo = mo_coeff.shape
        nocc, nvir = t1.shape
        mo_e_o = mo_energy[:nocc]
        mo_e_v = mo_energy[nocc:]
    nao_pair = nao * (nao+1) // 2

    log.debug('Build ccsd rdm1 intermediates')
    if d1 is None:
        d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)

    log.debug('Build ccsd rdm2 intermediates')
    if d2 is None:
        fd2intermediate = lib.H5TmpFile()
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
    time1 = log.timer_debug1('rdm2 intermediates', *time1)
    log.debug('Build ccsd response_rdm1')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris, d1, d2)
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    dm1mo = response_dm1(mycc, t1, t2, l1, l2, eris, (Ioo, Ivv, Ivo, Xvo))
    if with_frozen:
        OA, VA, OF, VF = index_frozen_active(mycc)
        dco = Ioo[OF[:,None],OA] / lib.direct_sum('i-j->ij', mo_e_o[OF], mo_e_o[OA])
        dfv = Ivv[VF[:,None],VA] / lib.direct_sum('a-b->ab', mo_e_v[VF], mo_e_v[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None]+nocc,VA+nocc] = dvv + dvv.T
        dm1mo[VF[:,None]+nocc,VA+nocc] = dfv
        dm1mo[VA[:,None]+nocc,VF+nocc] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T
    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))

    im1 = numpy.zeros_like(dm1mo)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T
    im1 = reduce(numpy.dot, (mo_coeff, im1, mo_coeff.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('symmetrized rdm2 and MO->AO transformation')
# Roughly, dm2*2 is computed in _rdm2_mo2ao
    fdm2 = lib.H5TmpFile()
    _rdm2_mo2ao(mycc, d2, eris.mo_coeff, fdm2)  # transform the active orbitals
    time1 = log.timer_debug1('MO->AO transformation', *time1)

    log.debug('h1 and JK1')
    h1 = mf_grad.get_hcore(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf4sij = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1+dm1.T), p1))
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    hf_dm1 = mycc._scf.make_rdm1(mycc._scf.mo_coeff, mycc._scf.mo_occ)
    dm1p = hf_dm1 + dm1*2
    dm1 += hf_dm1
    zeta += mf_grad.make_rdm1e(mycc._scf.mo_energy, mycc._scf.mo_coeff,
                               mycc._scf.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*1e6/8/(nao**3*2.5)))
    ioblksize = fdm2['dm2'].chunks[1]
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k]  = numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        h1ao = mf_grad._grad_rinv(mol, ia)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1)
        de[k] += numpy.einsum('xji,ij->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf4sij[p0:p1]) * 2

# 2e AO integrals dot 2pdm
        ip1 = p0
        for b0, b1, nf in shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=(b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas))
            eri1 = eri1.reshape(3,nf,nao,-1)
            dm2buf = numpy.empty((nf,nao,nao_pair))
            for i0, i1 in lib.prange(0, nao_pair, ioblksize):
                _load_block_tril(fdm2['dm2'], ip0, ip1, i0, i1, dm2buf[:,:,i0:i1])
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
            dm2buf = None
# HF part
            eri1 = lib.unpack_tril(eri1.reshape(3*nf*nao,-1)).reshape(3,nf,nao,nao,nao)
            de[k] -= numpy.einsum('xijkl,ij,kl->x', eri1, hf_dm1[ip0:ip1], dm1p)
            de[k] -= numpy.einsum('xijkl,ij,kl->x', eri1, dm1p[ip0:ip1], hf_dm1)
            de[k] += numpy.einsum('xijkl,jk,il->x', eri1, hf_dm1, dm1p[ip0:ip1]) * .5
            de[k] += numpy.einsum('xijkl,jk,il->x', eri1, dm1p, hf_dm1[ip0:ip1]) * .5
            eri1 = None
        log.debug('grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1('grad of atom %d'%ia, *time1)

    de += rhf_grad.grad_nuc(mol)
    log.timer('CCSD gradients', *time0)
    return de


def as_scanner(grad_cc):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CCSD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CCSD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, cc
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> cc_scanner = cc.CCSD(scf.RHF(mol)).as_scanner()
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    import copy
    logger.info(grad_cc, 'Set nuclear gradients of %s as a scanner', grad_cc.__class__)
    class CCSD_GradScanner(grad_cc.__class__, lib.GradScanner):
        def __init__(self, g):
            self.__dict__.update(g.__dict__)
            self._cc = grad_cc._cc.as_scanner()
        def __call__(self, mol):
            # The following simple version also works.  But eris object is
            # recomputed in cc_scanner and solve_lambda.
            # cc_scanner = self._cc
            # cc_scanner(mol)
            # eris = cc_scanner.ao2mo()
            # cc_scanner.solve_lambda(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris)
            # mf_grad = mf_scanner.nuc_grad_method()
            # de = self.kernel(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris, mf_grad=mf_grad)

            cc = self._cc
            mf_scanner = cc._scf
            mf_scanner(mol)
            cc.mol = mol
            cc.mo_coeff = mf_scanner.mo_coeff
            cc.mo_occ = mf_scanner.mo_occ
            eris = cc.ao2mo(cc.mo_coeff)
            cc.kernel(cc.t1, cc.t2, eris=eris)
            cc.solve_lambda(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris)
            mf_grad = mf_scanner.nuc_grad_method()
            de = self.kernel(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris, mf_grad=mf_grad)
            return cc.e_tot, de
        @property
        def converged(self):
            cc = self._cc
            return all((cc._scf.converged, cc.converged, cc.converged_lambda))
    return CCSD_GradScanner(grad_cc)


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

def _rdm2_mo2ao(mycc, d2, mo_coeff, fsave=None):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = time.clock(), time.time()
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo = d2

    nvir, nocc = dvovo.shape[:2]
    nov = nocc * nvir
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    nvir_pair = nvir * (nvir+1) //2

    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = numpy.empty((nrow,nao_pair))
        fdrv = getattr(_ccsd.libcc, 'AO2MOnr_e2_drv')
        pao_loc = ctypes.POINTER(ctypes.c_void_p)()
        fdrv(_ccsd.libcc.AO2MOtranse2_nr_s1, _ccsd.libcc.CCmmm_transpose_sum,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    fswap = lib.H5TmpFile()
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nvir_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('v', (nao_pair,nvir_pair), 'f8', chunks=(nao_pair,blksize))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        fswap['v'][:,p0:p1] = _trans(lib.unpack_tril(_cp(dvvvv[p0:p1])),
                                     (nocc,nmo,nocc,nmo)).T
    time1 = log.timer_debug1('_rdm2_mo2ao pass 1', *time1)

# transform dm2_ij to get lower triangular (dm2+dm2.transpose(0,1,3,2))
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('o', (nmo,nocc,nao_pair), 'f8', chunks=(nmo,nocc,blksize))
    buf1 = numpy.zeros((nocc,nocc,nmo,nmo))
    buf1[:,:,:nocc,:nocc] = doooo
    buf1[:,:,nocc:,nocc:] = _cp(dvvoo).transpose(2,3,0,1)
    buf1 = _trans(buf1.reshape(nocc**2,-1), (0,nmo,0,nmo))
    fswap['o'][:nocc] = buf1.reshape(nocc,nocc,nao_pair)
    for p0, p1 in lib.prange(nocc, nmo, nocc):
        buf1 = numpy.zeros((p1-p0,nocc,nmo,nmo))
        buf1[:,:,:nocc,:nocc] = dvooo[p0-nocc:p1-nocc]
        buf1[:,:,:nocc,nocc:] = dvoov[p0-nocc:p1-nocc]
        buf1[:,:,nocc:,:nocc] = dvovo[p0-nocc:p1-nocc]
        buf1[:,:,nocc:,nocc:] = dvovv[p0-nocc:p1-nocc]
        buf1 = _trans(buf1.reshape((p1-p0)*nocc,-1), (0,nmo,0,nmo))
        fswap['o'][p0:p1] = buf1.reshape(p1-p0,nocc,nao_pair)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 2', *time1)

# transform dm2_kl then dm2 + dm2.transpose(2,3,0,1)
    gsave = fsave.create_dataset('dm2', (nao_pair,nao_pair), 'f8', chunks=(nao_pair,blksize))
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmo,nmo))
        buf1[:,nocc:,nocc:] = lib.unpack_tril(_cp(fswap['v'][p0:p1]))
        buf1[:,:,:nocc] = fswap['o'][:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, (0,nmo,0,nmo))
        ic = 0
        idx = diagidx[diagidx<p1]
        if p0 > 0:
            buf1 = _cp(gsave[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            buf1[:,idx[p0<=idx]-p0] *= .5
            gsave[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        buf2[:,idx] *= .5
        gsave[p0:p1] = buf2
    time1 = log.timer_debug1('_rdm2_mo2ao pass 3', *time1)
    if incore:
        return fsave['dm2'].value
    else:
        return fsave

#
# .
# . .
# ----+             -----------
# ----|-+       =>  -----------
# . . | | .
# . . | | . .
#
def _load_block_tril(dat, row0, row1, col0, col1, out=None):
    shape = dat.shape
    nd = int(numpy.sqrt(shape[0]*2))
    if out is None:
        out = numpy.empty((row1-row0,nd,col1-col0)+shape[2:])
    dat1 = dat[row0*(row0+1)//2:row1*(row1+1)//2,col0:col1]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i+1
        out[i-row0,:i+1] = dat1[p0:p1]
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
    for i in range(row1, nd):
        i2 = i*(i+1)//2
        out[:,i] = dat[i2+row0:i2+row1,col0:col1]
    return out

def _cp(a):
    return numpy.array(a, copy=False, order='C')

def _make_frozen_orbital_eris(mycc):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    cput0 = time.clock(), time.time()
    moidx = ccsd.get_moidx(mycc)
    mo_frozen = numpy.asarray(mycc.mo_coeff[:,~moidx], order='F')
    mo_active = numpy.asarray(mycc.mo_coeff[:, moidx], order='F')
    nao, nmo = mo_active.shape
    nocc = mycc.nocc
    nvir = nmo - nocc
    nao_pair = nao * (nao+1) // 2
    nmo_pair = nmo * (nmo+1) // 2
    nvir_pair = nvir * (nvir+1) // 2
    n_OF = numpy.count_nonzero(mycc.mo_occ > 0) - nocc
    n_VF = mo_frozen.shape[1] - n_OF
    nfrozen = mo_frozen.shape[1]
    eris = ccsd._ChemistsERIs()

    mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mycc._scf._eri is not None and
        (mem_incore+mem_now < mycc.max_memory) or mycc.mol.incore_anyway):

        eri0 = ao2mo.kernel(mycc._scf._eri, (mo_frozen, mo_active,
                                             mo_active, mo_active))
        eri0 = lib.unpack_tril(eri0).reshape(nfrozen,nmo,nmo,nmo)
        eris.fooo = eri0[:,:nocc,:nocc,:nocc].copy()
        eris.fvoo = eri0[:,nocc:,:nocc,:nocc].copy()
        eris.voof = eri0[:,:nocc,nocc:,:nocc].transpose(2,3,1,0).copy()
        eris.vovf = eri0[:,nocc:,nocc:,:nocc].transpose(2,3,1,0).copy()
        vidx = numpy.tril_indices(nvir)
        eris.vvof = eri0[:,:nocc,nocc:,nocc:].transpose(2,3,1,0)[vidx]
        eris.vvvf = eri0[:,nocc:,nocc:,nocc:].transpose(2,3,1,0)[vidx]

    elif hasattr(mycc._scf, 'with_df'):
        raise NotImplementedError('DF-CCSD gradients')

    else:
        mol = mycc.mol
        fswap = lib.H5TmpFile()
        max_memory = max(2000, mycc.max_memory-mem_now)
        int2e = mol._add_suffix('int2e')
        ao2mo.outcore.half_e1(mol, (mo_active,mo_frozen), fswap, int2e,
                              's4', 1, max_memory, verbose=log)
        tril2sq = lib.square_mat_in_trilu_indices(nmo)
        ooidx = tril2sq[:nocc,:nocc]
        voidx = tril2sq[nocc:,:nocc]
        vvidx = tril2sq[nocc:,nocc:][numpy.tril_indices(nvir)]
        ao_loc = mol.ao_loc_nr()

        max_memory = max(2000, mycc.max_memory-mem_now)
        unit = n_OF * (nao_pair + nmo_pair)
        blksize = min(nocc, max(4, int(max_memory*.9e6/8/unit)))
        eris.feri = lib.H5TmpFile()
        eris.fooo = eris.feri.create_dataset('fooo', (nfrozen,nocc,nocc,nocc), 'f8')
        eris.voof = eris.feri.create_dataset('voof', (nvir,nocc,nocc,nfrozen), 'f8',
                                             chunks=(nvir,nocc,blksize,nfrozen))
        eris.vvof = eris.feri.create_dataset('vvof', (nvir_pair,nocc,nfrozen), 'f8',
                                             chunks=(nvir_pair,blksize,nfrozen))
        fload = ao2mo.outcore._load_from_h5g
        for p0, p1 in lib.prange(0, nocc, blksize):
            dat = numpy.empty(((p1-p0)*nfrozen,nao_pair))
            fload(fswap['0'], p0*nfrozen, p1*nfrozen, dat)
            dat = ao2mo._ao2mo.nr_e2(dat, mo_active, (0,nmo,0,nmo),
                                     's4', 's2', ao_loc=ao_loc)
            dat = dat.reshape(p1-p0,nfrozen,nmo_pair)
            eris.fooo[:,p0:p1] = dat[:,:,ooidx].transpose(1,0,2,3)
            eris.vvof[:,p0:p1] = dat[:,:,vvidx].transpose(2,0,1)
            eris.voof[:,:,p0:p1] = dat[:,:,voidx].transpose(2,3,0,1)

        blksize = min(nvir, max(4, int(max_memory*.9e6/8/unit)))
        eris.fvoo = eris.feri.create_dataset('fvoo', (nfrozen,nvir,nocc,nocc), 'f8')
        eris.vovf = eris.feri.create_dataset('vovf', (nvir,nocc,nvir,nfrozen), 'f8',
                                             chunks=(nvir,nocc,blksize,nfrozen))
        eris.vvvf = eris.feri.create_dataset('vvvf', (nvir_pair,nvir,nfrozen), 'f8',
                                             chunks=(nvir_pair,blksize,nfrozen))
        for p0, p1 in lib.prange(0, nvir, blksize):
            dat = numpy.empty(((p1-p0)*nfrozen,nao_pair))
            fload(fswap['0'], (nocc+p0)*nfrozen, (nocc+p1)*nfrozen, dat)
            dat = ao2mo._ao2mo.nr_e2(dat, mo_active, (0,nmo,0,nmo),
                                     's4', 's2', ao_loc=ao_loc)
            dat = dat.reshape(p1-p0,nfrozen,nmo_pair)
            eris.fvoo[:,p0:p1] = dat[:,:,ooidx].transpose(1,0,2,3)
            eris.vvvf[:,p0:p1] = dat[:,:,vvidx].transpose(2,0,1)
            eris.vovf[:,:,p0:p1] = dat[:,:,voidx].transpose(2,3,0,1)

    log.timer_debug1('CCSD gradients frozen orbital integrals', *cput0)
    return eris


def index_frozen_active(cc):
    nocc = numpy.count_nonzero(cc.mo_occ > 0)
    moidx = ccsd.get_moidx(cc)
    OA = numpy.where( moidx[:nocc])[0] # occupied active orbitals
    OF = numpy.where(~moidx[:nocc])[0] # occupied frozen orbitals
    VA = numpy.where( moidx[nocc:])[0] # virtual active orbitals
    VF = numpy.where(~moidx[nocc:])[0] # virtual frozen orbitals
    return OA, VA, OF, VF

class Gradients(lib.StreamObject):
    def __init__(self, mycc):
        self._cc = mycc
        self.stdout = mycc.stdout
        self.verbose = mycc.verbose
        self.de = None

    def kernel(self, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
               mf_grad=None, d1=None, d2=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if t1 is None: t1 = self._cc.t1
        if t2 is None: t2 = self._cc.t2
        if l1 is None: l1 = self._cc.l1
        if l2 is None: l2 = self._cc.l2
        if eris is None:
            eris = self._cc.ao2mo()
        if t1 is None or t2 is None:
            t1, t2 = self._cc.kernel(eris=eris)
        if l1 is None or l2 is None:
            l1, l2 = self._cc.solve_lambda(eris=eris)
        if atmlst is None: atmlst = range(self._cc.mol.natm)

        self.de = kernel(self._cc, t1, t2, l1, l2, eris, atmlst,
                         mf_grad, d1, d2, log)
        if self.verbose >= logger.NOTE:
            log.note('--------------- CCSD gradients ---------------')
            rhf_grad._write(self, self._cc.mol, self.de, atmlst)
            log.note('----------------------------------------------')
        return self.de

    as_scanner = as_scanner


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import grad

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.max_memory = 1
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]
    print(lib.finger(g1) - -0.036999389889460096)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.frozen = [0,1,10,11,12]
    mycc.max_memory = 1
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
#[[ -7.81105940e-17   3.81840540e-15   1.20415540e-02]
# [  1.73095055e-16  -7.94568837e-02  -6.02077699e-03]
# [ -9.49844615e-17   7.94568837e-02  -6.02077699e-03]]
    print(lib.finger(g1) - 0.10599632044533455)

    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.76',
        basis = '631g',
        unit='Bohr')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]
