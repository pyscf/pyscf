#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
kpoint-adapted and spin-adapted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab
'''

import time
from functools import reduce
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf.pbc.lib import kpts_helper

def kernel(mp, mo_energy, mo_coeff, eris=None, verbose=logger.NOTE):
    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    nkpts = mp.nkpts
    t2 = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)

    woovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.fock.dtype)
    emp2 = 0
    foo = eris.fock[:,:nocc,:nocc].copy()
    fvv = eris.fock[:,nocc:,nocc:].copy()
    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    kconserv = mp.khelper.kconserv
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki,ka,kj]
        eia = np.diagonal(foo[ki]).reshape(-1,1) - np.diagonal(fvv[ka])
        ejb = np.diagonal(foo[kj]).reshape(-1,1) - np.diagonal(fvv[kb])
        eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
        woovv[ki,kj,ka] = (2*eris.oovv[ki,kj,ka] - eris.oovv[ki,kj,kb].transpose(0,1,3,2))
        t2[ki,kj,ka] = eris.oovv[ki,kj,ka] / eijab

    t2 = t2.conj()
    emp2 = np.einsum('pqrijab,pqrijab',t2,woovv).real
    emp2 /= nkpts

    return emp2, t2


def get_frozen_mask(mp):
    moidx = [np.ones(x.size, dtype=np.bool) for x in mp.mo_occ]
    if isinstance(mp.frozen, (int, np.integer)):
        for idx in moidx:
            idx[:mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, np.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    else:
        raise NotImplementedError
    return moidx


def get_nocc(mp):
    '''The number of occupied orbitals per k-point.'''
    if mp._nocc is not None:
        return mp._nocc
    elif isinstance(mp.frozen, (int, np.integer)):
        nocc = int(mp.mo_occ[0].sum()) // 2 - mp.frozen
    elif isinstance(mp.frozen[0], (int, np.integer)):
        occ_idx = mp.mo_occ[0] > 0
        occ_idx[list(mp.frozen)] = False
        nocc = np.count_nonzero(occ_idx)
    else:
        raise NotImplementedError
    return nocc

def get_nmo(mp):
    '''The number of molecular orbitals per k-point.'''
    if mp._nmo is not None:
        return mp._nmo
    if isinstance(mp.frozen, (int, np.integer)):
        nmo = len(mp.mo_occ[0]) - mp.frozen
    elif isinstance(mp.frozen[0], (int, np.integer)):
        nmo = len(mp.mo_occ[0]) - len(mp.frozen)
    else:
        raise NotImplementedError
    return nmo


class KMP2(mp2.MP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.nkpts = len(self.kpts)
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, eris, verbose=self.verbose)
        logger.log(self, 'KMP2 energy = %.15g', self.e_corr)
        return self.e_corr, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff, verbose=self.verbose)


def _mem_usage(nkpts, nocc, nvir):
    nmo = nocc + nvir
    basic = (nkpts**3*nocc**2*nvir**2*2)*16 / 1e6
    # Roughly, factor of two for safety (t2 array, temp arrays, copying, etc)
    basic *= 2
    incore = nmo**4*16 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ERIS:
    def __init__(self, mp, mo_coeff=None, verbose=None):
        cput0 = (time.clock(), time.time())
        moidx = get_frozen_mask(mp)
        nkpts = mp.nkpts
        nmo = mp.nmo

        nao = mp.mo_coeff[0].shape[0]
        dtype = mp.mo_coeff[0].dtype
        self.mo_coeff = np.zeros((nkpts,nao,nmo), dtype=dtype)
        self.fock = np.zeros((nkpts,nmo,nmo), dtype=dtype)
        if mo_coeff is None:
            for kp in range(nkpts):
                self.mo_coeff[kp] = mp.mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            for kp in range(nkpts):
                self.fock[kp] = np.diag(mp.mo_energy[kp][moidx[kp]]).astype(dtype)
        else:  # If mo_coeff is not canonical orbital
            for kp in range(nkpts):
                self.mo_coeff[kp] = mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            dm = mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)
            # Don't use get_veff(), because mp._scf might be DFT,
            # but veff should be Fock, not Kohn-Sham.
            #fockao = mp._scf.get_hcore() + mp._scf.get_veff(mp.mol, dm)
            vj, vk = mp._scf.get_jk(mp.mol, dm)
            veff = vj - vk * .5
            fockao = mp._scf.get_hcore() + veff
            for kp in range(nkpts):
                self.fock[kp] = reduce(np.dot, (mo_coeff[kp].T.conj(), fockao[kp], mo_coeff[kp])).astype(dtype)

        nocc = mp.nocc
        nmo = mp.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = mp._scf.with_df.ao2mo

        kconserv = mp.khelper.kconserv

        max_memory = max(2000, mp.max_memory*.9-mem_now)
        log = logger.Logger(mp.stdout, mp.verbose)
        if mp.max_memory < mem_basic:
            log.warn('Not enough memory for integral transformation. '
                     'Available mem %s MB, required mem %s MB',
                     max_memory, mem_basic)

        self.oovv = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)

        if (mp.mol.incore_anyway or
                (mem_incore+mem_now < mp.max_memory)):
            # more memory, less work because of irreducible k-points
            log.debug('transform (pq|rs) incore for irreducible k-points')

            khelper = mp.khelper

            for (ikp,ikq,ikr) in khelper.symm_map.keys():
                iks = kconserv[ikp,ikq,ikr]
                eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                                 (mp.kpts[ikp],mp.kpts[ikq],mp.kpts[ikr],mp.kpts[iks]), compact=False)
                if dtype == np.float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                for (kp,kq,kr) in khelper.symm_map[(ikp,ikq,ikr)]:
                    eri_kpt_symm = khelper.transform_symm(eri_kpt,kp,kq,kr).transpose(0,2,1,3)
                    self.oovv[kp,kr,kq] = eri_kpt_symm[:nocc,:nocc,nocc:,nocc:] / nkpts

            self.dtype = dtype
        else:
            # less memory, more work because no irreducible k-points
            log.debug('transform (ia|jb) incore for all k-points')
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        orbo_p = mo_coeff[kp,:,:nocc]
                        orbo_r = mo_coeff[kr,:,:nocc]
                        orbv_q = mo_coeff[kq,:,nocc:]
                        orbv_s = mo_coeff[ks,:,nocc:]
                        eri_kpt = fao2mo((orbo_p,orbv_q,orbo_r,orbv_s),
                                        (mp.kpts[kp],mp.kpts[kq],mp.kpts[kr],mp.kpts[ks]), 
                                        compact=False)
                        if dtype == np.float: eri_kpt = eri_kpt.real
                        eri_kpt = eri_kpt.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
                        self.oovv[kp,kr,kq] = eri_kpt / nkpts

            self.dtype = eri_kpt.dtype

        log.timer('integral transformation', *cput0)


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp 

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    ehf = kmf.kernel()

    mymp = mp.KMP2(kmf)
    emp2, t2 = mymp.kernel()
    print(emp2 - -0.204721432828996)

