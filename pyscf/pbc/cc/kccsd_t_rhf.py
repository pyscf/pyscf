#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#
"""Module for running restricted closed-shell k-point ccsd(t)"""

import ctypes
import h5py
import numpy as np
import pyscf.pbc.cc.kccsd_rhf
import time

from itertools import product
from pyscf import lib
from pyscf.cc import _ccsd
from pyscf.lib import logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

#einsum = np.einsum
einsum = lib.einsum

# CCSD(T) equations taken from Scuseria, JCP (94), 1991
#
# NOTE: As pointed out in cc/ccsd_t_slow.py, there is an error in this paper
#     and the equation should read [ia] >= [jb] >= [kc] (since the only
#     symmetry in spin-less operators is the exchange of a column of excitation
#     ooperators).
def kernel(mycc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    '''Returns the CCSD(T) for restricted closed-shell systems with k-points.

    Note:
        Returns real part of the CCSD(T) energy, raises warning if there is
        a complex part.

    Args:
        mycc (:class:`RCCSD`): Coupled-cluster object storing results of
            a coupled-cluster calculation.
        eris (:class:`_ERIS`): Integral object holding the relevant electron-
            repulsion integrals and Fock matrix elements
        t1 (:obj:`ndarray`): t1 coupled-cluster amplitudes
        t2 (:obj:`ndarray`): t2 coupled-cluster amplitudes
        max_memory (float): Maximum memory used in calculation (NOT USED)
        verbose (int, :class:`Logger`): verbosity of calculation

    Returns:
        energy_t (float): The real-part of the k-point CCSD(T) energy.
    '''
    assert isinstance(mycc, pyscf.pbc.cc.kccsd_rhf.RCCSD)
    cpu1 = cpu0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    if eris is None:
        raise TypeError('Electron repulsion integrals, `eris`, must be passed in '
                        'to the CCSD(T) kernel or created in the cc object for '
                        'the k-point CCSD(T) to run!')
    if t1 is None or t2 is None:
        raise TypeError('Must pass in t1/t2 amplitudes to k-point CCSD(T)! (Maybe '
                        'need to run `.ccsd()` on the ccsd object?)')

    cell = mycc._scf.cell
    kpts = mycc.kpts

    # The dtype of any local arrays that will be created
    dtype = t1.dtype

    nkpts, nocc, nvir = t1.shape

    mo_energy_occ = [eris.mo_energy[ki][:nocc] for ki in range(nkpts)]
    mo_energy_vir = [eris.mo_energy[ki][nocc:] for ki in range(nkpts)]
    mo_energy = np.asarray([eris.mo_energy[ki] for ki in range(nkpts)], dtype=np.float, order='C')
    fov = eris.fock[:, :nocc, nocc:]

    mo_e = mo_energy
    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    # Set up class for k-point conservation
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    # Create necessary temporary eris for fast read
    feri_tmp, t2T, eris_vvop, eris_vooo_C = create_t3_eris(mycc, kconserv, [eris.vovv, eris.oovv, eris.ooov, t2])
    t1T = np.array([x.T for x in t1], dtype=np.complex, order='C')
    fvo = np.array([x.T for x in fov], dtype=np.complex, order='C')
    cpu1 = log.timer_debug1('CCSD(T) tmp eri creation', *cpu1)

    #def get_w_old(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1, out=None):
    #    '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
    #    km = kconserv[kc, kk, kb]
    #    kf = kconserv[kk, kc, kj]
    #    ret = einsum('kjcf,fiba->abcijk', t2[kk,kj,kc,:,:,c0:c1,:], eris.vovv[kf,ki,kb,:,:,b0:b1,a0:a1].conj())
    #    ret = ret - einsum('mkbc,jima->abcijk', t2[km,kk,kb,:,:,b0:b1,c0:c1], eris.ooov[kj,ki,km,:,:,:,a0:a1].conj())
    #    return ret

    def get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts

        Uses tranposed eris for fast data access.'''
        km = kconserv[kc, kk, kb]
        kf = kconserv[kk, kc, kj]
        out = einsum('cfjk,abif->abcijk', t2T[kc,kf,kj,c0:c1,:,:,:], eris_vvop[ka,kb,ki,a0:a1,b0:b1,:,nocc:])
        out = out - einsum('cbmk,aijm->abcijk', t2T[kc,kb,km,c0:c1,b0:b1,:,:], eris_vooo_C[ka,ki,kj,a0:a1,:,:,:])
        return out

    def get_permuted_w(ki, kj, kk, ka, kb, kc, orb_indices):
        '''Pijkabc operating on Wijkabc intermediate as described in Scuseria paper'''
        a0, a1, b0, b1, c0, c1 = orb_indices
        out = get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1)
        out = out + get_w(kj, kk, ki, kb, kc, ka, b0, b1, c0, c1, a0, a1).transpose(2,0,1,5,3,4)
        out = out + get_w(kk, ki, kj, kc, ka, kb, c0, c1, a0, a1, b0, b1).transpose(1,2,0,4,5,3)
        out = out + get_w(ki, kk, kj, ka, kc, kb, a0, a1, c0, c1, b0, b1).transpose(0,2,1,3,5,4)
        out = out + get_w(kk, kj, ki, kc, kb, ka, c0, c1, b0, b1, a0, a1).transpose(2,1,0,5,4,3)
        out = out + get_w(kj, ki, kk, kb, ka, kc, b0, b1, a0, a1, c0, c1).transpose(1,0,2,4,3,5)
        return out

    def get_rw(ki, kj, kk, ka, kb, kc, orb_indices):
        '''R operating on Wijkabc intermediate as described in Scuseria paper'''
        a0, a1, b0, b1, c0, c1 = orb_indices
        ret = (4. * get_permuted_w(ki,kj,kk,ka,kb,kc,orb_indices) +
               1. * get_permuted_w(kj,kk,ki,ka,kb,kc,orb_indices).transpose(0,1,2,5,3,4) +
               1. * get_permuted_w(kk,ki,kj,ka,kb,kc,orb_indices).transpose(0,1,2,4,5,3) -
               2. * get_permuted_w(ki,kk,kj,ka,kb,kc,orb_indices).transpose(0,1,2,3,5,4) -
               2. * get_permuted_w(kk,kj,ki,ka,kb,kc,orb_indices).transpose(0,1,2,5,4,3) -
               2. * get_permuted_w(kj,ki,kk,ka,kb,kc,orb_indices).transpose(0,1,2,4,3,5))
        return ret

    #def get_v_old(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
    #    '''Vijkabc intermediate as described in Scuseria paper'''
    #    km = kconserv[ki,ka,kj]
    #    kf = kconserv[ki,ka,kj]
    #    out = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
    #    if kk == kc:
    #        out = out + einsum('kc,ijab->abcijk', 0.5*t1[kk,:,c0:c1], eris.oovv[ki,kj,ka,:,:,a0:a1,b0:b1].conj())
    #        out = out + einsum('kc,ijab->abcijk', 0.5*fov[kk,:,c0:c1], t2[ki,kj,ka,:,:,a0:a1,b0:b1])
    #    return out

    def get_v(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Vijkabc intermediate as described in Scuseria paper'''
        #km = kconserv[ki,ka,kj]
        #kf = kconserv[ki,ka,kj]
        out = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
        if kk == kc:
            out = out + einsum('ck,baji->abcijk', 0.5*t1T[kk,c0:c1,:], eris_vvop[kb,ka,kj,b0:b1,a0:a1,:,:nocc])
            # We see this is the same t2T term needed for the `w` contraction:
            #     einsum('cbmk,aijm->abcijk', t2T[kc,kb,km,c0:c1,b0:b1], eris_vooo_C[ka,ki,kj,a0:a1])
            #
            # For the kpoint indices [kk,ki,kj,kc,ka,kb] we have that we need
            #     t2T[kb,ka,km], where km = kconserv[kb,kj,ka]
            # The remaining k-point not used in t2T, i.e. kc, has the condition kc == kk in the case of
            # get_v.  So, we have from 3-particle conservation
            #     (kk-kc) + ki + kj - ka - kb = 0,
            # i.e. ki = km.
            out = out + einsum('ck,baij->abcijk', 0.5*fvo[kk,c0:c1,:], t2T[kb,ka,ki,b0:b1,a0:a1,:,:])
        return out

    def get_permuted_v(ki, kj, kk, ka, kb, kc, orb_indices):
        '''Pijkabc operating on Vijkabc intermediate as described in Scuseria paper'''
        a0, a1, b0, b1, c0, c1 = orb_indices
        ret = get_v(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1)
        ret = ret + get_v(kj, kk, ki, kb, kc, ka, b0, b1, c0, c1, a0, a1).transpose(2,0,1,5,3,4)
        ret = ret + get_v(kk, ki, kj, kc, ka, kb, c0, c1, a0, a1, b0, b1).transpose(1,2,0,4,5,3)
        ret = ret + get_v(ki, kk, kj, ka, kc, kb, a0, a1, c0, c1, b0, b1).transpose(0,2,1,3,5,4)
        ret = ret + get_v(kk, kj, ki, kc, kb, ka, c0, c1, b0, b1, a0, a1).transpose(2,1,0,5,4,3)
        ret = ret + get_v(kj, ki, kk, kb, ka, kc, b0, b1, a0, a1, c0, c1).transpose(1,0,2,4,3,5)
        return ret

    def contract_t3Tv(kpt_indices, orb_indices, data):
        '''Calculate t3T(ransposed) array using C driver.'''
        ki, kj, kk, ka, kb, kc = kpt_indices
        a0, a1, b0, b1, c0, c1 = orb_indices
        slices = np.array([a0, a1, b0, b1, c0, c1], dtype=np.int32)

        mo_offset = np.array([ki,kj,kk,ka,kb,kc], dtype=np.int32)

        vvop_ab = np.asarray(data[0][0], dtype=np.complex, order='C')
        vvop_ac = np.asarray(data[0][1], dtype=np.complex, order='C')
        vvop_ba = np.asarray(data[0][2], dtype=np.complex, order='C')
        vvop_bc = np.asarray(data[0][3], dtype=np.complex, order='C')
        vvop_ca = np.asarray(data[0][4], dtype=np.complex, order='C')
        vvop_cb = np.asarray(data[0][5], dtype=np.complex, order='C')

        vooo_aj = np.asarray(data[1][0], dtype=np.complex, order='C')
        vooo_ak = np.asarray(data[1][1], dtype=np.complex, order='C')
        vooo_bi = np.asarray(data[1][2], dtype=np.complex, order='C')
        vooo_bk = np.asarray(data[1][3], dtype=np.complex, order='C')
        vooo_ci = np.asarray(data[1][4], dtype=np.complex, order='C')
        vooo_cj = np.asarray(data[1][5], dtype=np.complex, order='C')

        t2T_cj = np.asarray(data[2][0], dtype=np.complex, order='C')
        t2T_bk = np.asarray(data[2][1], dtype=np.complex, order='C')
        t2T_ci = np.asarray(data[2][2], dtype=np.complex, order='C')
        t2T_ak = np.asarray(data[2][3], dtype=np.complex, order='C')
        t2T_bi = np.asarray(data[2][4], dtype=np.complex, order='C')
        t2T_aj = np.asarray(data[2][5], dtype=np.complex, order='C')

        t2T_cb = np.asarray(data[3][0], dtype=np.complex, order='C')
        t2T_bc = np.asarray(data[3][1], dtype=np.complex, order='C')
        t2T_ca = np.asarray(data[3][2], dtype=np.complex, order='C')
        t2T_ac = np.asarray(data[3][3], dtype=np.complex, order='C')
        t2T_ba = np.asarray(data[3][4], dtype=np.complex, order='C')
        t2T_ab = np.asarray(data[3][5], dtype=np.complex, order='C')

        data = [vvop_ab, vvop_ac, vvop_ba, vvop_bc, vvop_ca, vvop_cb,
                vooo_aj, vooo_ak, vooo_bi, vooo_bk, vooo_ci, vooo_cj,
                t2T_cj, t2T_cb, t2T_bk, t2T_bc, t2T_ci, t2T_ca, t2T_ak,
                t2T_ac, t2T_bi, t2T_ba, t2T_aj, t2T_ab]
        data_ptrs = [x.ctypes.data_as(ctypes.c_void_p) for x in data]
        data_ptrs = (ctypes.c_void_p*24)(*data_ptrs)

        a0, a1, b0, b1, c0, c1 = task
        t3Tw = np.empty((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=np.complex, order='C')
        t3Tv = np.empty((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=np.complex, order='C')

        drv = _ccsd.libcc.CCsd_zcontract_t3T
        drv(t3Tw.ctypes.data_as(ctypes.c_void_p),
            t3Tv.ctypes.data_as(ctypes.c_void_p),
            mo_e.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(nkpts),
            mo_offset.ctypes.data_as(ctypes.c_void_p),
            slices.ctypes.data_as(ctypes.c_void_p),
            data_ptrs)
        return t3Tw, t3Tv

    def get_data(kpt_indices):
        idx_args = get_data_slices(kpt_indices, task, kconserv)
        vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices = idx_args
        vvop_data = [eris_vvop[tuple(x)] for x in vvop_indices]
        vooo_data = [eris_vooo_C[tuple(x)] for x in vooo_indices]
        t2T_vvop_data = [t2T[tuple(x)] for x in t2T_vvop_indices]
        t2T_vooo_data = [t2T[tuple(x)] for x in t2T_vooo_indices]
        data = [vvop_data, vooo_data, t2T_vvop_data, t2T_vooo_data]
        return data

    energy_t = 0.0

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blkmin = 4
    # temporary t3 array is size:    2 * nkpts**3 * blksize**3 * nocc**3 * 16
    vir_blksize = min(nvir, max(blkmin, int((max_memory*.9e6/16/nocc**3/nkpts**3/2)**(1./3))))
    tasks = []
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    log.debug('virtual blksize = %d (nvir = %d)', nvir, vir_blksize)
    for a0, a1 in lib.prange(0, nvir, vir_blksize):
        for b0, b1 in lib.prange(0, nvir, vir_blksize):
            for c0, c1 in lib.prange(0, nvir, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka in range(nkpts):
        for kb in range(ka+1):
            for task_id, task in enumerate(tasks):
                a0,a1,b0,b1,c0,c1 = task
                my_permuted_w = np.zeros((nkpts,)*3 + (a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
                my_permuted_v = np.zeros((nkpts,)*3 + (a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
                for ki, kj, kk in product(range(nkpts), repeat=3):
                    # Find momentum conservation condition for triples
                    # amplitude t3ijkabc
                    kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                    if not (ka >= kb and kb >= kc):
                        continue

                    kpt_indices = [ki,kj,kk,ka,kb,kc]
                    data = get_data(kpt_indices)
                    t3Tw, t3Tv = contract_t3Tv(kpt_indices, task, data)
                    my_permuted_w[ki,kj,kk] = t3Tw
                    my_permuted_v[ki,kj,kk] = t3Tv
                    #my_permuted_w[ki,kj,kk] = get_permuted_w(ki,kj,kk,ka,kb,kc,task)
                    #my_permuted_v[ki,kj,kk] = get_permuted_v(ki,kj,kk,ka,kb,kc,task)

                for ki, kj, kk in product(range(nkpts), repeat=3):
                    # eigenvalue denominator: e(i) + e(j) + e(k)
                    eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                                     [0,nocc,kj,mo_e_o,nonzero_opadding],
                                     [0,nocc,kk,mo_e_o,nonzero_opadding])

                    # Find momentum conservation condition for triples
                    # amplitude t3ijkabc
                    kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                    if not (ka >= kb and kb >= kc):
                        continue

                    if ka == kb and kb == kc:
                        symm_kpt = 1.
                    elif ka == kb or kb == kc:
                        symm_kpt = 3.
                    else:
                        symm_kpt = 6.

                    eabc = _get_epqr([a0,a1,ka,mo_e_v,nonzero_vpadding],
                                     [b0,b1,kb,mo_e_v,nonzero_vpadding],
                                     [c0,c1,kc,mo_e_v,nonzero_vpadding],
                                     fac=[-1.,-1.,-1.])
                    eijkabc = (eijk[None,None,None,:,:,:] + eabc[:,:,:,None,None,None])

                    pwijk = my_permuted_w[ki,kj,kk] + my_permuted_v[ki,kj,kk]
                    rwijk = (4. * my_permuted_w[ki,kj,kk] +
                             1. * my_permuted_w[kj,kk,ki].transpose(0,1,2,5,3,4) +
                             1. * my_permuted_w[kk,ki,kj].transpose(0,1,2,4,5,3) -
                             2. * my_permuted_w[ki,kk,kj].transpose(0,1,2,3,5,4) -
                             2. * my_permuted_w[kk,kj,ki].transpose(0,1,2,5,4,3) -
                             2. * my_permuted_w[kj,ki,kk].transpose(0,1,2,4,3,5))
                    rwijk = rwijk / eijkabc
                    energy_t += symm_kpt * einsum('abcijk,abcijk', rwijk, pwijk.conj())

    energy_t *= (1. / 3)
    energy_t /= nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s', energy_t.imag)
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
    log.note('CCSD(T) correction per cell (imag) = %.15g', energy_t.imag)
    return energy_t.real



###################################
# Helper function for t3 creation
###################################

def check_read_success(filename, **kwargs):
    '''Determine criterion for successfully reading a dataset based on its
    meta values.

    For now, returns False.'''
    def check_write_complete(filename, **kwargs):
        '''Check for `completed` attr in file.'''
        import os
        mode = kwargs.get('mode', 'r')
        if not os.path.isfile(filename):
            return False
        f = h5py.File(filename, mode=mode, **kwargs)
        return f.attrs.get('completed', False)
    write_complete = check_write_complete(filename, **kwargs)
    return False and write_complete


def transpose_t2(t2, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates t2.transpose(2,3,1,0).'''
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc), dtype=t2.dtype)

    # Check if it's stored in lower triangular form
    if len(t2.shape) == 7 and t2.shape[:2] == (nkpts, nkpts):
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            out[ka,kb,kj] = t2[ki,kj,ka].transpose(2,3,1,0)
    elif len(t2.shape) == 6 and t2.shape[:2] == (nkpts*(nkpts+1)//2, nkpts):
        for ki, kj, ka in product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            # t2[ki,kj,ka] = t2[tril_index(ki,kj),ka]  ki<kj
            # t2[kj,ki,kb] = t2[ki,kj,ka].transpose(1,0,3,2)  ki<kj
            #              = t2[tril_index(ki,kj),ka].transpose(1,0,3,2)
            if ki <= kj:
                tril_idx = (kj*(kj+1))//2 + ki
                out[ka,kb,kj] = t2[tril_idx,ka].transpose(2,3,1,0).copy()
                out[kb,ka,ki] = out[ka,kb,kj].transpose(1,0,3,2)
    else:
        raise ValueError('No known conversion for t2 shape %s' % t2.shape)
    return out


def create_eris_vvop(vovv, oovv, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates vvop from vovv and oovv array (physicist notation).'''
    nmo = nocc + nvir
    assert(vovv.shape == (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir))
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo), dtype=vovv.dtype)
    else:
        assert(out.shape == (nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo))

    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki,ka,kj]
        out[ki,kj,ka,:,:,:,nocc:] = vovv[kb,ka,kj].conj().transpose(3,2,1,0)
        if oovv is not None:
            out[ki,kj,ka,:,:,:,:nocc] = oovv[kb,ka,kj].conj().transpose(3,2,1,0)
    return out


def create_eris_vooo(ooov, nkpts, nocc, nvir, kconserv, out=None):
    '''Creates vooo from ooov array.

    This is not exactly chemist's notation, but close.  Here a chemist notation vooo
    is created from physicist ooov, and then the last two indices of vooo are swapped.
    '''
    assert(ooov.shape == (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir))
    if out is None:
        out = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nocc), dtype=ooov.dtype)

    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki,kj,ka]
        # <bj|ai> -> (ba|ji)    (Physicist->Chemist)
        # (ij|ab) = (ba|ij)*    (Permutational symmetry)
        # out = (ij|ab).transpose(0,1,3,2)
        out[ki,kj,kb] = ooov[kb,kj,ka].conj().transpose(3,1,0,2)
    return out


def create_t3_eris(mycc, kconserv, eris, tmpfile='tmp_t3_eris.h5'):
    '''Create/transpose necessary eri integrals needed for fast read-in by CCSD(T).'''
    eris_vovv, eris_oovv, eris_ooov, t2 = eris
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc

    nmo = nocc + nvir
    feri_tmp = None
    h5py_kwargs = {}
    feri_tmp_filename = tmpfile
    dtype = np.result_type(eris_vovv, eris_oovv, eris_ooov, t2)
    if not check_read_success(feri_tmp_filename):
        feri_tmp = lib.H5TmpFile(feri_tmp_filename, 'w', **h5py_kwargs)
        t2T_out = feri_tmp.create_dataset('t2T',
                      (nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc), dtype=dtype)
        eris_vvop_out = feri_tmp.create_dataset('vvop',
                            (nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo), dtype=dtype)
        eris_vooo_C_out = feri_tmp.create_dataset('vooo_C',
                              (nkpts,nkpts,nkpts,nvir,nocc,nocc,nocc), dtype=dtype)

        transpose_t2(t2, nkpts, nocc, nvir, kconserv, out=t2T_out)
        create_eris_vvop(eris_vovv, eris_oovv, nkpts, nocc, nvir, kconserv, out=eris_vvop_out)
        create_eris_vooo(eris_ooov, nkpts, nocc, nvir, kconserv, out=eris_vooo_C_out)

        feri_tmp.attrs['completed'] = True
        feri_tmp.close()

    feri_tmp = lib.H5TmpFile(feri_tmp_filename, 'r', **h5py_kwargs)
    t2T = feri_tmp['t2T']
    eris_vvop = feri_tmp['vvop']
    eris_vooo_C = feri_tmp['vooo_C']

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    unit = nkpts**3 * (nvir**2 * nocc**2 + nvir**2 * nmo * nocc + nvir * nocc**3)
    if (unit*16 < max_memory):  # Store all in memory
        t2T = t2T[:]
        eris_vvop = eris_vvop[:]
        eris_vooo_C = eris_vooo_C[:]

    return feri_tmp, t2T, eris_vvop, eris_vooo_C


def _convert_to_int(kpt_indices):
    '''Convert all kpoint indices for 3-particle operator to integers.'''
    out_indices = [0]*6
    for ix, x in enumerate(kpt_indices):
        assert isinstance(x, (int, np.int, np.ndarray, list))
        if isinstance(x, (np.ndarray)) and (x.ndim == 0):
            out_indices[ix] = int(x)
        else:
            out_indices[ix] = x
    return out_indices


def _tile_list(kpt_indices):
    '''Similar to a cartesian product but for a list of kpoint indices for
    a 3-particle operator.'''
    max_length = 0
    out_indices = [0]*6
    for ix, x in enumerate(kpt_indices):
        if hasattr(x, '__len__'):
            max_length = max(max_length, len(x))

    if max_length == 0:
        return kpt_indices
    else:
        for ix, x in enumerate(kpt_indices):
            if isinstance(x, (int, np.int)):
                out_indices[ix] = [x] * max_length
            else:
                out_indices[ix] = x

    return map(list, zip(*out_indices))


def zip_kpoints(kpt_indices):
    '''Similar to a cartesian product but for a list of kpoint indices for
    a 3-particle operator.  Ensures all indices are integers.'''
    out_indices = _convert_to_int(kpt_indices)
    out_indices = _tile_list(out_indices)
    return out_indices


def get_data_slices(kpt_indices, orb_indices, kconserv):
    kpt_indices = zip_kpoints(kpt_indices)
    if isinstance(kpt_indices[0], (int, np.int)):  # Ensure we are working
        kpt_indices = [kpt_indices]                # with a list of lists

    a0,a1,b0,b1,c0,c1 = orb_indices
    length = len(kpt_indices)*6

    def _vijk_indices(kpt_indices, orb_indices, transpose=(0, 1, 2)):
        '''Get indices needed for t3 construction and a given transpose of (a,b,c).'''
        kpt_indices = ([kpt_indices[x] for x in transpose] +
                       [kpt_indices[x+3] for x in transpose])
        orb_indices = lib.flatten([[orb_indices[2*x], orb_indices[2*x+1]]
                                   for x in transpose])

        ki, kj, kk, ka, kb, kc = kpt_indices
        a0, a1, b0, b1, c0, c1 = orb_indices

        kf = kconserv[ka,ki,kb]
        km = kconserv[kc,kk,kb]
        sl00 = slice(None, None)

        vvop_idx = [ka, kb, ki, slice(a0,a1), slice(b0,b1), sl00, sl00]
        vooo_idx = [ka, ki, kj, slice(a0,a1), sl00, sl00, sl00]
        t2T_vvop_idx = [kc, kf, kj, slice(c0,c1), sl00, sl00, sl00]
        t2T_vooo_idx = [kc, kb, km, slice(c0,c1), sl00, sl00, sl00]
        return vvop_idx, vooo_idx, t2T_vvop_idx, t2T_vooo_idx

    vvop_indices = [0] * length
    vooo_indices = [0] * length
    t2T_vvop_indices = [0] * length
    t2T_vooo_indices = [0] * length

    transpose = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    count = 0
    for kpt in kpt_indices:
        for t in transpose:
            vvop_idx, vooo_idx, t2T_vvop_idx, t2T_vooo_idx = _vijk_indices(kpt, orb_indices, t)
            vvop_indices[count] = vvop_idx
            vooo_indices[count] = vooo_idx
            t2T_vvop_indices[count] = t2T_vvop_idx
            t2T_vooo_indices[count] = t2T_vooo_idx
            count += 1

    return vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices

def _get_epqr(pindices,qindices,rindices,fac=[1.0,1.0,1.0],large_num=LARGE_DENOM):
    '''Create a denominator

        fac[0]*e[kp,p0:p1] + fac[1]*e[kq,q0:q1] + fac[2]*e[kr,r0:r1]

    where padded elements have been replaced by a large number.

    Args:
        pindices (5-list of object):
            A list of p0, p1, kp, orbital values, and non-zero indices for the first
            denominator element.
        qindices (5-list of object):
            A list of q0, q1, kq, orbital values, and non-zero indices for the second
            denominator element.
        rindices (5-list of object):
            A list of r0, r1, kr, orbital values, and non-zero indices for the third
            denominator element.
        fac (3-list of float):
            Factors to multiply the first and second denominator elements.
        large_num (float):
            Number to replace the padded elements.
    '''
    def get_idx(x0,x1,kx,n0_p):
        return np.logical_and(n0_p[kx] >= x0, n0_p[kx] < x1)

    assert(all([len(x) == 5 for x in [pindices,qindices]]))
    p0,p1,kp,mo_e_p,nonzero_p = pindices
    q0,q1,kq,mo_e_q,nonzero_q = qindices
    r0,r1,kr,mo_e_r,nonzero_r = rindices
    fac_p, fac_q, fac_r = fac

    epqr = large_num * np.ones((p1-p0,q1-q0,r1-r0), dtype=mo_e_p[0].dtype)
    idxp = get_idx(p0,p1,kp,nonzero_p)
    idxq = get_idx(q0,q1,kq,nonzero_q)
    idxr = get_idx(r0,r1,kr,nonzero_r)
    n0_ovp_pqr = np.ix_(nonzero_p[kp][idxp]-p0, nonzero_q[kq][idxq]-q0, nonzero_r[kr][idxr]-r0)
    epqr[n0_ovp_pqr] = lib.direct_sum('p,q,r->pqr', fac_p*mo_e_p[kp][p0:p1],
                                      fac_q*mo_e_q[kq][q0:q1],
                                      fac_r*mo_e_r[kr][r0:r1])[n0_ovp_pqr]
    #epqr[n0_ovp_pqr] = temp[n0_ovp_pqr]
    return epqr

if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf.pbc import cc

    cell = gto.Cell()
    cell.atom = '''
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
    cell.verbose = 4
    cell.mesh = [24, 24, 24]
    cell.build()

    nmp = [1,1,4]
    kpts = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    kmf.conv_tol = 1e-12
    kmf.conv_tol_grad = 1e-12
    kmf.direct_scf_tol = 1e-16
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    eris = mycc.ao2mo()
    ecc, t1, t2 = mycc.kernel(eris=eris)
    energy_t = kernel(mycc, eris=eris, verbose=9)

    # Start of supercell calculations
    from pyscf.pbc.tools.pbc import super_cell
    supcell = super_cell(cell, nmp)
    supcell.build()
    kmf = scf.RHF(supcell, exxdiv=None)
    kmf.conv_tol = 1e-12
    kmf.conv_tol_grad = 1e-12
    kmf.direct_scf_tol = 1e-16
    sup_ehf = kmf.kernel()

    myscc = cc.RCCSD(kmf)
    eris = myscc.ao2mo()
    sup_ecc, t1, t2 = myscc.kernel(eris=eris)
    sup_energy_t = myscc.ccsd_t(eris=eris)
    print("Kpoint    CCSD: %20.16f" % ecc)
    print("Supercell CCSD: %20.16f" % (sup_ecc/np.prod(nmp)))
    print("Kpoint    CCSD(T): %20.16f" % energy_t)
    print("Supercell CCSD(T): %20.16f" % (sup_energy_t/np.prod(nmp)))
