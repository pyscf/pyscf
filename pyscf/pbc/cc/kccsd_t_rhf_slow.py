#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#
"""Module for running restricted closed-shell k-point ccsd(t)"""

import h5py
import itertools
import numpy as np
import pyscf.pbc.cc.kccsd_rhf
import time

from itertools import product
from pyscf import lib
#from pyscf import _ccsd
from pyscf.lib import logger
from pyscf.lib.misc import tril_product
from pyscf.lib.misc import flatten
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.lib.numpy_helper import pack_tril
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf import __config__

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
    fov = eris.fock[:, :nocc, nocc:]

    # Set up class for k-point conservation
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    # Create necessary temporary eris for fast read
    feri_tmp, t2T, eris_vvop, eris_vooo_C = create_t3_eris(nkpts, nocc, nvir, kconserv)
    t1T = np.array([x.T for x in t1])
    fvo = np.array([x.T for x in fov])
    cpu1 = log.timer_debug1('CCSD(T) tmp eri creation', *cpu1)

    def get_w_old(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1, out=None):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[kk, kc, kj]
        ret = einsum('kjcf,fiba->abcijk', t2[kk,kj,kc,:,:,c0:c1,:], eris.vovv[kf,ki,kb,:,:,b0:b1,a0:a1].conj())
        ret = ret - einsum('mkbc,jima->abcijk', t2[km,kk,kb,:,:,b0:b1,c0:c1], eris.ooov[kj,ki,km,:,:,:,a0:a1].conj())
        return ret

    def get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts

        Uses tranposed eris for fast data access.'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[kk, kc, kj]
        out = einsum('cfjk,abif->abcijk', t2T[kc,kf,kj,c0:c1,:,:,:], eris_vvop[ka,kb,ki,a0:a1,b0:b1,:,nocc:])
        out = out - einsum('bckm,aijm->abcijk', t2T[kb,kc,kk,b0:b1,c0:c1,:,:], eris_vooo_C[ka,ki,kj,a0:a1,:,:,:])
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

    def get_v_old(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki,ka,kj]
        kf = kconserv[ki,ka,kj]
        out = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
        if kk == kc:
            out = out + einsum('kc,ijab->abcijk', t1[kk,:,c0:c1], eris.oovv[ki,kj,ka,:,:,a0:a1,b0:b1].conj())
            out = out + einsum('kc,ijab->abcijk', fov[kk,:,c0:c1], t2[ki,kj,ka,:,:,a0:a1,b0:b1])
        return out

    def get_v(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki,ka,kj]
        kf = kconserv[ki,ka,kj]
        out = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
        if kk == kc:
            out = out + einsum('ck,baji->abcijk', t1T[kk,c0:c1,:], eris_vvop[kb,ka,kj,b0:b1,a0:a1,:,:nocc])
            out = out + einsum('ck,abji->abcijk', fvo[kk,c0:c1,:], t2T[ka,kb,kj,a0:a1,b0:b1,:,:])
        return out

    def get_permuted_v(ki, kj, kk, ka, kb, kc, orb_indices):
        '''Pijkabc operating on Vijkabc intermediate as described in Scuseria paper'''
        a0, a1, b0, b1, c0, c1 = orb_indices
        tmp = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
        ret = get_v(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1)
        ret = ret + get_v(kj, kk, ki, kb, kc, ka, b0, b1, c0, c1, a0, a1).transpose(2,0,1,5,3,4)
        ret = ret + get_v(kk, ki, kj, kc, ka, kb, c0, c1, a0, a1, b0, b1).transpose(1,2,0,4,5,3)
        ret = ret + get_v(ki, kk, kj, ka, kc, kb, a0, a1, c0, c1, b0, b1).transpose(0,2,1,3,5,4)
        ret = ret + get_v(kk, kj, ki, kc, kb, ka, c0, c1, b0, b1, a0, a1).transpose(2,1,0,5,4,3)
        ret = ret + get_v(kj, ki, kk, kb, ka, kc, b0, b1, a0, a1, c0, c1).transpose(1,0,2,4,3,5)
        return ret

    energy_t = 0.0

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blkmin = 4
    # temporary t3 array is size:    blksize**3 * nocc**3 * 16
    vir_blksize = min(nvir, max(blkmin, int((max_memory*.9e6/16/nocc**3)**(1./3))))
    tasks = []
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    log.debug('virtual blksize = %d (nvir = %d)', nvir, vir_blksize)
    for a0, a1 in lib.prange(0, nvir, vir_blksize):
        for b0, b1 in lib.prange(0, nvir, vir_blksize):
            for c0, c1 in lib.prange(0, nvir, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka in range(nkpts):
        for kb in range(ka+1):

            for ki, kj, kk in product(range(nkpts), repeat=3):
                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = LARGE_DENOM * np.ones((nocc,)*3, dtype=mo_energy_occ[0].dtype)
                n0_ovp_ijk = np.ix_(nonzero_opadding[ki], nonzero_opadding[kj], nonzero_opadding[kk])
                eijk[n0_ovp_ijk] = lib.direct_sum('i,j,k->ijk', mo_energy_occ[ki], mo_energy_occ[kj], mo_energy_occ[kk])[n0_ovp_ijk]

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

                for task_id, task in enumerate(tasks):
                    orb_indices = a0,a1,b0,b1,c0,c1
                    eijkabc = (eijk[None,None,None,:,:,:] -
                               mo_energy_vir[ka][a0:a1][:,None,None,None,None,None] -
                               mo_energy_vir[kb][b0:b1][None,:,None,None,None,None] -
                               mo_energy_vir[kc][c0:c1][None,None,:,None,None,None])
                    pwijk = (       get_permuted_w(ki,kj,kk,ka,kb,kc,task) +
                              0.5 * get_permuted_v(ki,kj,kk,ka,kb,kc,task) )
                    rwijk = get_rw(ki,kj,kk,ka,kb,kc,task) / eijkabc
                    energy_t += symm_kpt * einsum('abcijk,abcijk', pwijk, rwijk.conj())

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
        kb = kconserv[ki,ka,kj]
        # <bj|ai> -> (ba|ji)    (Physicist->Chemist)
        # (ij|ab) = (ba|ij)*    (Permutational symmetry)
        # out = (ij|ab).transpose(0,1,3,2)
        out[ki,kj,kb] = ooov[kb,kj,ka].conj().transpose(3,1,0,2)
    return out


def create_t3_eris(nkpts, nocc, nvir, kconserv, tmpfile='tmp_t3_eris.h5', dtype=np.complex):
    '''Create/transpose necessary eri integrals needed for fast read-in by CCSD(T).'''
    nmo = nocc + nvir
    feri_tmp = None
    h5py_kwargs = {}
    feri_tmp_filename = tmpfile
    if not check_read_success(feri_tmp_filename):
        feri_tmp = lib.H5TmpFile(feri_tmp_filename, 'w', **h5py_kwargs)
        t2T_out = feri_tmp.create_dataset('t2T',
                      (nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc), dtype=dtype)
        eris_vvop_out = feri_tmp.create_dataset('vvop',
                            (nkpts,nkpts,nkpts,nvir,nvir,nocc,nmo), dtype=dtype)
        eris_vooo_C_out = feri_tmp.create_dataset('vooo_C',
                              (nkpts,nkpts,nkpts,nvir,nocc,nocc,nocc), dtype=dtype)

        transpose_t2(t2, nkpts, nocc, nvir, kconserv, out=t2T_out)
        create_eris_vvop(eris.vovv, eris.oovv, nkpts, nocc, nvir, kconserv, out=eris_vvop_out)
        create_eris_vooo(eris.ooov, nkpts, nocc, nvir, kconserv, out=eris_vooo_C_out)

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
        eris_vooo_C[:]

    return feri_tmp, t2T, eris_vvop, eris_vooo_C


# Gamma point calculation
#
# Parameters
# ----------
#     mesh : [24, 24, 24]
#     kpt  : [1, 1, 2]
# Returns
# -------
#     SCF     : -8.65192329453 Hartree per cell
#     CCSD    : -0.15529836941 Hartree per cell
#     CCSD(T) : -0.00191451068 Hartree per cell

# Gamma point calculation
#
# Parameters
# ----------
#     mesh : [24, 24, 24]
#     kpt  : [1, 1, 3]
# Returns
# -------
#     SCF     : -9.45719492074 Hartree per cell
#     CCSD    : -0.16615913445 Hartree per cell
#     CCSD(T) : -0.00403785264 Hartree per cell

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
    cell.conv_tol = 1e-12
    cell.conv_tol_grad = 1e-12
    cell.direct_scf_tol = 1e-16
    cell.unit = 'B'
    cell.verbose = 4
    cell.mesh = [24, 24, 24]
    cell.build()

    kpts = cell.make_kpts([1, 1, 3])
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    eris = mycc.ao2mo()
    ecc, t1, t2 = mycc.kernel(eris=eris)
    energy_t = kernel(mycc, eris=eris, verbose=9)
