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

    cpu1 = log.timer_debug1('CCSD(T) tmp eri creation', *cpu1)

    def get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1, out=None):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[kk, kc, kj]
        ret = einsum('kjcf,fiba->abcijk', t2[kk,kj,kc,:,:,c0:c1,:], eris.vovv[kf,ki,kb,:,:,b0:b1,a0:a1].conj())
        ret = ret - einsum('mkbc,jima->abcijk', t2[km,kk,kb,:,:,b0:b1,c0:c1], eris.ooov[kj,ki,km,:,:,:,a0:a1].conj())
        return ret

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

    def get_v(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki,ka,kj]
        kf = kconserv[ki,ka,kj]
        out = np.zeros((a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)
        if kk == kc:
            out = out + einsum('kc,ijab->abcijk', t1[kk,:,c0:c1], eris.oovv[ki,kj,ka,:,:,a0:a1,b0:b1].conj())
            out = out + einsum('kc,ijab->abcijk', fov[kk,:,c0:c1], t2[ki,kj,ka,:,:,a0:a1,b0:b1])
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

                eabc = LARGE_DENOM * np.ones((nvir,)*3, dtype=mo_energy_occ[0].dtype)
                n0_ovp_abc = np.ix_(nonzero_vpadding[ka], nonzero_vpadding[kb], nonzero_vpadding[kc])
                eabc[n0_ovp_abc] = lib.direct_sum('i,j,k->ijk', mo_energy_vir[ka], mo_energy_vir[kb], mo_energy_vir[kc])[n0_ovp_abc]
                for task_id, task in enumerate(tasks):
                    orb_indices = a0,a1,b0,b1,c0,c1
                    eijkabc = (eijk[None,None,None,:,:,:] - eabc[a0:a1,b0:b1,c0:c1,None,None,None])
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
