#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#
"""Module for running k-point ccsd(t)"""

import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc import scf
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nocc, get_nmo
from pyscf.lib import linalg_helper
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools.pbc import super_cell

#einsum = np.einsum
einsum = lib.einsum

# CCSD(T) equations taken from Scuseria, JCP (94), 1991
#
# NOTE: As pointed out in cc/ccsd_t_slow.py, there is an error in this paper
#     and the equation should read [ia] >= [jb] >= [kc] (since the only
#     symmetry in spin-less operators is the exchange of a column of excitation
#     ooperators).
def kernel(mycc, eris=None, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    '''Returns the k-point CCSD(T) for a closed-shell system using spatial orbitals.

    Note:
        Returns real part of the CCSD(T) energy, raises warning if there is
        a complex part.

    Args:
        mycc (:class:`GCCSD`): Coupled-cluster object storing results of
            a coupled-cluster calculation.
        eris (:class:`_ERIS`): Integral object holding the relevant electron-
            repulsion integrals and Fock matrix elements
        t1 (:obj:`ndarray`): t1 restricted coupled-cluster amplitudes
        t2 (:obj:`ndarray`): t2 restricted coupled-cluster amplitudes
        max_memory (float): Maximum memory used in calculation
        verbose (int, :class:`Logger`) : verbosity of calculation

    Returns:
        energy_t : float
            The real-part of the k-point CCSD(T) energy.
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if eris is None: eris = mycc.eris
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

    mo_energy_occ = [eris.fock[i].diagonal()[:nocc] for i in range(nkpts)]
    mo_energy_vir = [eris.fock[i].diagonal()[nocc:] for i in range(nkpts)]
    fov = eris.fock[:, :nocc, nocc:]

    # Set up class for k-point conservation
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    def get_w(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[kk, kc, kj]
        ret =       einsum('kjf,fi->ijk', t2[kk, kj, kc, :, :, c, :], -eris.vovv[kf, ki, kb, :, :, b, a].conj())
        ret = ret - einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, c], -eris.ooov[kj, ki, km, :, :, :, a].conj())
        return ret

    def get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Pijkabc operating on Wijkabc intermediate as described in Scuseria paper'''
        ret =       get_w(ki, kj, kk, ka, kb, kc, a, b, c)
        ret = ret + get_w(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
        ret = ret + get_w(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
        ret = ret + get_w(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
        ret = ret + get_w(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
        ret = ret + get_w(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)
        return ret

    def get_rw(ki, kj, kk, ka, kb, kc, a, b, c):
        '''R operating on Wijkabc intermediate as described in Scuseria paper'''
        ret = ( 4. * get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c) +
                1. * get_permuted_w(kk, ki, kj, ka, kb, kc, a, b, c).transpose(1, 2, 0) +
                1. * get_permuted_w(kj, kk, ki, ka, kb, kc, a, b, c).transpose(2, 0, 1) -
                2. * get_permuted_w(kk, kj, ki, ka, kb, kc, a, b, c).transpose(2, 1, 0) -
                2. * get_permuted_w(ki, kk, kj, ka, kb, kc, a, b, c).transpose(0, 2, 1) -
                2. * get_permuted_w(kj, ki, kk, ka, kb, kc, a, b, c).transpose(1, 0, 2) )
        return ret

    def get_v(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[ki, ka, kj]
        ret = np.zeros((nocc, nocc, nocc), dtype=dtype)
        if kk == kc:
            ret = ret + einsum('k,ij->ijk',  t1[kk, :, c], -eris.oovv[ki, kj, ka, :, :, a, b].conj())
            ret = ret + einsum('k,ij->ijk', fov[kk, :, c],         t2[ki, kj, ka, :, :, a, b])
        return ret

    def get_permuted_v(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Pijkabc operating on Vijkabc intermediate as described in Scuseria paper'''
        ret =       get_v(ki, kj, kk, ka, kb, kc, a, b, c)
        ret = ret + get_v(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
        ret = ret + get_v(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
        ret = ret + get_v(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
        ret = ret + get_v(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
        ret = ret + get_v(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)
        return ret

    energy_t = 0.0

    for ki in range(nkpts):
        for kj in range(ki+1):
            for kk in range(kj+1):
                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = lib.direct_sum('i,j,k->ijk', mo_energy_occ[ki], mo_energy_occ[kj], mo_energy_occ[kk])

                for ka in range(nkpts):
                    for kb in range(nkpts):
                        # Find momentum conservation condition for triples
                        # amplitude t3ijkabc
                        kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])

                        ia_index = ki*nkpts + ka
                        jb_index = kj*nkpts + kb
                        kc_index = kk*nkpts + kc
                        if not (ia_index >= jb_index and
                                jb_index >= kc_index):
                            continue

                        # Factors to include for symmetry
                        if (ia_index == jb_index and
                            jb_index == kc_index):
                            symm_fac = 1.  # only one unique [ia, jb, kc] index
                        elif (ia_index == jb_index or
                              jb_index == kc_index):
                            symm_fac = 3.  # three unique permutations of [ia, jb, kc]
                        else:
                            symm_fac = 6.  # six unique permutations of [ia, jb, kc]

                        for a in range(nvir):
                            for b in range(nvir):
                                for c in range(nvir):
                                    # Form energy denominator
                                    eijkabc = (eijk - mo_energy_vir[ka][a] - mo_energy_vir[kb][b] - mo_energy_vir[kc][c])

                                    pwijk = (       get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c) +
                                              0.5 * get_permuted_v(ki, kj, kk, ka, kb, kc, a, b, c) )
                                    rwijk = get_rw(ki, kj, kk, ka, kb, kc, a, b, c) / eijkabc

                                    energy_t += symm_fac * einsum('ijk,ijk', pwijk, rwijk.conj())

                                    #w_int0 = get_w(ki, kj, kk, ka, kb, kc, a, b, c)
                                    #w_int1 = get_w(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
                                    #w_int2 = get_w(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
                                    #w_int3 = get_w(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
                                    #w_int4 = get_w(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
                                    #w_int5 = get_w(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)

                                    #v_int0 = get_v(ki, kj, kk, ka, kb, kc, a, b, c)
                                    #v_int1 = get_v(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
                                    #v_int2 = get_v(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
                                    #v_int3 = get_v(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
                                    #v_int4 = get_v(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
                                    #v_int5 = get_v(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)

                                    ## Creating permuted W_ijkabc + V_ijkabc intermediate
                                    #pwijk  = w_int0 + 0.5 * v_int0
                                    #pwijk += w_int1 + 0.5 * v_int1
                                    #pwijk += w_int2 + 0.5 * v_int2
                                    #pwijk += w_int3 + 0.5 * v_int3
                                    #pwijk += w_int4 + 0.5 * v_int4
                                    #pwijk += w_int5 + 0.5 * v_int5

                                    #rwijk = np.zeros_like(w_int0)

                                    ## (i, j, k) -> (i, j, k)
                                    #rwijk += 4. * w_int0
                                    #rwijk += 4. * w_int1
                                    #rwijk += 4. * w_int2
                                    #rwijk += 4. * w_int3
                                    #rwijk += 4. * w_int4
                                    #rwijk += 4. * w_int5

                                    ## (i, j, k) -> (k, i, j)
                                    #rwijk += 1. * get_w(kk, ki, kj, ka, kb, kc, a, b, c).transpose(1, 2, 0)
                                    #rwijk += 1. * get_w(ki, kj, kk, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(1, 2, 0)
                                    #rwijk += 1. * get_w(kj, kk, ki, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(1, 2, 0)
                                    #rwijk += 1. * get_w(kk, kj, ki, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(1, 2, 0)
                                    #rwijk += 1. * get_w(kj, ki, kk, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(1, 2, 0)
                                    #rwijk += 1. * get_w(ki, kk, kj, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(1, 2, 0)

                                    ## (i, j, k) -> (j, k, i)
                                    #rwijk += 1. * get_w(kj, kk, ki, ka, kb, kc, a, b, c).transpose(2, 0, 1)
                                    #rwijk += 1. * get_w(kk, ki, kj, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(2, 0, 1)
                                    #rwijk += 1. * get_w(ki, kj, kk, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(2, 0, 1)
                                    #rwijk += 1. * get_w(kj, ki, kk, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(2, 0, 1)
                                    #rwijk += 1. * get_w(ki, kk, kj, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(2, 0, 1)
                                    #rwijk += 1. * get_w(kk, kj, ki, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(2, 0, 1)

                                    #rwijk += get_rw(ki, kj, kk, ka, kb, kc, a, b, c)

                                    #rwijk /= eijkabc

                                    #energy_t += symm_fac * einsum('ijk,ijk', pwijk, rwijk.conj())

    energy_t *= (1./3)
    energy_t /= nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s',
                 energy_t.imag)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
    log.note('CCSD(T) correction per cell (imag) = %.15g', energy_t.imag)
    return energy_t.real

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
    cell.verbose = 5
    cell.mesh = [24, 24, 24]
    cell.build()


    #mk_mesh = [1, 1, 3]
    #scell = super_cell(cell, mk_mesh)
    #mf = scf.RHF(scell, exxdiv=None)
    #mf.kernel()
    #mycc = pyscf.cc.RCCSD(mf)
    #ecc, t1, t2 = mycc.kernel()
    #energy_t = mycc.ccsd_t()
    #print "ccsd(t) energy per cell = ", energy_t / np.prod(mk_mesh)
    kpts = cell.make_kpts([1, 1, 2])
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()
    energy_t = kernel(mycc)
