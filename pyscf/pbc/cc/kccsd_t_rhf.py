#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#
"""Module for running restricted closed-shell k-point ccsd(t)"""

import itertools
import numpy as np
import pyscf.pbc.cc.kccsd_rhf

from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.misc import tril_product
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.misc import flatten
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.lib.numpy_helper import pack_tril
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM

#einsum = np.einsum
einsum = lib.einsum


# CCSD(T) equations taken from Scuseria, JCP (94), 1991
#
# NOTE: As pointed out in cc/ccsd_t_slow.py, there is an error in this paper
#     and the equation should read [ia] >= [jb] >= [kc] (since the only
#     symmetry in spin-less operators is the exchange of a column of excitation
#     ooperators).
def kernel(mycc, eris=None, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
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
        ret = einsum('kjf,fi->ijk', t2[kk, kj, kc, :, :, c, :], -eris.vovv[kf, ki, kb, :, :, b, a].conj())
        ret = ret - einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, c], -eris.ooov[kj, ki, km, :, :, :, a].conj())
        return ret

    def get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Pijkabc operating on Wijkabc intermediate as described in Scuseria paper'''
        ret = get_w(ki, kj, kk, ka, kb, kc, a, b, c)
        ret = ret + get_w(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
        ret = ret + get_w(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
        ret = ret + get_w(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
        ret = ret + get_w(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
        ret = ret + get_w(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)
        return ret

    def get_rw(ki, kj, kk, ka, kb, kc, a, b, c):
        '''R operating on Wijkabc intermediate as described in Scuseria paper'''
        ret = (4. * get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c) +
               1. * get_permuted_w(kk, ki, kj, ka, kb, kc, a, b, c).transpose(1, 2, 0) +
               1. * get_permuted_w(kj, kk, ki, ka, kb, kc, a, b, c).transpose(2, 0, 1) -
               2. * get_permuted_w(kk, kj, ki, ka, kb, kc, a, b, c).transpose(2, 1, 0) -
               2. * get_permuted_w(ki, kk, kj, ka, kb, kc, a, b, c).transpose(0, 2, 1) -
               2. * get_permuted_w(kj, ki, kk, ka, kb, kc, a, b, c).transpose(1, 0, 2))
        return ret

    def get_v(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[ki, ka, kj]
        ret = np.zeros((nocc, nocc, nocc), dtype=dtype)
        if kk == kc:
            ret = ret + einsum('k,ij->ijk', t1[kk, :, c], -eris.oovv[ki, kj, ka, :, :, a, b].conj())
            ret = ret + einsum('k,ij->ijk', fov[kk, :, c], t2[ki, kj, ka, :, :, a, b])
        return ret

    def get_permuted_v(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Pijkabc operating on Vijkabc intermediate as described in Scuseria paper'''
        ret = get_v(ki, kj, kk, ka, kb, kc, a, b, c)
        ret = ret + get_v(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
        ret = ret + get_v(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
        ret = ret + get_v(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
        ret = ret + get_v(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
        ret = ret + get_v(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)
        return ret

    energy_t = 0.0

    for ki in range(nkpts):
        for kj in range(ki + 1):
            for kk in range(kj + 1):
                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = lib.direct_sum('i,j,k->ijk', mo_energy_occ[ki], mo_energy_occ[kj], mo_energy_occ[kk])

                for ka in range(nkpts):
                    for kb in range(nkpts):
                        # Find momentum conservation condition for triples
                        # amplitude t3ijkabc
                        kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])

                        ia_index = ki * nkpts + ka
                        jb_index = kj * nkpts + kb
                        kc_index = kk * nkpts + kc
                        if not (ia_index >= jb_index and jb_index >= kc_index):
                            continue

                        # Factors to include for permutational symmetry among k-points
                        if (ia_index == jb_index and jb_index == kc_index):
                            symm_kpt = 1.  # only one unique [ia, jb, kc] index
                        elif (ia_index == jb_index or jb_index == kc_index):
                            symm_kpt = 3.  # three unique permutations of [ia, jb, kc]
                        else:
                            symm_kpt = 6.  # six unique permutations of [ia, jb, kc]

                        # Determine the a, b, c indices we will loop over as
                        # determined by the k-point symmetry.
                        abc_indices = cartesian_prod([range(nvir)] * 3)
                        symm_3d = symm_2d_ab = symm_2d_bc = False
                        if ia_index == jb_index == kc_index:  # ka == kb == kc
                            symm_3d = True
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[0, 1, 2])  # loop a >= b >= c
                            symm_3d = True
                        elif ia_index == jb_index:  # ka == kb
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[0, 1])  # loop a >= b
                            symm_2d_ab = True
                        elif jb_index == kc_index:
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[1, 2])  # loop b >= c
                            symm_2d_bc = True

                        for a, b, c in abc_indices:
                            # Form energy denominator
                            eijkabc = (eijk - mo_energy_vir[ka][a] - mo_energy_vir[kb][b] - mo_energy_vir[kc][c])
                            # When padding for non-equal nocc per k-point, some fock elements will be zero
                            idx = np.where(abs(eijkabc) < LOOSE_ZERO_TOL)[0]
                            eijkabc[idx] = LARGE_DENOM

                            # See symm_3d and abc_indices above for description of factors
                            symm_abc = 1.
                            if symm_3d:
                                if a == b == c:
                                    symm_abc = 1.
                                elif a == b or b == c:
                                    symm_abc = 3.
                                else:
                                    symm_abc = 6.
                            elif symm_2d_ab:
                                if a == b:
                                    symm_abc = 1.
                                else:
                                    symm_abc = 2.
                            elif symm_2d_bc:
                                if b == c:
                                    symm_abc = 1.
                                else:
                                    symm_abc = 2.

                            # The simplest written algorithm can be accomplished with the following four lines

                            #pwijk = (       get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c) +
                            #          0.5 * get_permuted_v(ki, kj, kk, ka, kb, kc, a, b, c) )
                            #rwijk = get_rw(ki, kj, kk, ka, kb, kc, a, b, c) / eijkabc
                            #energy_t += symm_fac * einsum('ijk,ijk', pwijk, rwijk.conj())

                            # Creating permuted W_ijkabc intermediate
                            w_int0 = get_w(ki, kj, kk, ka, kb, kc, a, b, c)
                            w_int1 = get_w(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
                            w_int2 = get_w(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
                            w_int3 = get_w(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
                            w_int4 = get_w(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
                            w_int5 = get_w(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)

                            # Creating permuted V_ijkabc intermediate
                            v_int0 = get_v(ki, kj, kk, ka, kb, kc, a, b, c)
                            v_int1 = get_v(kj, kk, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1)
                            v_int2 = get_v(kk, ki, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0)
                            v_int3 = get_v(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
                            v_int4 = get_v(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
                            v_int5 = get_v(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)

                            # Creating permuted W_ijkabc + 0.5 * V_ijkabc intermediate
                            pwijk = w_int0 + 0.5 * v_int0
                            pwijk += w_int1 + 0.5 * v_int1
                            pwijk += w_int2 + 0.5 * v_int2
                            pwijk += w_int3 + 0.5 * v_int3
                            pwijk += w_int4 + 0.5 * v_int4
                            pwijk += w_int5 + 0.5 * v_int5

                            # Creating R[W] intermediate
                            rwijk = np.zeros((nocc, nocc, nocc), dtype=dtype)

                            # Adding in contribution 4. * P[(i, j, k) -> (i, j, k)]
                            rwijk += 4. * w_int0
                            rwijk += 4. * w_int1
                            rwijk += 4. * w_int2
                            rwijk += 4. * w_int3
                            rwijk += 4. * w_int4
                            rwijk += 4. * w_int5

                            # Adding in contribution 1. * P[(i, j, k) -> (k, i, j)]
                            rwijk += 1. * get_w(kk, ki, kj, ka, kb, kc, a, b, c).transpose(1, 2, 0)
                            rwijk += 1. * get_w(ki, kj, kk, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(1, 2, 0)
                            rwijk += 1. * get_w(kj, kk, ki, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(1, 2, 0)
                            rwijk += 1. * get_w(kk, kj, ki, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(1, 2, 0)
                            rwijk += 1. * get_w(kj, ki, kk, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(1, 2, 0)
                            rwijk += 1. * get_w(ki, kk, kj, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(1, 2, 0)

                            # Adding in contribution 1. * P[(i, j, k) -> (j, k, i)]
                            rwijk += 1. * get_w(kj, kk, ki, ka, kb, kc, a, b, c).transpose(2, 0, 1)
                            rwijk += 1. * get_w(kk, ki, kj, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(2, 0, 1)
                            rwijk += 1. * get_w(ki, kj, kk, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(2, 0, 1)
                            rwijk += 1. * get_w(kj, ki, kk, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(2, 0, 1)
                            rwijk += 1. * get_w(ki, kk, kj, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(2, 0, 1)
                            rwijk += 1. * get_w(kk, kj, ki, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(2, 0, 1)

                            # Adding in contribution -2. * P[(i, j, k) -> (k, j, i)]
                            rwijk += -2. * get_w(kk, kj, ki, ka, kb, kc, a, b, c).transpose(2, 1, 0)
                            rwijk += -2. * get_w(kj, ki, kk, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(2, 1, 0)
                            rwijk += -2. * get_w(ki, kk, kj, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(2, 1, 0)
                            rwijk += -2. * get_w(kk, ki, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(2, 1, 0)
                            rwijk += -2. * get_w(ki, kj, kk, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(2, 1, 0)
                            rwijk += -2. * get_w(kj, kk, ki, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(2, 1, 0)

                            # Adding in contribution -2. * P[(i, j, k) -> (i, k, j)]
                            rwijk += -2. * get_w(ki, kk, kj, ka, kb, kc, a, b, c).transpose(0, 2, 1)
                            rwijk += -2. * get_w(kk, kj, ki, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(0, 2, 1)
                            rwijk += -2. * get_w(kj, ki, kk, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(0, 2, 1)
                            rwijk += -2. * get_w(ki, kj, kk, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(0, 2, 1)
                            rwijk += -2. * get_w(kj, kk, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(0, 2, 1)
                            rwijk += -2. * get_w(kk, ki, kj, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(0, 2, 1)

                            # Adding in contribution -2. * P[(i, j, k) -> (j, i, k)]
                            rwijk += -2. * get_w(kj, ki, kk, ka, kb, kc, a, b, c).transpose(1, 0, 2)
                            rwijk += -2. * get_w(ki, kk, kj, kb, kc, ka, b, c, a).transpose(2, 0, 1).transpose(1, 0, 2)
                            rwijk += -2. * get_w(kk, kj, ki, kc, ka, kb, c, a, b).transpose(1, 2, 0).transpose(1, 0, 2)
                            rwijk += -2. * get_w(kj, kk, ki, ka, kc, kb, a, c, b).transpose(0, 2, 1).transpose(1, 0, 2)
                            rwijk += -2. * get_w(kk, ki, kj, kc, kb, ka, c, b, a).transpose(2, 1, 0).transpose(1, 0, 2)
                            rwijk += -2. * get_w(ki, kj, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2).transpose(1, 0, 2)

                            rwijk /= eijkabc

                            energy_t += symm_abc * symm_kpt * einsum('ijk,ijk', pwijk, rwijk.conj())

    energy_t *= (1. / 3)
    energy_t /= nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s', energy_t.imag)
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

    kpts = cell.make_kpts([1, 1, 3])
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()
    energy_t = kernel(mycc)
