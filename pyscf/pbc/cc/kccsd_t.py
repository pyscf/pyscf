#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#
"""Module for running k-point ccsd(t)"""

import numpy as np
import pyscf.pbc.cc.kccsd

from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.misc import tril_product
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

#einsum = np.einsum
einsum = lib.einsum


# CCSD(T) equations taken from Tu, Yang, Wang, and Guo JPC (135), 2011
#
# There are some complex conjugates not included in the equations
# by Watts, Gauss, Bartlett JCP (98), 1993
def kernel(mycc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    '''Returns the CCSD(T) for general spin-orbital integrals with k-points.

    Note:
        Returns real part of the CCSD(T) energy, raises warning if there is
        a complex part.

    Args:
        mycc (:class:`GCCSD`): Coupled-cluster object storing results of
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
    assert isinstance(mycc, pyscf.pbc.cc.kccsd.GCCSD)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

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

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    # Set up class for k-point conservation
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    energy_t = 0.0

    for ki in range(nkpts):
        for kj in range(ki + 1):
            for kk in range(kj + 1):
                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                                 [0,nocc,kj,mo_e_o,nonzero_opadding],
                                 [0,nocc,kk,mo_e_o,nonzero_opadding])

                # Factors to include for permutational symmetry among k-points for occupied space
                if ki == kj and kj == kk:
                    symm_ijk_kpt = 1.  # only one degeneracy
                elif ki == kj or kj == kk:
                    symm_ijk_kpt = 3.  # 3 degeneracies when only one k-point is unique
                else:
                    symm_ijk_kpt = 6.  # 3! combinations of arranging 3 distinct k-points

                for ka in range(nkpts):
                    for kb in range(ka + 1):

                        # Find momentum conservation condition for triples
                        # amplitude t3ijkabc
                        kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                        if kc not in range(kb + 1):
                            continue

                        # -1.0 so the LARGE_DENOM does not cancel with the one from eijk
                        eabc = _get_epqr([0,nvir,ka,mo_e_v,nonzero_vpadding],
                                         [0,nvir,kb,mo_e_v,nonzero_vpadding],
                                         [0,nvir,kc,mo_e_v,nonzero_vpadding],
                                         fac=[-1.,-1.,-1.])

                        # Factors to include for permutational symmetry among k-points for virtual space
                        if ka == kb and kb == kc:
                            symm_abc_kpt = 1.  # one unique combination of (ka, kb, kc)
                        elif ka == kb or kb == kc:
                            symm_abc_kpt = 3.  # 3 unique combinations of (ka, kb, kc)
                        else:
                            symm_abc_kpt = 6.  # 6 unique combinations of (ka, kb, kc)

                        # Determine the a, b, c indices we will loop over as
                        # determined by the k-point symmetry.
                        abc_indices = cartesian_prod([range(nvir)] * 3)
                        symm_3d = symm_2d_ab = symm_2d_bc = False
                        if ka == kc:  # == kb from lower triangular summation
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[0, 1, 2])  # loop a >= b >= c
                            symm_3d = True
                        elif ka == kb:
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[0, 1])  # loop a >= b
                            symm_2d_ab = True
                        elif kb == kc:
                            abc_indices = tril_product(range(nvir), repeat=3, tril_idx=[1, 2])  # loop b >= c
                            symm_2d_bc = True

                        for a, b, c in abc_indices:
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

                            # Form energy denominator
                            eijkabc = (eijk[:,:,:] + eabc[a,b,c])

                            # Form connected triple excitation amplitude
                            t3c = np.zeros((nocc, nocc, nocc), dtype=dtype)

                            # First term: 1 - p(ij) - p(ik)
                            ke = kconserv[kj, ka, kk]
                            t3c = t3c + einsum('jke,ie->ijk', t2[kj, kk, ka, :, :, a, :], -eris.ovvv[ki, ke, kc, :, :, c, b].conj())
                            ke = kconserv[ki, ka, kk]
                            t3c = t3c - einsum('ike,je->ijk', t2[ki, kk, ka, :, :, a, :], -eris.ovvv[kj, ke, kc, :, :, c, b].conj())
                            ke = kconserv[kj, ka, ki]
                            t3c = t3c - einsum('jie,ke->ijk', t2[kj, ki, ka, :, :, a, :], -eris.ovvv[kk, ke, kc, :, :, c, b].conj())

                            km = kconserv[kb, ki, kc]
                            t3c = t3c - einsum('mi,jkm->ijk', t2[km, ki, kb, :, :, b, c], eris.ooov[kj, kk, km, :, :, :, a].conj())
                            km = kconserv[kb, kj, kc]
                            t3c = t3c + einsum('mj,ikm->ijk', t2[km, kj, kb, :, :, b, c], eris.ooov[ki, kk, km, :, :, :, a].conj())
                            km = kconserv[kb, kk, kc]
                            t3c = t3c + einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, c], eris.ooov[kj, ki, km, :, :, :, a].conj())

                            # Second term: - p(ab) + p(ab) p(ij) + p(ab) p(ik)
                            ke = kconserv[kj, kb, kk]
                            t3c = t3c - einsum('jke,ie->ijk', t2[kj, kk, kb, :, :, b, :], -eris.ovvv[ki, ke, kc, :, :, c, a].conj())
                            ke = kconserv[ki, kb, kk]
                            t3c = t3c + einsum('ike,je->ijk', t2[ki, kk, kb, :, :, b, :], -eris.ovvv[kj, ke, kc, :, :, c, a].conj())
                            ke = kconserv[kj, kb, ki]
                            t3c = t3c + einsum('jie,ke->ijk', t2[kj, ki, kb, :, :, b, :], -eris.ovvv[kk, ke, kc, :, :, c, a].conj())

                            km = kconserv[ka, ki, kc]
                            t3c = t3c + einsum('mi,jkm->ijk', t2[km, ki, ka, :, :, a, c], eris.ooov[kj, kk, km, :, :, :, b].conj())
                            km = kconserv[ka, kj, kc]
                            t3c = t3c - einsum('mj,ikm->ijk', t2[km, kj, ka, :, :, a, c], eris.ooov[ki, kk, km, :, :, :, b].conj())
                            km = kconserv[ka, kk, kc]
                            t3c = t3c - einsum('mk,jim->ijk', t2[km, kk, ka, :, :, a, c], eris.ooov[kj, ki, km, :, :, :, b].conj())

                            # Third term: - p(ac) + p(ac) p(ij) + p(ac) p(ik)
                            ke = kconserv[kj, kc, kk]
                            t3c = t3c - einsum('jke,ie->ijk', t2[kj, kk, kc, :, :, c, :], -eris.ovvv[ki, ke, ka, :, :, a, b].conj())
                            ke = kconserv[ki, kc, kk]
                            t3c = t3c + einsum('ike,je->ijk', t2[ki, kk, kc, :, :, c, :], -eris.ovvv[kj, ke, ka, :, :, a, b].conj())
                            ke = kconserv[kj, kc, ki]
                            t3c = t3c + einsum('jie,ke->ijk', t2[kj, ki, kc, :, :, c, :], -eris.ovvv[kk, ke, ka, :, :, a, b].conj())

                            km = kconserv[kb, ki, ka]
                            t3c = t3c + einsum('mi,jkm->ijk', t2[km, ki, kb, :, :, b, a], eris.ooov[kj, kk, km, :, :, :, c].conj())
                            km = kconserv[kb, kj, ka]
                            t3c = t3c - einsum('mj,ikm->ijk', t2[km, kj, kb, :, :, b, a], eris.ooov[ki, kk, km, :, :, :, c].conj())
                            km = kconserv[kb, kk, ka]
                            t3c = t3c - einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, a], eris.ooov[kj, ki, km, :, :, :, c].conj())

                            # Form disconnected triple excitation amplitude contribution
                            t3d = np.zeros((nocc, nocc, nocc), dtype=dtype)

                            # First term: 1 - p(ij) - p(ik)
                            if ki == ka:
                                t3d = t3d + einsum('i,jk->ijk',  t1[ki, :, a], -eris.oovv[kj, kk, kb, :, :, b, c].conj())
                                t3d = t3d + einsum('i,jk->ijk',-fov[ki, :, a],         t2[kj, kk, kb, :, :, b, c])

                            if kj == ka:
                                t3d = t3d - einsum('j,ik->ijk',  t1[kj, :, a], -eris.oovv[ki, kk, kb, :, :, b, c].conj())
                                t3d = t3d - einsum('j,ik->ijk',-fov[kj, :, a],         t2[ki, kk, kb, :, :, b, c])

                            if kk == ka:
                                t3d = t3d - einsum('k,ji->ijk',  t1[kk, :, a], -eris.oovv[kj, ki, kb, :, :, b, c].conj())
                                t3d = t3d - einsum('k,ji->ijk',-fov[kk, :, a],         t2[kj, ki, kb, :, :, b, c])

                            # Second term: - p(ab) + p(ab) p(ij) + p(ab) p(ik)
                            if ki == kb:
                                t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, b], -eris.oovv[kj, kk, ka, :, :, a, c].conj())
                                t3d = t3d - einsum('i,jk->ijk',-fov[ki, :, b],         t2[kj, kk, ka, :, :, a, c])

                            if kj == kb:
                                t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, b], -eris.oovv[ki, kk, ka, :, :, a, c].conj())
                                t3d = t3d + einsum('j,ik->ijk',-fov[kj, :, b],         t2[ki, kk, ka, :, :, a, c])

                            if kk == kb:
                                t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, b], -eris.oovv[kj, ki, ka, :, :, a, c].conj())
                                t3d = t3d + einsum('k,ji->ijk',-fov[kk, :, b],         t2[kj, ki, ka, :, :, a, c])

                            # Third term: - p(ac) + p(ac) p(ij) + p(ac) p(ik)
                            if ki == kc:
                                t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, c], -eris.oovv[kj, kk, kb, :, :, b, a].conj())
                                t3d = t3d - einsum('i,jk->ijk',-fov[ki, :, c],         t2[kj, kk, kb, :, :, b, a])

                            if kj == kc:
                                t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, c], -eris.oovv[ki, kk, kb, :, :, b, a].conj())
                                t3d = t3d + einsum('j,ik->ijk',-fov[kj, :, c],         t2[ki, kk, kb, :, :, b, a])

                            if kk == kc:
                                t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, c], -eris.oovv[kj, ki, kb, :, :, b, a].conj())
                                t3d = t3d + einsum('k,ji->ijk',-fov[kk, :, c],         t2[kj, ki, kb, :, :, b, a])

                            t3c_plus_d = t3c + t3d
                            t3c_plus_d /= eijkabc

                            energy_t += symm_abc_kpt * symm_ijk_kpt * symm_abc * einsum('ijk,ijk', t3c, t3c_plus_d.conj())

    energy_t = (1. / 36) * energy_t / nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s',
                 energy_t.imag)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
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

if __name__ == '__main__':
    from pyscf.pbc import gto
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
    cell.verbose = 5
    cell.mesh = [24, 24, 24]
    cell.build()

    kpts = cell.make_kpts([1, 1, 3])
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KGCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()

    energy_t = kernel(mycc)
