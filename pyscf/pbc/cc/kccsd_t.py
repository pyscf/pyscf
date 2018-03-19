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

# CCSD(T) equations taken from Tu, Yang, Wang, and Guo JPC (135), 2011
#
# There are some complex conjugates not included in the equations
# by Watts, Gauss, Bartlett JCP (98), 1993
def kernel(mycc, eris=None, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    """
    This function returns the CCSD(T) energy.
    """
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if eris is None: eris = mycc.eris
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

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

    energy_t = 0.0

    symm_on = True
    for ki in range(nkpts):
        for kj in range(ki+1):
            for kk in range(kj+1):

                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = lib.direct_sum('i,j,k->ijk', mo_energy_occ[ki], mo_energy_occ[kj], mo_energy_occ[kk])

                # count the degeneracy of all (ki, kj, kk)
                if ki == kj and kj == kk:
                    symm_ijk = 1.  # only one degeneracy
                elif ki == kj or kj == kk:
                    symm_ijk = 3.  # 3 degeneracies when only one k-point is unique
                else:
                    symm_ijk = 6.  # 3! combinations of arranging 3 distinct k-points

                e_cont = 0.0
                for ka in range(nkpts):
                    for kb in range(ka+1):

                        # Find momentum conservation condition for triples
                        # amplitude t3ijkabc
                        kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                        if kc not in range(kb+1):
                            continue

                        # count the degeneracy of all (ka, kb, kc)
                        if ka == kb and kb == kc:
                            symm_abc = 1.  # only one degeneracy
                        elif ka == kb or kb == kc:
                            symm_abc = 3.  # 3 degeneracies when only one k-point is unique
                        else:
                            symm_abc = 6.  # 3! combinations of arranging 3 distinct k-points

                        for a in range(nvir):
                            for b in range(nvir):
                                for c in range(nvir):

                                    # Form energy denominator
                                    eijkabc = (eijk - mo_energy_vir[ka][a] - mo_energy_vir[kb][b] - mo_energy_vir[kc][c])

                                    # Form connected triple excitation amplitude
                                    t3c = np.zeros((nocc,nocc,nocc), dtype=dtype)

                                    # First term: 1 - p(ij) - p(ik)
                                    ke = kconserv[kj, ka, kk]
                                    t3c = t3c + einsum('jke,ie->ijk', t2[kj, kk, ka, :, :, a, :], eris.ovvv[ki, ke, kc, :, :, c, b].conj())
                                    ke = kconserv[ki, ka, kk]
                                    t3c = t3c - einsum('ike,je->ijk', t2[ki, kk, ka, :, :, a, :], eris.ovvv[kj, ke, kc, :, :, c, b].conj())
                                    ke = kconserv[kj, ka, ki]
                                    t3c = t3c - einsum('jie,ke->ijk', t2[kj, ki, ka, :, :, a, :], eris.ovvv[kk, ke, kc, :, :, c, b].conj())

                                    km = kconserv[kb, ki, kc]
                                    t3c = t3c - einsum('mi,jkm->ijk', t2[km, ki, kb, :, :, b, c], -eris.ooov[kj, kk, km, :, :, :, a].conj())
                                    km = kconserv[kb, kj, kc]
                                    t3c = t3c + einsum('mj,ikm->ijk', t2[km, kj, kb, :, :, b, c], -eris.ooov[ki, kk, km, :, :, :, a].conj())
                                    km = kconserv[kb, kk, kc]
                                    t3c = t3c + einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, c], -eris.ooov[kj, ki, km, :, :, :, a].conj())

                                    # Second term: - p(ab) + p(ab) p(ij) + p(ab) p(ik)
                                    ke = kconserv[kj, kb, kk]
                                    t3c = t3c - einsum('jke,ie->ijk', t2[kj, kk, kb, :, :, b, :], eris.ovvv[ki, ke, kc, :, :, c, a].conj())
                                    ke = kconserv[ki, kb, kk]
                                    t3c = t3c + einsum('ike,je->ijk', t2[ki, kk, kb, :, :, b, :], eris.ovvv[kj, ke, kc, :, :, c, a].conj())
                                    ke = kconserv[kj, kb, ki]
                                    t3c = t3c + einsum('jie,ke->ijk', t2[kj, ki, kb, :, :, b, :], eris.ovvv[kk, ke, kc, :, :, c, a].conj())

                                    km = kconserv[ka, ki, kc]
                                    t3c = t3c + einsum('mi,jkm->ijk', t2[km, ki, ka, :, :, a, c], -eris.ooov[kj, kk, km, :, :, :, b].conj())
                                    km = kconserv[ka, kj, kc]
                                    t3c = t3c - einsum('mj,ikm->ijk', t2[km, kj, ka, :, :, a, c], -eris.ooov[ki, kk, km, :, :, :, b].conj())
                                    km = kconserv[ka, kk, kc]
                                    t3c = t3c - einsum('mk,jim->ijk', t2[km, kk, ka, :, :, a, c], -eris.ooov[kj, ki, km, :, :, :, b].conj())

                                    # Third term: - p(ac) + p(ac) p(ij) + p(ac) p(ik)
                                    ke = kconserv[kj, kc, kk]
                                    t3c = t3c - einsum('jke,ie->ijk', t2[kj, kk, kc, :, :, c, :], eris.ovvv[ki, ke, ka, :, :, a, b].conj())
                                    ke = kconserv[ki, kc, kk]
                                    t3c = t3c + einsum('ike,je->ijk', t2[ki, kk, kc, :, :, c, :], eris.ovvv[kj, ke, ka, :, :, a, b].conj())
                                    ke = kconserv[kj, kc, ki]
                                    t3c = t3c + einsum('jie,ke->ijk', t2[kj, ki, kc, :, :, c, :], eris.ovvv[kk, ke, ka, :, :, a, b].conj())

                                    km = kconserv[kb, ki, ka]
                                    t3c = t3c + einsum('mi,jkm->ijk', t2[km, ki, kb, :, :, b, a], -eris.ooov[kj, kk, km, :, :, :, c].conj())
                                    km = kconserv[kb, kj, ka]
                                    t3c = t3c - einsum('mj,ikm->ijk', t2[km, kj, kb, :, :, b, a], -eris.ooov[ki, kk, km, :, :, :, c].conj())
                                    km = kconserv[kb, kk, ka]
                                    t3c = t3c - einsum('mk,jim->ijk', t2[km, kk, kb, :, :, b, a], -eris.ooov[kj, ki, km, :, :, :, c].conj())

                                    # Form disconnected triple excitation amplitude contribution
                                    t3d = np.zeros((nocc,nocc,nocc), dtype=dtype)

                                    # First term: 1 - p(ij) - p(ik)
                                    if ki == ka:
                                        t3d = t3d + einsum('i,jk->ijk',  t1[ki, :, a], eris.oovv[kj, kk, kb, :, :, b, c].conj())
                                        t3d = t3d + einsum('i,jk->ijk', fov[ki, :, a],        t2[kj, kk, kb, :, :, b, c])

                                    if kj == ka:
                                        t3d = t3d - einsum('j,ik->ijk',  t1[kj, :, a], eris.oovv[ki, kk, kb, :, :, b, c].conj())
                                        t3d = t3d - einsum('j,ik->ijk', fov[kj, :, a],        t2[ki, kk, kb, :, :, b, c])

                                    if kk == ka:
                                        t3d = t3d - einsum('k,ji->ijk',  t1[kk, :, a], eris.oovv[kj, ki, kb, :, :, b, c].conj())
                                        t3d = t3d - einsum('k,ji->ijk', fov[kk, :, a],        t2[kj, ki, kb, :, :, b, c])

                                    # Second term: - p(ab) + p(ab) p(ij) + p(ab) p(ik)
                                    if ki == kb:
                                        t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, b], eris.oovv[kj, kk, ka, :, :, a, c].conj())
                                        t3d = t3d - einsum('i,jk->ijk', fov[ki, :, b],        t2[kj, kk, ka, :, :, a, c])

                                    if kj == kb:
                                        t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, b], eris.oovv[ki, kk, ka, :, :, a, c].conj())
                                        t3d = t3d + einsum('j,ik->ijk', fov[kj, :, b],        t2[ki, kk, ka, :, :, a, c])

                                    if kk == kb:
                                        t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, b], eris.oovv[kj, ki, ka, :, :, a, c].conj())
                                        t3d = t3d + einsum('k,ji->ijk', fov[kk, :, b],        t2[kj, ki, ka, :, :, a, c])

                                    # Third term: - p(ac) + p(ac) p(ij) + p(ac) p(ik)
                                    if ki == kc:
                                        t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, c], eris.oovv[kj, kk, kb, :, :, b, a].conj())
                                        t3d = t3d - einsum('i,jk->ijk', fov[ki, :, c],        t2[kj, kk, kb, :, :, b, a])

                                    if kj == kc:
                                        t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, c], eris.oovv[ki, kk, kb, :, :, b, a].conj())
                                        t3d = t3d + einsum('j,ik->ijk', fov[kj, :, c],        t2[ki, kk, kb, :, :, b, a])

                                    if kk == kc:
                                        t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, c], eris.oovv[kj, ki, kb, :, :, b, a].conj())
                                        t3d = t3d + einsum('k,ji->ijk', fov[kk, :, c],        t2[kj, ki, kb, :, :, b, a])

                                    t3c_plus_d = t3c + t3d
                                    t3c_plus_d /= eijkabc

                                    energy_t += symm_abc * symm_ijk * (1./36) * einsum('ijk,ijk', t3c, t3c_plus_d.conj())

    energy_t = energy_t / nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn(mycc, 'Non-zero imaginary part of CCSD(T) energy was found %s',
                 energy_t.imag)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
    return energy_t.real

def check_antiperm_symmetry(array, idx1, idx2, tolerance=1e-8):
    '''
    Checks whether an array with k-point symmetry has antipermutational symmetry
    with respect to switching two indices idx1, idx2.  For 2-particle arrays,
    idx1 and idx2 must be in the range [0,3], while for 3-particle arrays they
    must be in the range [0,6].

    For a 3-particle array, such as the T3 amplitude
        t3[ki, kj, kk, ka, kb, i, j, a, b, c],
    setting `idx1 = 0` and `idx2 = 1` would switch the orbital indices i, j as well
    as the kpoint indices ki, kj.
    '''
    # Checking to make sure bounds of idx1 and idx2 are O.K.
    assert(idx1 >= 0 and idx2 >= 0)
    assert(idx1 != idx2)

    array_shape_len = len(array.shape)
    nparticles = (array_shape_len + 1) / 4
    assert(idx1 < ( 2 * nparticles - 1 ) and idx2 < ( 2 * nparticles - 1 ))

    if (nparticles > 3):
        raise NotImplementedError("Currently set up for only up to 3 particle "
                                  "arrays. Input array has %d particles.")

    kpt_idx1 = idx1
    kpt_idx2 = idx2

    # Start of the orbital index, located after k-point indices
    orb_idx1 = (2 * nparticles - 1) + idx1
    orb_idx2 = (2 * nparticles - 1) + idx2

    sign = (-1)**(abs(idx1 - idx2) + 1)
    out_array_indices = np.arange(array_shape_len)

    out_array_indices[kpt_idx1], out_array_indices[kpt_idx2] = \
            out_array_indices[kpt_idx2], out_array_indices[kpt_idx1]
    out_array_indices[orb_idx1], out_array_indices[orb_idx2] = \
            out_array_indices[orb_idx2], out_array_indices[orb_idx1]
    return (np.linalg.norm(array + array.transpose(out_array_indices)) <
            tolerance)

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
    cell.verbose = 5
    cell.mesh = [24, 24, 24]
    cell.build()

    kpts = cell.make_kpts([1, 1, 3])
    kpts -= kpts[0]+0.02
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KGCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()

    energy_t = kernel(mycc)

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
