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

    energy_t = 0.0

    def get_w(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
        km = kconserv[ki, ka, kj]
        kf = kconserv[ki, ka, kb]
        #ibaf
        #bifa
        #ret =       einsum('kjf,if->ijk', t2[kk, kj, kc, :, :, c, :], eris.ovvv[ki, ka, kb, :, a, b, :])
        ret =       einsum('kjf,if->ijk', t2[kk, kj, kc, :, :, c, :], eris.vovv[kb, ki, kf, b, :, :, a])
        #ret = ret - einsum('mk,jmi->ijk', t2[km, kk, kb, :, :, b, c], eris.ooov[kj, km, ki, :, :, :, a].conj())
        return ret

    def get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Pijkabc operating on Wijkabc intermediate'''
        ret =       get_w(ki, kj, kk, ka, kb, kc, a, b, c)
        ret = ret + get_w(kj, kk, ki, kb, kc, ka, b, c, a).transpose(1, 2, 0)
        ret = ret + get_w(kk, ki, kj, kc, ka, kb, c, a, b).transpose(2, 0, 1)
        ret = ret + get_w(ki, kk, kj, ka, kc, kb, a, c, b).transpose(0, 2, 1)
        ret = ret + get_w(kk, kj, ki, kc, kb, ka, c, b, a).transpose(2, 1, 0)
        ret = ret + get_w(kj, ki, kk, kb, ka, kc, b, a, c).transpose(1, 0, 2)
        return ret

    def get_rw(ki, kj, kk, ka, kb, kc, a, b, c):
        '''R operating on Wijkabc intermediate'''
        ret = ( 4. * get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c) +
                1. * get_permuted_w(kk, ki, kj, ka, kb, kc, a, b, c).transpose(2, 0, 1) +
                1. * get_permuted_w(kj, kk, ki, ka, kb, kc, a, b, c).transpose(1, 2, 0) -
                2. * get_permuted_w(kk, kj, ki, ka, kb, kc, a, b, c).transpose(2, 1, 0) -
                2. * get_permuted_w(ki, kk, kj, ka, kb, kc, a, b, c).transpose(0, 2, 1) -
                2. * get_permuted_w(kj, ki, kk, ka, kb, kc, a, b, c).transpose(1, 0, 2) )
        return ret

    def get_v(ki, kj, kk, ka, kb, kc, a, b, c):
        '''Vijkabc intermediate as described in Scuseria paper'''
        km = kconserv[ki, ka, kj]
        ret = einsum('kjf,if->ijk', t2[kk, kj, kc, :, :, c, :], eris.ovvv[ki, ka, kb, :, a, b, :])
        ret = ret - einsum('mk,jmi->ijk', t2[km, kk, kb, :, :, b, c], eris.ooov[kj, km, ki, :, :, :, a].conj())
        return 0.0*ret

    symm_on = True
    t3c_full = np.zeros((nkpts,nkpts,nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir,nvir,nvir), dtype=dtype)
    for ki in range(nkpts):
        #for kj in range(nkpts):
        #    for kk in range(nkpts):
        for kj in range(nkpts):
            for kk in range(nkpts):

                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = lib.direct_sum('i,j,k->ijk', mo_energy_occ[ki], mo_energy_occ[kj], mo_energy_occ[kk])

                symm_ijk = 1.

                for ka in range(nkpts):
                    for kb in range(nkpts):

                        # Find momentum conservation condition for triples
                        # amplitude t3ijkabc
                        kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                        #if kc not in range(kb+1):
                        #    continue

                        symm_abc = 1.
                        #if ka == kb and kb == kc:
                        #    symm_abc = 2.
                        #elif ka == kb or kb == kc:
                        #    symm_abc = 3.

                        e_cont = 0.0
                        for a in range(nvir):
                            for b in range(nvir):
                                for c in range(nvir):

                                    # Form energy denominator
                                    eijkabc = (eijk - mo_energy_vir[ka][a] - mo_energy_vir[kb][b] - mo_energy_vir[kc][c])

                                    # Form connected triple excitation amplitude
                                    t3c = np.zeros((nocc,nocc,nocc), dtype=dtype)

                                    pwijk = get_permuted_w(ki, kj, kk, ka, kb, kc, a, b, c)
                                    rwijk =         get_rw(ki, kj, kk, ka, kb, kc, a, b, c) / eijkabc
                                    t3c_full[ki, kj, kk, ka, kb, :, :, :, a, b, c] = pwijk

                                    # Form disconnected triple excitation amplitude contribution
                                    t3d = np.zeros((nocc,nocc,nocc), dtype=dtype)

                                    ## First term: 1 - p(ij) - p(ik)
                                    #if ki == ka:
                                    #    t3d = t3d + einsum('i,jk->ijk',  t1[ki, :, a], eris.oovv[kj, kk, kb, :, :, b, c].conj())
                                    #    t3d = t3d + einsum('i,jk->ijk', fov[ki, :, a],        t2[kj, kk, kb, :, :, b, c])

                                    #if kj == ka:
                                    #    t3d = t3d - einsum('j,ik->ijk',  t1[kj, :, a], eris.oovv[ki, kk, kb, :, :, b, c].conj())
                                    #    t3d = t3d - einsum('j,ik->ijk', fov[kj, :, a],        t2[ki, kk, kb, :, :, b, c])

                                    #if kk == ka:
                                    #    t3d = t3d - einsum('k,ji->ijk',  t1[kk, :, a], eris.oovv[kj, ki, kb, :, :, b, c].conj())
                                    #    t3d = t3d - einsum('k,ji->ijk', fov[kk, :, a],        t2[kj, ki, kb, :, :, b, c])

                                    ## Second term: - p(ab) + p(ab) p(ij) + p(ab) p(ik)
                                    #if ki == kb:
                                    #    t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, b], eris.oovv[kj, kk, ka, :, :, a, c].conj())
                                    #    t3d = t3d - einsum('i,jk->ijk', fov[ki, :, b],        t2[kj, kk, ka, :, :, a, c])

                                    #if kj == kb:
                                    #    t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, b], eris.oovv[ki, kk, ka, :, :, a, c].conj())
                                    #    t3d = t3d + einsum('j,ik->ijk', fov[kj, :, b],        t2[ki, kk, ka, :, :, a, c])

                                    #if kk == kb:
                                    #    t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, b], eris.oovv[kj, ki, ka, :, :, a, c].conj())
                                    #    t3d = t3d + einsum('k,ji->ijk', fov[kk, :, b],        t2[kj, ki, ka, :, :, a, c])

                                    ## Third term: - p(ac) + p(ac) p(ij) + p(ac) p(ik)
                                    #if ki == kc:
                                    #    t3d = t3d - einsum('i,jk->ijk',  t1[ki, :, c], eris.oovv[kj, kk, kb, :, :, b, a].conj())
                                    #    t3d = t3d - einsum('i,jk->ijk', fov[ki, :, c],        t2[kj, kk, kb, :, :, b, a])

                                    #if kj == kc:
                                    #    t3d = t3d + einsum('j,ik->ijk',  t1[kj, :, c], eris.oovv[ki, kk, kb, :, :, b, a].conj())
                                    #    t3d = t3d + einsum('j,ik->ijk', fov[kj, :, c],        t2[ki, kk, kb, :, :, b, a])

                                    #if kk == kc:
                                    #    t3d = t3d + einsum('k,ji->ijk',  t1[kk, :, c], eris.oovv[kj, ki, kb, :, :, b, a].conj())
                                    #    t3d = t3d + einsum('k,ji->ijk', fov[kk, :, c],        t2[kj, ki, kb, :, :, b, a])

                                    t3c_plus_d = t3c + t3d
                                    t3c_plus_d /= eijkabc

                                    energy_t += symm_abc * symm_ijk * einsum('ijk,ijk', pwijk, rwijk.conj())
                                    e_cont += einsum('ijk,ijk', pwijk, rwijk.conj())
                        print [ki, kj, kk, ka, kb, kc], e_cont

    energy_t = 2. * energy_t / nkpts
    print -0.0153250712056708

    #                                                 ki,kj,kk,ka,kb,i,j,k,a,b,c
    print np.linalg.norm(t3c_full - t3c_full.transpose(1, 0, 2, 4, 3,6,5,7,9,8,10))

    if abs(energy_t.imag) > 1e-4:
        log.warn(mycc, 'Non-zero imaginary part of CCSD(T) energy was found %s',
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

    kpts = cell.make_kpts([1, 1, 2])
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    ehf = kmf.kernel()

    mycc = cc.KRCCSD(kmf)
    ecc, t1, t2 = mycc.kernel()

    energy_t = kernel(mycc)
