#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Convert the k-sampled MO/integrals to corresponding Gamma-point supercell
MO/integrals.

Zhihao Cui <zcui@caltech.edu>
Qiming Sun <osirpt.sun@gmail.com>

See also the original implementation at
https://github.com/zhcui/local-orbital-and-cdft/blob/master/k2gamma.py
'''

from timeit import default_timer as timer
from functools import reduce
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc import scf

def kpts_to_kmesh(cell, kpts):
    '''Guess kmesh'''
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = (len(np.unique(scaled_k[:,0])),
             len(np.unique(scaled_k[:,1])),
             len(np.unique(scaled_k[:,2])))
    logger.debug(cell, "Guessed kmesh= %r", kmesh)
    return kmesh

def translation_vectors_for_kmesh(cell, kmesh):
    '''
    Translation vectors to construct super-cell of which the gamma point is
    identical to the k-point mesh of primitive cell
    '''
    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_vec_abs = np.einsum('nu, uv -> nv', R_vec_rel, latt_vec)
    return R_vec_abs

def get_phase(cell, kpts, kmesh=None):
    '''
    The unitary transformation that transforms the supercell basis k-mesh
    adapted basis.
    '''
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    R_vec_abs = translation_vectors_for_kmesh(cell, kmesh)

    NR = len(R_vec_abs)
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    phase /= np.sqrt(NR)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = tools.super_cell(cell, kmesh)
    return scell, phase

def double_translation_indices(kmesh):
    '''Indices to utilize the translation symmetry in the 2D matrix.

    D[M,N] = D[N-M]

    The return index maps the 2D subscripts to 1D subscripts.

    D2 = D1[double_translation_indices()]

    D1 holds all the symmetry unique elements in D2
    '''

    tx = translation_map(kmesh[0])
    ty = translation_map(kmesh[1])
    tz = translation_map(kmesh[2])
    idx = np.ravel_multi_index([tx[:,None,None,:,None,None],
                                ty[None,:,None,None,:,None],
                                tz[None,None,:,None,None,:]], kmesh)
    nk = np.prod(kmesh)
    return idx.reshape(nk, nk)

def translation_map(nk):
    ''' Generate
    [0    1 .. n  ]
    [n    0 .. n-1]
    [n-1  n .. n-2]
    [...  ...  ...]
    [1    2 .. 0  ]
    '''
    idx = np.repeat(np.arange(nk)[None,:], nk-1, axis=0)
    strides = idx.strides
    t_map = np.ndarray((nk, nk), strides=(strides[0]-strides[1], strides[1]),
                       dtype=int, buffer=np.append(idx.ravel(), 0))
    return t_map


def rotate_mo_to_real(cell, mo_energy, mo_coeff, degen_tol=1e-3, rotate_degen=True):
    """Applies a phase factor to each MO, minimizing the maximum imaginary element.

    Typically, this should reduce the imaginary part of a non-degenerate, Gamma point orbital to zero.
    However, for degenerate subspaces, addition treatment is required.
    """

    # Output orbitals
    mo_coeff_out = mo_coeff.copy()

    for mo_idx, mo_e in enumerate(mo_energy):
        # Check if MO is degnerate
        if mo_idx == 0:
            degen = (abs(mo_e - mo_energy[mo_idx+1]) < degen_tol)
        elif mo_idx == (len(mo_energy)-1):
            degen = (abs(mo_e - mo_energy[mo_idx-1]) < degen_tol)
        else:
            degen = (abs(mo_e - mo_energy[mo_idx-1]) < degen_tol) or (abs(mo_e - mo_energy[mo_idx+1]) < degen_tol)
        if degen and not rotate_degen:
            continue

        mo_c = mo_coeff[:,mo_idx]
        norm_in = np.linalg.norm(mo_c.imag)
        # Find phase which makes the largest element of |C| real
        maxidx = np.argmax(abs(mo_c.imag))
        maxval = mo_c[maxidx]
        # Determine -phase of maxval and rotate to real axis
        phase = -np.angle(maxval)
        mo_c2 = mo_c*np.exp(1j*phase)

        # Only perform rotation if imaginary norm is decreased
        norm_out = np.linalg.norm(mo_c2.imag)
        if (norm_out < norm_in):
            mo_coeff_out[:,mo_idx] = mo_c2
        else:
            norm_out = norm_in
        if norm_out > 1e-8 and not degen:
            logger.warn(cell, "Non-degenerate MO %4d at E= %+12.8f Ha: ||Im(C)||= %6.2e !", mo_idx, mo_e, norm_out)

    return mo_coeff_out

def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None, degen_tol=1e-3, imag_tol=1e-9):
    logger.debug(cell, "Starting mo_k2gamma")
    scell, phase = get_phase(cell, kpts, kmesh)

    # Supercell Gamma-point MO energies
    e_gamma = np.hstack(mo_energy)
    # The number of MOs may be k-point dependent (eg. due to linear dependency)
    nmo_k = np.asarray([ck.shape[-1] for ck in mo_coeff])
    nk = len(mo_coeff)
    nao = mo_coeff[0].shape[0]
    nr = phase.shape[0]
    # Transform mo_coeff from k-points to supercell Gamma-point:
    c_gamma = []
    for k in range(nk):
        c_k = np.einsum('R,um->Rum', phase[:,k], mo_coeff[k])
        c_k = c_k.reshape(nr*nao, nmo_k[k])
        c_gamma.append(c_k)
    c_gamma = np.hstack(c_gamma)
    assert c_gamma.shape == (nr*nao, sum(nmo_k))
    # Sort according to MO energy
    sort = np.argsort(e_gamma)
    e_gamma, c_gamma = e_gamma[sort], c_gamma[:,sort]
    # Determine overlap by unfolding for better accuracy
    s_k = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts, pbcopt=lib.c_null_ptr())
    s_gamma = to_supercell_ao_integrals(cell, kpts, s_k)
    # Orthogonality error of unfolded MOs
    err_orth = abs(np.linalg.multi_dot((c_gamma.conj().T, s_gamma, c_gamma)) - np.eye(c_gamma.shape[-1])).max()
    if err_orth > 1e-4:
        logger.error(cell, "Orthogonality error of MOs= %.2e !!!", err_orth)
    else:
        logger.debug(cell, "Orthogonality error of MOs= %.2e", err_orth)

    # Make Gamma point MOs real:

    # Try to remove imaginary parts by multiplication of simple phase factors
    c_gamma = rotate_mo_to_real(cell, e_gamma, c_gamma, degen_tol=degen_tol)

    # For degenerated MOs, the transformed orbitals in super cell may not be
    # real. Construct a sub Fock matrix in super-cell to find a proper
    # transformation that makes the transformed MOs real.
    #e_k_degen = abs(e_gamma[1:] - e_gamma[:-1]) < degen_tol
    #degen_mask = np.append(False, e_k_degen) | np.append(e_k_degen, False)

    # Get eigenvalue solver with linear-dependency treatment
    eigh = cell.eigh_factory(lindep_threshold=1e-13, fallback_mode=True)

    c_gamma_out = c_gamma.copy()
    mo_mask = (np.linalg.norm(c_gamma.imag, axis=0) > imag_tol)
    logger.debug(cell, "Number of MOs with imaginary coefficients: %d out of %d", np.count_nonzero(mo_mask), len(mo_mask))
    if np.any(mo_mask):
        #mo_mask = np.s_[:]
        #if np.any(~degen_mask):
        #    err_imag = abs(c_gamma[:,~degen_mask].imag).max()
        #    logger.debug(cell, "Imaginary part in non-degenerate MO coefficients= %.2e", err_imag)
        #    # Diagonalize Fock matrix spanned by degenerate MOs only
        #    if err_imag < 1e-8:
        #        mo_mask = degen_mask

        # F
        #mo_mask = (np.linalg.norm(c_gamma.imag, axis=0) > imag_tol)

        # Shift all MOs above the eig=0 subspace, so they can be extracted below
        shift = 1.0 - min(e_gamma[mo_mask])
        cs = np.dot(c_gamma[:,mo_mask].conj().T, s_gamma)
        f_gamma = np.dot(cs.T.conj() * (e_gamma[mo_mask] + shift), cs)
        logger.debug(cell, "Imaginary parts of Fock matrix: ||Im(F)||= %.2e  max|Im(F)|= %.2e", np.linalg.norm(f_gamma.imag), abs(f_gamma.imag).max())

        e, v = eigh(f_gamma.real, s_gamma)

        # Extract MOs from rank-deficient Fock matrix
        mask = (e > 0.5)
        assert np.count_nonzero(mask) == len(e_gamma[mo_mask])
        e, v = e[mask], v[:,mask]
        e_delta = e_gamma[mo_mask] - (e-shift)
        if abs(e_delta).max() > 1e-4:
            logger.error(cell, "Error of MO energies: ||dE||= %.2e  max|dE|= %.2e !!!", np.linalg.norm(e_delta), abs(e_delta).max())
        else:
            logger.debug(cell, "Error of MO energies: ||dE||= %.2e  max|dE|= %.2e", np.linalg.norm(e_delta), abs(e_delta).max())
        c_gamma_out[:,mo_mask] = v

    err_imag = abs(c_gamma_out.imag).max()
    if err_imag > 1e-4:
        logger.error(cell, "Imaginary part in gamma-point MOs: max|Im(C)|= %7.2e !!!", err_imag)
    else:
        logger.debug(cell, "Imaginary part in gamma-point MOs: max|Im(C)|= %7.2e", err_imag)
    c_gamma_out = c_gamma_out.real

    # Determine mo_phase, i.e. the unitary transformation from k-adapted orbitals to gamma-point orbitals
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(nk,nao,nr*nao)
    mo_phase = []
    for k in range(nk):
        mo_phase_k = lib.einsum('um,uv,vi->mi', mo_coeff[k].conj(), s_k_g[k], c_gamma_out)
        mo_phase.append(mo_phase_k)

    return scell, e_gamma, c_gamma_out, mo_phase

def k2gamma(kmf, kmesh=None):
    r'''
    convert the k-sampled mean-field object to the corresponding supercell
    gamma-point mean-field object.

    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \frac{1}{\sqrt{N_{\UC}}}
         \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}
    '''
    def transform(mo_energy, mo_coeff, mo_occ):
        scell, E_g, C_gamma = mo_k2gamma(kmf.cell, mo_energy, mo_coeff,
                                         kmf.kpts, kmesh)[:3]
        E_sort_idx = np.argsort(np.hstack(mo_energy))
        mo_occ = np.hstack(mo_occ)[E_sort_idx]
        return scell, E_g, C_gamma, mo_occ

    if isinstance(kmf, scf.khf.KRHF):
        scell, E_g, C_gamma, mo_occ = transform(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)
        mf = scf.RHF(scell)
    elif isinstance(kmf, scf.kuhf.KUHF):
        scell, Ea, Ca, occ_a = transform(kmf.mo_energy[0], kmf.mo_coeff[0], kmf.mo_occ[0])
        scell, Eb, Cb, occ_b = transform(kmf.mo_energy[1], kmf.mo_coeff[1], kmf.mo_occ[1])
        mf = scf.UHF(scell)
        E_g = [Ea, Eb]
        C_gamma = [Ca, Cb]
        mo_occ = [occ_a, occ_b]
    else:
        raise NotImplementedError('SCF object %s not supported' % kmf)

    mf.mo_coeff = C_gamma
    mf.mo_energy = E_g
    mf.mo_occ = mo_occ
    mf.converged = kmf.converged
    # Scale energy by number of primitive cells within supercell
    mf.e_tot = len(kmf.kpts)*kmf.e_tot

    # Use unfolded overlap matrix for better error cancellation
    #s_k = kmf.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kmf.kpts, pbcopt=lib.c_null_ptr())
    s_k = kmf.get_ovlp()
    ovlp = to_supercell_ao_integrals(kmf.cell, kmf.kpts, s_k)
    assert np.allclose(ovlp, ovlp.T)
    ovlp = (ovlp + ovlp.T) / 2
    mf.get_ovlp = lambda *args : ovlp

    return mf


def to_supercell_ao_integrals(cell, kpts, ao_ints):
    '''Transform from the unitcell k-point AO integrals to the supercell
    gamma-point AO integrals.
    '''
    scell, phase = get_phase(cell, kpts)
    NR, Nk = phase.shape
    nao = cell.nao
    scell_ints = np.einsum('Rk,kij,Sk->RiSj', phase, ao_ints, phase.conj())
    assert abs(scell_ints.imag).max() < 1e-5
    return scell_ints.reshape(NR*nao,NR*nao).real


def to_supercell_mo_integrals(kmf, mo_ints):
    '''Transform from the unitcell k-point MO integrals to the supercell
    gamma-point MO integrals.
    '''
    cell = kmf.cell
    kpts = kmf.kpts

    mo_k = np.array(kmf.mo_coeff)
    Nk, nao, nmo = mo_k.shape
    e_k = np.array(kmf.mo_energy)
    scell, E_g, C_gamma, mo_phase = mo_k2gamma(cell, e_k, mo_k, kpts)

    scell_ints = lib.einsum('xui,xuv,xvj->ij', mo_phase.conj(), mo_ints, mo_phase)
    assert(abs(scell_ints.imag).max() < 1e-7)
    return scell_ints.real


if __name__ == '__main__':
    from pyscf.pbc import gto, dft
    cell = gto.Cell()
    cell.atom = '''
    H 0.  0.  0.
    H 0.5 0.3 0.4
    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 4.
    cell.unit='B'
    cell.build()

    kmesh = [2, 2, 1]
    kpts = cell.make_kpts(kmesh)

    print("Transform k-point integrals to supercell integral")
    scell, phase = get_phase(cell, kpts)
    NR, Nk = phase.shape
    nao = cell.nao
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s = scell.pbc_intor('int1e_ovlp')
    s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
    print(abs(s-s1.reshape(s.shape)).max())

    s = scell.pbc_intor('int1e_ovlp').reshape(NR,nao,NR,nao)
    s1 = np.einsum('Rk,RiSj,Sk->kij', phase.conj(), s, phase)
    print(abs(s1-s_k).max())

    kmf = dft.KRKS(cell, kpts)
    ekpt = kmf.run()

    mf = k2gamma(kmf, kmesh)
    c_g_ao = mf.mo_coeff

    # The following is to check whether the MO is correctly coverted:

    print("Supercell gamma MO in AO basis from conversion:")
    scell = tools.super_cell(cell, kmesh)
    mf_sc = dft.RKS(scell)

    s = mf_sc.get_ovlp()
    mf_sc.run()
    sc_mo = mf_sc.mo_coeff

    nocc = scell.nelectron // 2
    print("Supercell gamma MO from direct calculation:")
    print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
    print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])

