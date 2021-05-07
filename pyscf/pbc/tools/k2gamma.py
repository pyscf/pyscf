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
import scipy.optimize
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

def rotate_mo_to_real(cell, mo_energy, mo_coeff, ovlp, degen_tol=1e-3, rotate_degen=True, method="max-element"):
    """Applies a phase factor to each MO, minimizing the norm of the imaginary part.

    Typically, this should reduce the imaginary part of a non-degenerate, Gamma point orbital to zero.
    However, for degenerate subspaces, addition treatment is generally required.
    """

    # Output orbitals
    mo_coeff_rot = mo_coeff.copy()

    for mo, mo_e in enumerate(mo_energy):
        # Check if MO is degnerate
        if mo == 0:
            degen = (abs(mo_e - mo_energy[mo+1]) < degen_tol)
        elif mo == (len(mo_energy)-1):
            degen = (abs(mo_e - mo_energy[mo-1]) < degen_tol)
        else:
            degen = (abs(mo_e - mo_energy[mo-1]) < degen_tol) or (abs(mo_e - mo_energy[mo+1]) < degen_tol)

        if degen and not rotate_degen:
            continue

        mo_c = mo_coeff[:,mo]

        rotate = False
        norm0 = np.linalg.norm(mo_c.imag)

        if method == "max-element":
            maxidx = np.argmax(abs(mo_c.imag))
            maxval = mo_c[maxidx]
            ratio0 = abs(maxval.imag/maxval.real)

            if ratio0 > 1e-8 and abs(maxval.imag) > 1e-8:
                # Determine -phase of maxval
                phase = -np.angle(maxval)
                # Rotate to real axis
                mo_c1 = mo_c*np.exp(1j*phase)
                maxval1 = mo_c1[maxidx]
                ratio1 = abs(maxval1.imag/maxval1.real)
                norm1 = np.linalg.norm(mo_c1.imag)
            else:
                norm1 = norm0

        # Optimization
        elif method == "optimize":

            # Function to optimize
            def funcval(phase, mo):
                mo1 = mo*np.exp(1j*phase)
                fval = np.linalg.norm(mo1.imag) #/ np.linalg.norm(mo1.real)
                return fval

            phase0 = 0.0
            res = scipy.optimize.minimize(funcval, x0=phase0, args=(mo_c,))
            phase = res.x
            norm1 = funcval(phase, mo_c)
        else:
            raise ValueError()

        # Only perform rotation if imaginary norm is decreased
        if (norm1 < norm0):
            mo_coeff_rot[:,mo] = mo_c1
            logger.debug(cell, "MO %4d at E=%+12.8e: degen= %5r |Im C|= %.2e -> %.2e (phase= %+.8f)", mo, mo_e, degen, norm0, norm1, phase)
        else:
            logger.debug(cell, "MO %4d at E=%+12.8e: degen= %5r |Im C|= %.2e", mo, mo_e, degen, norm0)

    return mo_coeff_rot

def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None, degen_method="fock", degen_tol=1e-3):
    logger.debug(cell, "starting mo_k2gamma")
    scell, phase = get_phase(cell, kpts, kmesh)

    E_g = np.hstack(mo_energy)

    # The number of MOs may be k-point dependent (eg. due to remove_linear_dep)
    Nmo_k = np.asarray([ck.shape[-1] for ck in mo_coeff])
    logger.debug(cell, "Nmo(k)= %r", Nmo_k)
    if np.all(Nmo_k == Nmo_k[0]):
        C_k = np.asarray(mo_coeff)
        Nk, Nao, Nmo = C_k.shape
        Nmo_k = None
    else:
        Nk = len(mo_coeff)
        Nao_k = np.asarray([ck.shape[0] for ck in mo_coeff])
        assert np.all(Nao_k == Nao_k[0]), ("ERROR: Nao_k= %r" % Nao_k)
        Nao = Nao_k[0]
    NR = phase.shape[0]

    # Transform AO indices
    if Nmo_k is None:
        C_gamma = np.einsum('Rk,kum->Rukm', phase, C_k)
        C_gamma = C_gamma.reshape(Nao*NR, Nk*Nmo)
    else:
        C_gamma = []
        for k in range(Nk):
            C_k = np.einsum("R,um->Rum", phase[:,k], mo_coeff[k])
            C_k = C_k.reshape(NR*Nao, Nmo_k[k])
            C_gamma.append(C_k)
        C_gamma = np.hstack(C_gamma)
        assert C_gamma.shape == (NR*Nao, sum(Nmo_k))


    E_sort_idx = np.argsort(E_g)
    E_g = E_g[E_sort_idx]
    C_gamma = C_gamma[:,E_sort_idx]
    #ovlp = scell.pbc_intor('int1e_ovlp', hermi=1, pbcopt=lib.c_null_ptr())
    # Determine overlap by unfolding for better error cancelation?
    s_k = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts, pbcopt=lib.c_null_ptr())
    ovlp = to_supercell_ao_integrals(cell, kpts, s_k)
    #logger.debug(cell, "Difference between overlap matrixes: norm= %.2e max= %.2e", np.linalg.norm(s2-s), abs(s2-s).max())

    #assert(abs(reduce(np.dot, (C_gamma.conj().T, s, C_gamma))
    #           - np.eye(Nmo*Nk)).max() < 1e-5)
    ortherr = abs(np.linalg.multi_dot((C_gamma.conj().T, ovlp, C_gamma)) - np.eye(C_gamma.shape[-1])).max()
    logger.debug(cell, "Orthogonality error= %.2e" % ortherr)
    if ortherr > 1e-4:
        logger.error(cell, "ERROR: Unfolded MOs are not orthogonal!")

    #ortherr = abs(np.linalg.multi_dot((C_gamma.conj().T, s2, C_gamma)) - np.eye(Nmo*Nk)).max()
    #logger.debug(cell, "Orthogonality error= %.2e" % ortherr)
    #if ortherr > 1e-4:
    #    logger.error(cell, "ERROR: Unfolded MOs are not orthogonal!")

    #assert ortherr < 1e-5

    # For degenerated MOs, the transformed orbitals in super cell may not be
    # real. Construct a sub Fock matrix in super-cell to find a proper
    # transformation that makes the transformed MOs real.
    E_k_degen = abs(E_g[1:] - E_g[:-1]) < degen_tol
    degen_mask = np.append(False, E_k_degen) | np.append(E_k_degen, False)

    # Try to remove imaginary parts by multiplication of simple phase factors
    C_gamma = rotate_mo_to_real(cell, E_g, C_gamma, ovlp=ovlp, degen_tol=degen_tol)

    if np.any(~degen_mask):
        cimag_nondegen = abs(C_gamma[:,~degen_mask].imag).max()
        logger.debug(cell, "Imaginary part in non-degenerate MO coefficients= %5.2e" % cimag_nondegen)
    else:
        cimag_nondegen = 0.0
        logger.debug(cell, "No non-degenerate MOs found")

    # Only fock can deal with significant imaginary parts outside of imaginary subspaces:
    if degen_method == "dm" and cimag_nondegen >= 1e-4:
        degen_method = "fock"
        print("Significant imaginary parts - changing degen_method to %s" % degen_method)

    # Eigenvalue solver with linear-dependency treatment
    #eigh = scell.eigh_factory()

    t0 = timer()
    if degen_method == "fock":
        C_gamma_out = C_gamma.copy()
        if np.any(E_k_degen):
            # Diagonalize Fock matrix within degenerate subspace only
            if cimag_nondegen < 1e-4:

                # Shift all MOs above the eig=0 subspace (so they can be identified later)
                shift = 1.0 - min(E_g[degen_mask])
                #f = np.dot(C_gamma[:,degen_mask] * (E_g[degen_mask] + shift), C_gamma[:,degen_mask].conj().T)
                #f = np.linalg.multi_dot((ovlp, f, ovlp))
                cs = np.dot(C_gamma[:,degen_mask].conj().T, ovlp)
                f = np.dot(cs.T.conj() * (E_g[degen_mask] + shift), cs)
                fimag = abs(f.imag).max()
                logger.debug(cell, "max|Im(F)|= %.2e", fimag)

                #fimag = abs(np.dot(f, ovlp).imag).max()
                #logger.debug(cell, "max|Im(F.S)|= %.2e", fimag)

                #assert(abs(f.imag).max() < 1e-3), "max|Im(F)| = %.2e" % abs(f.imag).max()

                #e, na_orb = scipy.linalg.eigh(f.real, ovlp, type=2)
                e, na_orb = scipy.linalg.eigh(f.real, ovlp)
                #e, na_orb = eigh(f.real, ovlp)

                # Extract MOs from rank-deficient fock matrix
                mask = (e > 0.5)
                #logger.debug(cell, "Eigenvalues > 1e-14: %r", e[abs(e) > 1e-14])
                assert np.count_nonzero(mask) == np.count_nonzero(degen_mask)
                logger.debug(cell, "Max error of MO energies= %.2e", abs(E_g[degen_mask]-(e[mask]-shift)).max())
                #E_g[degen_mask] = e+shift
                C_gamma_out = C_gamma_out.real
                C_gamma_out[:,degen_mask] = na_orb[:,mask]
            # Diagonalize whole Fock matrix
            else:
                #f = np.dot(C_gamma * E_g, C_gamma.conj().T)
                #f = np.linalg.multi_dot((ovlp, f, ovlp))

                # f may be rank deficient, due to linear-dependency treatment
                shift = 1.0 - min(E_g)
                cs = np.dot(C_gamma.conj().T, ovlp)
                f = np.dot(cs.T.conj() * (E_g + shift), cs)

                fimag = abs(f.imag).max()
                logger.debug(cell, "max|Im(F)|= %.2e", fimag)

                #fimag = abs(np.dot(f, ovlp).imag).max()
                #logger.debug(cell, "max|Im(F.S)|= %.2e", fimag)
                #assert(abs(f.imag).max() < 1e-3), "max|Im(F)|= %.2e" % abs(f.imag).max()

                #e, C_gamma = scipy.linalg.eigh(f.real, ovlp, type=2)
                # Replace E_g - it is possible that additional MO are being removed due to linear dependency treatment
                #E_g, C_gamma_out = eigh(f.real, ovlp)
                e, C_gamma_out = scipy.linalg.eigh(f.real, ovlp)
                #E_g = e
                mask = (e > 0.5)
                assert np.count_nonzero(mask) == len(E_g)
                logger.debug(cell, "Max error of MO energies= %.2e", abs(E_g-(e[mask]-shift)).max())
                C_gamma_out = C_gamma_out[:,mask].copy()


    # Degeneracy treatment based on DM
    elif degen_method == "dm":
        raise NotImplementedError()
        assert cimag_nondegen < 1e-4
        print("Initial imaginary part in Gamma-point MO coefficients= %5.2e" % abs(C_gamma.imag).max())
        idx0 = 0
        # Looping over stop-index, append state with energy 1e9 to guarantee closing of last subspace
        for idx, e1 in enumerate(np.hstack((E_g[1:], 1e9)), 1):

            # Close off previous subspace
            if ((e1-E_g[idx-1]) > degen_tol):

                dsize = (idx-idx0)
                dspace = np.s_[idx0:idx]
                emean, emin, emax = E_g[dspace].mean(), E_g[dspace].min(), E_g[dspace].max()
                cimag = abs(C_gamma[:,dspace].imag).max()

                idx0 = idx

                # Previous subspace is only of size 1
                #if dsize == 1:
                #    logger.debug(cell, "Nondegenerate eigenvalue at E= %12.8g: max|Im(C)|= %7.2e" % (emean, cimag))
                #    assert cimag < 1e-4
                #    continue

                # Subspace mo_coeff already real
                if cimag < 1e-8:
                    logger.debug(cell, "Subspace of size= %2d at E= %12.8g (min= %12.8g max=%12.8g spread=%7.2g): max|Im(C)|= %7.2e" %
                        (dsize, emean, emin, emax, emax-emin, cimag))
                    continue

                dm = np.dot(C_gamma[:,dspace], C_gamma[:,dspace].conj().T)
                dimag = abs(dm.imag).max()
                dimag2 = abs(np.dot(dm, ovlp).imag).max()
                # comparison to Fock matrix
                #shift = 1.0 - min(E_g[dspace])
                #f = np.dot(C_gamma[:,dspace]*(E_g[dspace]+shift), C_gamma[:,dspace].conj().T)
                #fimag = abs(f.imag.max())

                #print("Degenerate subspace of size= %2d at E= %12.8g (min= %12.8g max=%12.8g spread=%7.2g): max|Im(D)|= %7.2e max|Im(F)|= %7.2e" %
                #        (dsize, emean, emin, emax, emax-emin, dimag, fimag))
                logger.debug(cell, "Subspace of size= %2d at E= %12.8g (min= %12.8g max=%12.8g spread=%7.2g): max|Im(C)|= %7.2e max|Im(D)|= %7.2e max|Im(D.S)|= %7.2e" %
                        (dsize, emean, emin, emax, emax-emin, cimag, dimag, dimag2))
                if dimag > 1e-4: logger.warn(cell, "Large imaginary component in DM: %7.2e !", dimag)
                #assert (dimag < 1e-3), "max|Im(D)|= %.2e" % dimag
                e, v = scipy.linalg.eigh(dm.real, ovlp, type=2)
                assert (np.count_nonzero(e > 1e-3) == dsize)
                C_gamma[:,dspace] = v[:,e>1e-3]

        cimag = abs(C_gamma.imag).max()
        print("Final max|Im(C)| in Gamma-point MOs= %7.2e" % cimag)
        assert (cimag < 1e-3)
        C_gamma = C_gamma.real
    else:
        print("Unknown value for degen_mode= %s", degen_mode)

    logger.debug(cell, "Time for degeneracy treatment= %.2f s", (timer()-t0))

    assert np.allclose(C_gamma_out.imag, 0)
    C_gamma_out = C_gamma_out.real

    #s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    #s_k = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts, pbcopt=lib.c_null_ptr())
    # overlap between k-point unitcell and gamma-point supercell
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(Nk,Nao,NR*Nao)
    # The unitary transformation from k-adapted orbitals to gamma-point orbitals
    if Nmo_k is None:
        mo_phase = lib.einsum('kum,kuv,vi->kmi', C_k.conj(), s_k_g, C_gamma_out)
    else:
        mo_phase = []
        for k in range(Nk):
            mo_phase_k = lib.einsum('um,uv,vi->mi', mo_coeff[k].conj(), s_k_g[k], C_gamma_out)
            mo_phase.append(mo_phase_k)

    return scell, E_g, C_gamma_out, mo_phase

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

