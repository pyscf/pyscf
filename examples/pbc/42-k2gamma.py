#!/usr/bin/env python

'''
Convert the k-sampled MO to corresponding Gamma-point supercell MO.
Zhihao Cui zcui@caltech.edu

https://github.com/zhcui/local-orbital-and-cdft/blob/master/k2gamma.py
'''

import numpy as np
from functools import reduce
import scipy.linalg as la
from pyscf import lib
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc import scf


def get_phase(cell, kpts, kmesh):
    '''
    phase of each k-points when transforming unitcell to supercell
    '''

    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_vec_abs = np.einsum('nu, uv -> nv', R_vec_rel, latt_vec)

    NR = len(R_vec_abs)
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    phase /= np.sqrt(NR)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = pbc_tools.super_cell(cell, kmesh)
    return scell, phase

def k2gamma(kmf, kmesh):
    r'''
    convert the k-sampled mean-field object to the corresponding supercell
    gamma-point mean-field object.

    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \frac{1}{\sqrt{N_{\UC}}}
         \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}
    '''

    scell, phase = get_phase(kmf.cell, kmf.kpts, kmesh)

    E_k = np.asarray(kmf.mo_energy).ravel()
    C_k = np.asarray(kmf.mo_coeff)
    Nk, Nao, Nmo = C_k.shape
    NR = phase.shape[0]

    C_gamma = np.einsum('Rk, kum -> Rukm', phase, C_k)
    C_gamma = C_gamma.reshape(Nao*NR, Nk*Nmo)

    E_k_sort_idx = np.argsort(E_k)
    E_k = E_k[E_k_sort_idx]
    C_gamma = C_gamma[:,E_k_sort_idx]

    s = scell.pbc_intor('int1e_ovlp')
    assert(abs(reduce(np.dot, (C_gamma.conj().T, s, C_gamma))
               - np.eye(Nmo*Nk)).max() < 1e-7)
    E_k_degen = abs(E_k[1:] - E_k[:-1]).max() < 1e-5
    if np.any(E_k_degen):
        degen_mask = np.append(False, E_k_degen) | np.append(E_k_degen, False)
        assert(abs(C_gamma[:,~degen_mask].imag).max() < 1e-5)

        shift = min(E_k[degen_mask]) - .1
        f = np.dot(C_gamma[:,degen_mask] * (E_k[degen_mask] - shift),
                   C_gamma[:,degen_mask].conj().T)
        assert(abs(f.imag).max() < 1e-5)

        e, na_orb = la.eigh(f.real, s, type=2)
        C_gamma[:,degen_mask] = na_orb[:, e>0]

    C_gamma = C_gamma.real
    assert(abs(reduce(np.dot, (C_gamma.conj().T, s, C_gamma))
               - np.eye(Nmo*Nk)).max() < 1e-7)

    mf = scf.RHF(scell)
    mf.mo_coeff = C_gamma
    mf.mo_energy = E_k
    mf.mo_occ = np.hstack(kmf.mo_occ)[E_k_sort_idx]
    return mf


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
    scell, phase = get_phase(cell, kpts, kmesh)
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s = scell.pbc_intor('int1e_ovlp')
    s1 = np.einsum('Rk,kij,Sk->RiSj', phase.conj(), s_k, phase)
    print(abs(s-s1.reshape(s.shape)).max())

    kmf = dft.KRKS(cell, kpts)
    ekpt = kmf.run()

    mf = k2gamma(kmf, kmesh)
    c_g_ao = mf.mo_coeff

    # The following is to check whether the MO is correctly coverted:

    print("Supercell gamma MO in AO basis from conversion:")
    scell = pbc_tools.super_cell(cell, kmesh)
    mf_sc = dft.RKS(scell)

    s = mf_sc.get_ovlp()
    mf_sc.run()
    sc_mo = mf_sc.mo_coeff

    nocc = scell.nelectron // 2
    print("Supercell gamma MO from direct calculation:")
    print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
    print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])

