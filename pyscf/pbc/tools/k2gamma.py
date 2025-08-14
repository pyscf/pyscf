#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
from fractions import Fraction
import itertools
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import group_by_conj_pairs

def kpts_to_kmesh(cell, kpts, precision=None, rcut=None):
    '''Find the minimal k-points mesh to include all input kpts'''
    kpts = np.asarray(kpts)
    assert kpts.ndim == 2
    scaled_kpts = cell.get_scaled_kpts(kpts)
    logger.debug3(cell, '    scaled_kpts kpts %s', scaled_kpts)
    # cell.nimgs are the upper limits for kmesh
    if rcut is None:
        kmesh = np.asarray(cell.nimgs) * 2 + 1
    else:
        nimgs = cell.get_bounding_sphere(rcut)
        kmesh = nimgs * 2 + 1
    if precision is None:
        precision = cell.precision * 1e2
    for i in range(3):
        floats = scaled_kpts[:,i]
        uniq_floats_idx = np.unique(floats.round(6), return_index=True)[1]
        uniq_floats = floats[uniq_floats_idx]
        fracs = [Fraction(x).limit_denominator(int(kmesh[i])) for x in uniq_floats]
        denominators = np.unique([x.denominator for x in fracs])
        common_denominator = reduce(np.lcm, denominators)
        fs = common_denominator * uniq_floats
        if abs(uniq_floats - np.rint(fs)/common_denominator).max() < precision:
            kmesh[i] = min(kmesh[i], common_denominator)
        if cell.verbose >= logger.DEBUG3:
            logger.debug3(cell, 'dim=%d common_denominator %d  error %g',
                          i, common_denominator, abs(fs - np.rint(fs)).max())
            logger.debug3(cell, '    unique kpts %s', uniq_floats)
            logger.debug3(cell, '    frac kpts %s', fracs)
    return kmesh

def translation_vectors_for_kmesh(cell, kmesh, wrap_around=False):
    '''
    Translation vectors to construct super-cell of which the gamma point is
    identical to the k-point mesh of primitive cell
    '''
    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    if wrap_around:
        R_rel_a[(kmesh[0]+1)//2:] -= kmesh[0]
        R_rel_b[(kmesh[1]+1)//2:] -= kmesh[1]
        R_rel_c[(kmesh[2]+1)//2:] -= kmesh[2]
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_vec_abs = np.dot(R_vec_rel, latt_vec)
    return R_vec_abs

def get_phase(cell, kpts, kmesh=None, wrap_around=False):
    '''
    The unitary transformation that transforms the supercell basis k-mesh
    adapted basis.
    '''
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    R_vec_abs = translation_vectors_for_kmesh(cell, kmesh, wrap_around)

    NR = len(R_vec_abs)
    phase = np.exp(1j*np.dot(R_vec_abs, kpts.T))
    phase /= np.sqrt(NR)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = tools.super_cell(cell, kmesh, wrap_around)
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

def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None):
    scell, phase = get_phase(cell, kpts, kmesh)

    E_g = np.hstack(mo_energy)
    C_k = np.asarray(mo_coeff)
    Nk, Nao, Nmo = C_k.shape
    NR = phase.shape[0]

    k_conj_groups = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    k_phase = np.eye(Nk, dtype=np.complex128)
    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    pairs = [[k, k_conj] for k, k_conj in k_conj_groups
             if k_conj is not None and k != k_conj]
    for idx in np.array(pairs):
        k_phase[idx[:,None],idx] = r2x2
    # Transform AO indices
    C_gamma = np.einsum('Rk,kum,kh->Ruhm', phase, C_k, k_phase)
    C_gamma = C_gamma.reshape(Nao*NR, Nk*Nmo)

    # Pure imaginary orbitals to real
    cR_max = abs(C_gamma.real).max(axis=0)
    C_gamma[:,cR_max < 1e-5] *= -1j

    E_sort_idx = np.argsort(E_g, kind='stable')
    E_g = E_g[E_sort_idx]

    cI_max = abs(C_gamma.imag).max(axis=0)
    if cI_max.max() < 1e-5:
        C_gamma = C_gamma.real[:,E_sort_idx]
    else:
        C_gamma = C_gamma[:,E_sort_idx]
        s = scell.pbc_intor('int1e_ovlp')
        # assert (abs(reduce(np.dot, (C_gamma.conj().T, s, C_gamma))
        #            - np.eye(Nmo*Nk)).max() < 1e-5)

        # For degenerated MOs, the transformed orbitals in super cell may not be
        # real. Construct a sub Fock matrix in super-cell to find a proper
        # transformation that makes the transformed MOs real.
        E_k_degen = abs(E_g[1:] - E_g[:-1]) < 1e-3
        degen_mask = np.append(False, E_k_degen) | np.append(E_k_degen, False)
        degen_mask[cI_max < 1e-5] = False
        if np.any(E_k_degen):
            c_rest = C_gamma[:,~degen_mask]
            if c_rest.size > 0 and abs(c_rest.imag).max() < 1e-4:
                shift = min(E_g[degen_mask]) - .1
                f = np.dot(C_gamma[:,degen_mask] * (E_g[degen_mask] - shift),
                           C_gamma[:,degen_mask].conj().T)
                assert (abs(f.imag).max() < 1e-4)

                e, na_orb = scipy.linalg.eigh(f.real, s, type=2)
                C_gamma = C_gamma.real
                C_gamma[:,degen_mask] = na_orb[:, e>1e-7]
            else:
                f = np.dot(C_gamma * E_g, C_gamma.conj().T)
                assert (abs(f.imag).max() < 1e-4)
                e, C_gamma = scipy.linalg.eigh(f.real, s, type=2)

    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    # overlap between k-point unitcell and gamma-point supercell
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(Nk,Nao,NR*Nao)
    # The unitary transformation from k-adapted orbitals to gamma-point orbitals
    mo_phase = lib.einsum('kum,kuv,vi->kmi', C_k.conj(), s_k_g, C_gamma)

    return scell, E_g, C_gamma, mo_phase

def k2gamma(kmf, kmesh=None):
    r'''
    convert the k-sampled mean-field object to the corresponding supercell
    gamma-point mean-field object.

    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \frac{1}{\sqrt{N_{\UC}}}
         \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}
    '''
    from pyscf.pbc import scf, dft
    if isinstance(kmf.kpts, KPoints):
        kmf = kmf.to_khf()

    def transform(mo_energy, mo_coeff, mo_occ):
        assert not isinstance(kmf.kpts, KPoints)
        kpts = kmf.kpts
        scell, E_g, C_gamma = mo_k2gamma(kmf.cell, mo_energy, mo_coeff,
                                         kpts, kmesh)[:3]
        E_sort_idx = np.argsort(np.hstack(mo_energy), kind='stable')
        mo_occ = np.hstack(mo_occ)[E_sort_idx]
        return scell, E_g, C_gamma, mo_occ

    mo_coeff = kmf.mo_coeff
    mo_energy = kmf.mo_energy
    mo_occ = kmf.mo_occ

    if isinstance(kmf, scf.khf.KRHF):
        scell, E_g, C_gamma, mo_occ = transform(mo_energy, mo_coeff, mo_occ)
    elif isinstance(kmf, scf.kuhf.KUHF):
        scell, Ea, Ca, occ_a = transform(mo_energy[0], mo_coeff[0], mo_occ[0])
        scell, Eb, Cb, occ_b = transform(mo_energy[1], mo_coeff[1], mo_occ[1])
        E_g = [Ea, Eb]
        C_gamma = [Ca, Cb]
        mo_occ = [occ_a, occ_b]
    else:
        raise NotImplementedError('SCF object %s not supported' % kmf)

    known_cls = {
        dft.kuks.KUKS  : dft.uks.UKS  ,
        dft.kroks.KROKS: dft.roks.ROKS,
        dft.krks.KRKS  : dft.rks.RKS  ,
        dft.kgks.KGKS  : dft.gks.GKS  ,
        scf.kuhf.KUHF  : scf.uhf.UHF  ,
        scf.krohf.KROHF: scf.rohf.ROHF,
        scf.khf.KRHF   : scf.hf.RHF   ,
        scf.kghf.KGHF  : scf.ghf.GHF  ,
    }
    if kmf.__class__ in known_cls:
        mf = known_cls[kmf.__class__](scell)
        mf.exxdiv = kmf.exxdiv
        if isinstance(mf, dft.KohnShamDFT):
            mf.xc = kmf.xc
    else:
        raise RuntimeError(f'k2gamma for SCF object {kmf} not supported.')

    mf.mo_coeff = C_gamma
    mf.mo_energy = E_g
    mf.mo_occ = mo_occ
    return mf


def to_supercell_ao_integrals(cell, kpts, ao_ints, kmesh=None, force_real=True):
    '''Transform from the unitcell k-point AO integrals to the supercell
    gamma-point AO integrals.
    '''
    scell, phase = get_phase(cell, kpts, kmesh=kmesh)
    NR, Nk = phase.shape
    nao = cell.nao
    scell_ints = np.einsum('Rk,kij,Sk->RiSj', phase, ao_ints, phase.conj())
    if force_real:
        return scell_ints.reshape(NR*nao,NR*nao).real
    else:
        return scell_ints.reshape(NR*nao,NR*nao)


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
    assert (abs(scell_ints.imag).max() < 1e-7)
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

    # The following is to check whether the MO is correctly converted:

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
