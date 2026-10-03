#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Jiachen Li <lijiachen.duke@gmail.com>
# Author: Jincheng Yu <pimetamon@gmail.com>
#

import numpy
import scipy

from pyscf import df, dft, pbc, scf
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import einsum, logger, StreamObject, current_memory
from pyscf.mp.mp2 import get_nocc, get_nmo
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.scf.hf import KohnShamDFT


def diagonalize_pprpa_singlet(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize singlet ppRPA matrix.

    Reference:
    [1] https://doi.org/10.1063/1.4828728

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): singlet correlation energy.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc + 1) * nocc / 2)  # number of hole-hole pairs

    # low triangular index (including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir)

    # A matrix: particle-particle block
    # two-electron integral part, <ab|cd>+<ab|dc>
    A = einsum("Pac,Pbd->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:])
    A += einsum("Pad,Pbc->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:])
    # scale the diagonal elements
    A[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # a=b
    A = A.transpose(2, 3, 0, 1)  # A_{ab,cd} to A_{cd,ab}
    A[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # c=d
    A = A.transpose(2, 3, 0, 1)  # A_{cd,ab} to A_{ab,cd}
    # orbital energy part
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[nocc:, None] + mo_energy[None, nocc:])
    orb_sum = orb_sum.reshape(-1) - 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>+<ab|ji>
    B = einsum("Pai,Pbj->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc])
    B += einsum("Paj,Pbi->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc])
    # scale the diagonal elements
    B[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # a=b
    B = B.transpose(2, 3, 0, 1)  # B_{ab,ij} to B_{ij,ab}
    B[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # i=j
    B = B.transpose(2, 3, 0, 1)  # B_{ij,ab} to B_{ab,ij}
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>+<ij|lk>
    C = einsum("Pik,Pjl->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc])
    C += einsum("Pil,Pjk->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc])
    # scale the diagonal elements
    C[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # i=j
    C = C.transpose(2, 3, 0, 1)  # C_{ij,kl} to C_{kl,ij}
    C[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # k=l
    C = C.transpose(2, 3, 0, 1)  # C_{kl,ij} to C_{ij,kl}
    # orbital energy part
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] + mo_energy[None, :nocc])
    orb_sum = orb_sum.reshape(-1) - 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # | B A   |
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenvalue and eigenvectors by ascending order
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]

    pprpa_orthonormalize_eigenvector(multi="s", nocc=nocc, exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize triplet ppRPA matrix.

    Reference:
    [1] https://doi.org/10.1063/1.4828728

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): triplet correlation energy, with a factor of 3.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc - 1) * nocc / 2)  # number of hole-hole pairs

    # low triangular index (not including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    # A matrix: particle-particle block
    # two-electron integral part, <ab|cd>-<ab|dc>
    A = einsum("Pac,Pbd->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:])
    A -= einsum("Pad,Pbc->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:])
    # orbital energy part
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[nocc:, None] + mo_energy[None, nocc:])
    orb_sum = orb_sum.reshape(-1) - 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>-<ab|ji>
    B = einsum("Pai,Pbj->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc])
    B -= einsum("Paj,Pbi->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc])
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>-<ij|lk>
    C = einsum("Pik,Pjl->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc])
    C -= einsum("Pil,Pjk->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc])
    # orbital energy part
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] + mo_energy[None, :nocc])
    orb_sum = orb_sum.reshape(-1) - 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B  A   |
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenvalue and eigenvectors by ascending order
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]

    pprpa_orthonormalize_eigenvector(multi="t", nocc=nocc, exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = (sum_exci - trace_A) * 3.0

    return exci, xy, ec


# utility function
def ij2index(r, c, row, col):
    """Get index of a row and column in a square matrix in a lower triangular matrix.

    Args:
        r (int): row index in s square matrix.
        c (int): column index in s square matrix.
        row (int array): row index array of a lower triangular matrix.
        col (int array): column index array of a lower triangular matrix.

    Returns:
        i (int): index in the lower triangular matrix.
    """
    for i in range(len(row)):
        if r == row[i] and c == col[i]:
            return i

    raise ValueError("cannot find the index!")


def inner_product(u, v, oo_dim):
    """Calculate inner product between two ppRPA eigenvectors.
    product = <Y1,Y2> - <X1,X2>, where X is occ-occ block, Y is vir-vir block.

    Args:
        u (double array): first vector.
        v (double array): second vector
        oo_dim (int): occ-occ block dimension

    Returns:
        inp (double): inner product.
    """
    product = numpy.sum(u[oo_dim:] * v[oo_dim:])
    product -= numpy.sum(u[:oo_dim] * v[:oo_dim])
    return product


def get_chemical_potential(nocc, mo_energy):
    """Get chemical potential as the average between HOMO and LUMO.
    In the case there is no occupied or virtual orbital, return 0.

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array/list): orbital energy.

    Returns:
        mu (double): chemical potential.
    """
    if nocc == 0:
        mu = 0.0
    else:
        mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5
    return mu


def ao2mo(pprpa):
    """Get three-center density-fitting matrix in MO active space.

    Args:
        pprpa: ppRPA object.

    Returns:
        Lpq (double ndarray): three-center DF matrix in MO active space.
    """
    mf = pprpa._scf
    mo_coeff = mf.mo_coeff
    nocc = pprpa.nocc
    nocc_act, nvir_act, nmo_act = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act

    nao = mo_coeff.shape[0]
    mo = numpy.asarray(mo_coeff, order='F')
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    if isinstance(mf, (scf.rhf.RHF, dft.rks.RKS)):
        # molecule
        if getattr(mf, 'with_df', None):
            pprpa.with_df = mf.with_df
        else:
            pprpa.with_df = df.DF(mf.mol)
            if pprpa.auxbasis is not None:
                pprpa.with_df.auxbasis = pprpa.auxbasis
            else:
                pprpa.with_df.auxbasis = df.make_auxbasis(
                    mf.mol, mp2fit=True)
        pprpa._keys.update(['with_df'])

        naux = pprpa.with_df.get_naoaux()
        mem_incore = (2*nao**2*naux) * 8/1e6
        mem_now = current_memory()[0]

        Lpq = None
        if (mem_incore + mem_now < pprpa.max_memory) or pprpa.mol.incore_anyway:
            Lpq = nr_e2(pprpa.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux, nmo_act, nmo_act)
        else:
            logger.warn(pprpa, 'Memory may not be enough!')
            raise NotImplementedError
    elif isinstance(mf, (pbc.scf.rhf.RHF, pbc.dft.rks.RKS)):
        # supercell
        if getattr(mf, 'with_df', None):
            pprpa.with_df = mf.with_df
        else:
            pprpa.with_df = df.DF(mf.mol)
            if pprpa.auxbasis is not None:
                pprpa.with_df.auxbasis = pprpa.auxbasis
            else:
                pprpa.with_df.auxbasis = pbc.df.make_auxbasis(
                    mf.mol, mp2fit=True)
        pprpa._keys.update(['with_df'])

        naux = pprpa.with_df.get_naoaux()
        mem_incore = (nao**2*naux) * 8/1e6
        mem_now = current_memory()[0]
        max_memory = max(4000, pprpa.max_memory - mem_now - mem_incore)

        kptijkl = _format_kpts(mf.with_df.kpts)
        Lpq = []
        for LpqR, _, _ in mf.with_df.sr_loop(kptijkl[:2],
                                             max_memory=0.8*max_memory,
                                             compact=False):
            LpqR = LpqR.reshape(-1, nao, nao)
            tmp = None
            tmp = nr_e2(LpqR, mo, ijslice, aosym='s1', mosym='s1', out=tmp)
            Lpq.append(tmp)
        Lpq = numpy.vstack(Lpq).reshape(naux, nmo_act, nmo_act)
        return Lpq


def pprpa_orthonormalize_eigenvector(multi, nocc, exci, xy):
    """Orthonormalize ppRPA eigenvector.
    The eigenvector is normalized as Y^2 - X^2 = 1.
    This function will rewrite input exci and xy, after calling this function,
    exci and xy will be re-ordered as [hole-hole, particle-particle].

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nroot = xy.shape[0]

    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)

    # determine the vector is pp or hh
    sig = numpy.zeros(shape=[nroot], dtype=numpy.double)
    for i in range(nroot):
        sig[i] = 1 if inner_product(xy[i], xy[i], oo_dim) > 0 else -1

    # eliminate parallel component
    for i in range(nroot):
        for j in range(i):
            if abs(exci[i] - exci[j]) < 1.0e-7:
                inp = inner_product(xy[i], xy[j], oo_dim)
                xy[i] -= sig[j] * xy[j] * inp

    # normalize
    for i in range(nroot):
        inp = inner_product(xy[i], xy[i], oo_dim)
        inp = numpy.sqrt(abs(inp))
        xy[i] /= inp

    # re-order all states by signs, first hh then pp
    hh_index = numpy.where(sig < 0)[0]
    pp_index = numpy.where(sig > 0)[0]
    exci_hh = exci[hh_index]
    exci_pp = exci[pp_index]
    exci[:len(hh_index)] = exci_hh
    exci[len(hh_index):] = exci_pp
    xy_hh = xy[hh_index]
    xy_pp = xy[pp_index]
    xy[:len(hh_index)] = xy_hh
    xy[len(hh_index):] = xy_pp

    return


# analysis functions
def pprpa_print_direct_eigenvector(pprpa, multi, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        pprpa (RppRPADirect): ppRPA object.
        multi (string): multiplicity.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nocc = pprpa.nocc
    nocc_act, nvir_act = pprpa.nocc_act, pprpa.nvir_act
    nocc_fro = nocc - nocc_act
    if multi == "s":
        oo_dim = int((nocc_act + 1) * nocc_act / 2)
        vv_dim = int((nvir_act + 1) * nvir_act / 2)
        is_singlet = 1
        logger.info(pprpa, "\n     print ppRPA excitations: singlet\n")
    elif multi == "t":
        oo_dim = int((nocc_act - 1) * nocc_act / 2)
        vv_dim = int((nvir_act - 1) * nvir_act / 2)
        is_singlet = 0
        logger.info(pprpa, "\n     print ppRPA excitations: triplet\n")

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, is_singlet-1)

    au2ev = 27.211386

    for istate in range(min(pprpa.hh_state, oo_dim)):
        logger.info(pprpa, "#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
            istate + 1, multi, (exci[oo_dim-istate-1] - exci0) * au2ev,
            exci[oo_dim-istate-1] * au2ev)
        full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=numpy.double)
        full[tri_row_o, tri_col_o] = xy[oo_dim-istate-1][:oo_dim]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > pprpa.print_thresh)
        for i, j in pairs:
            pprpa_print_a_pair(
                pprpa, is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                percentage=full[i, j])

        full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=numpy.double)
        full[tri_row_v, tri_col_v] = xy[oo_dim-istate-1][oo_dim:]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > pprpa.print_thresh)
        for a, b in pairs:
            pprpa_print_a_pair(
                pprpa, s_pp=True, p=a+nocc_fro+nocc_act, q=b+nocc_fro+nocc_act,
                percentage=full[a, b])

        logger.info(pprpa, "")

    for istate in range(min(pprpa.pp_state, vv_dim)):
        logger.info(pprpa, "#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
            istate + 1, multi, (exci[oo_dim+istate] - exci0) * au2ev,
            exci[oo_dim+istate] * au2ev)
        full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=numpy.double)
        full[tri_row_o, tri_col_o] = xy[oo_dim+istate][:oo_dim]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > pprpa.print_thresh)
        for i, j in pairs:
            pprpa_print_a_pair(
                pprpa, is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                percentage=full[i, j])

        full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=numpy.double)
        full[tri_row_v, tri_col_v] = xy[oo_dim+istate][oo_dim:]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > pprpa.print_thresh)
        for a, b in pairs:
            pprpa_print_a_pair(
                pprpa, is_pp=True, p=a+nocc_fro+nocc_act, q=b+nocc_fro+nocc_act,
                percentage=full[a, b])

        logger.info(pprpa, "")

    return


def analyze_pprpa_direct(pprpa):
    """Analyze ppRPA (direct diagonalization) excited states.

    Args:
        pprpa (RppRPADirect): ppRPA object.
    """
    oo_dim_s = int((pprpa.nocc_act + 1) * pprpa.nocc_act / 2)
    oo_dim_t = int((pprpa.nocc_act - 1) * pprpa.nocc_act / 2)

    logger.info(pprpa, "\nanalyze ppRPA results.")
    if pprpa.exci_s is not None and pprpa.exci_t is not None:
        logger.info(pprpa, "both singlet and triplet results found.")
        if pprpa.nelec == "n-2":
            exci0 = min(pprpa.exci_s[oo_dim_s], pprpa.exci_t[oo_dim_t])
        else:
            exci0 = max(pprpa.exci_s[oo_dim_s-1], pprpa.exci_t[oo_dim_t-1])
        pprpa_print_direct_eigenvector(
            pprpa, multi="s", exci0=exci0, exci=pprpa.exci_s, xy=pprpa.xy_s)
        pprpa_print_direct_eigenvector(
            pprpa, multi="t", exci0=exci0, exci=pprpa.exci_t, xy=pprpa.xy_t)
    else:
        if pprpa.exci_s is not None:
            logger.info(pprpa, "only singlet results found.")
            exci0 = pprpa.exci_s[oo_dim_s if pprpa.nelec == "n-2" else oo_dim_s-1]
            pprpa_print_direct_eigenvector(
                pprpa, multi="s", exci0=exci0, exci=pprpa.exci_s, xy=pprpa.xy_s)
        else:
            logger.info(pprpa, "only triplet results found.")
            exci0 = pprpa.exci_s[oo_dim_t if pprpa.nelec == "n-2" else oo_dim_t-1]
            pprpa_print_direct_eigenvector(
                pprpa, multi="t", exci0=exci0, exci=pprpa.exci_t, xy=pprpa.xy_t)
    return


def pprpa_print_a_pair(pprpa, is_pp, p, q, percentage):
    """Print the percentage of a pair in the eigenvector.

    Args:
        pprpa: ppRPA object.
        is_pp (bool): the eigenvector is in particle-particle channel.
        p (int): MO index of the first orbital.
        q (int): MO index of the second orbital.
        percentage (double): the percentage of this pair.
    """
    if is_pp:
        logger.info(pprpa, "    particle-particle pair: %5d %5d   %5.2f%%",
            p + 1, q + 1, percentage * 100)
    else:
        logger.info(pprpa, "    hole-hole pair:         %5d %5d   %5.2f%%",
            p + 1, q + 1, percentage * 100)
    return


class RppRPADirect(StreamObject):
    def __init__(
            self, mf, nocc_act=None, nvir_act=None, hh_state=5, pp_state=5,
            nelec="n-2", print_thresh=0.1, auxbasis=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        # options
        self.nocc_act = nocc_act  # number of active occupied orbitals
        self.nvir_act = nvir_act  # number of active virtual orbitals
        self.hh_state = hh_state  # number of hole-hole states to print
        self.pp_state = pp_state  # number of particle-particle states to print
        self.nelec = nelec  # "n-2" or "n+2" for system is an N-2 or N+2 system
        self.print_thresh = print_thresh  # threshold to print component
        self.auxbasis = auxbasis  # auxiliary basis set

        # internal flags
        self.multi = None  # multiplicity
        self.mu = None  # chemical potential
        self.nmo_act = None  # number of active orbitals
        self.mo_energy_act = None  # orbital energy in active space

        # results
        self.ec = None  # correlation energy
        self.ec_s = None  # singlet correlation energy
        self.ec_t = None  # triplet correlation energy
        self.exci = None  # two-electron addition energy
        self.xy = None  # ppRPA eigenvector
        self.exci_s = None  # singlet two-electron addition energy
        self.xy_s = None  # singlet two-electron addition eigenvector
        self.exci_t = None  # triplet two-electron addition energy
        self.xy_t = None  # triplet two-electron addition eigenvector

        ##################################################
        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of molecular orbitals
        self.nvir = None  # number of virtual orbitals
        self.mo_energy = numpy.array(self._scf.mo_energy)  # orbital energy
        self.Lpq = None  # three-center density-fitting matrix in MO
        self.mo_occ = self._scf.mo_occ  # used in get_nocc() and get_nmo(), not in ppRPA
        self.frozen = 0  # used in get_nocc() and get_nmo(), not in ppRPA

        return

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    ao2mo = ao2mo
    analyze = analyze_pprpa_direct

    def check_parameter(self):
        """Initialize and check options.
        """
        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ["n-2", "n+2"]

        self.nvir = self.nmo - self.nocc

        # adjust active space
        if self.nocc_act is None:
            self.nocc_act = self.nocc
        else:
            self.nocc_act = min(self.nocc_act, self.nocc)
        if self.nvir_act is None:
            self.nvir_act = self.nvir
        else:
            self.nvir_act = min(self.nvir_act, self.nvir)

        self.nmo_act = self.nocc_act + self.nvir_act
        self.mo_energy_act = self.mo_energy[self.nocc-self.nocc_act:self.nocc+self.nvir_act]

        if self.mu is None:
            self.mu = get_chemical_potential(self.nocc, self.mo_energy)

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('\n******** %s ********', self.__class__)
        if self.multi == "s":
            oo_dim = int((self.nocc_act + 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act + 1) * self.nvir_act / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc_act - 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act - 1) * self.nvir_act / 2)
        full_dim = oo_dim + vv_dim
        multiplicity = "singlet" if self.multi == "s" else "triplet"
        log.info('multiplicity = %s', multiplicity)
        log.info('nmo = %d', self.nmo)
        log.info('nocc = %d nvir = %d', self.nocc, self.nvir)
        log.info('nocc_act = %d nvir_act = %d', self.nocc_act, self.nvir_act)
        log.info('occ-occ dimension = %d vir-vir dimension = %d', oo_dim, vv_dim)
        log.info('full dimension = %d', full_dim)
        log.info('interested hh state = %d', self.hh_state)
        log.info('interested pp state = %d', self.pp_state)
        log.info('ground state = %s', self.nelec)
        log.info('print threshold = %.2f%%', self.print_thresh*100)
        log.info('')
        return

    def check_memory(self):
        """Check required memory.
        In direct diagonalization, dominant memory cost is saving A and full
        ppRPA matrix.
        """
        log = logger.Logger(self.stdout, self.verbose)
        if self.multi == "s":
            oo_dim = int((self.nocc + 1) * self.nocc / 2)
            vv_dim = int((self.nvir + 1) * self.nvir / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc - 1) * self.nocc / 2)
            vv_dim = int((self.nvir - 1) * self.nvir / 2)
        full_dim = oo_dim + vv_dim

        # ppRPA matrix: A block and full matrix, eigenvector
        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if mem < 1000:
            log.info("ppRPA needs at least %d MB memory.", mem)
        else:
            log.info("ppRPA needs at least %.1f GB memory.", mem / 1.0e3)
        return

    def kernel(self, multi):
        """Run ppRPA direct diagonalization.

        Args:
            multi (char): multiplicity.
        """
        self.multi = multi
        self.check_parameter()
        self.check_memory()

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        if self.Lpq is None:
            self.Lpq = self.ao2mo()
        if self.multi == "s":
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc_act, mo_energy=self.mo_energy_act, Lpq=self.Lpq,
                mu=self.mu)
        elif multi == "t":
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc_act, mo_energy=self.mo_energy_act, Lpq=self.Lpq,
                mu=self.mu)
        logger.timer(self, "ppRPA direct: %s" % multi, *cput0)
        return

    def get_correlation(self):
        """Get ppRPA correlation energy.
        Triplet contribution is multiplied by a factor of 3.

        Reference:
        [1] https://doi.org/10.1063/1.4828728

        Returns:
            ec (double): ppRPA correlation energy.
        """
        self.check_parameter()

        if self.Lpq is None:
            self.Lpq = self.ao2mo()
        if self.ec_s is None:
            cput0 = (logger.process_clock(), logger.perf_counter())
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc_act, mo_energy=self.mo_energy_act, Lpq=self.Lpq,
                mu=self.mu)
            logger.timer(self, "ppRPA correlation energy: singlet", *cput0)

        if self.ec_t is None:
            cput0 = (logger.process_clock(), logger.perf_counter())
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc_act, mo_energy=self.mo_energy_act, Lpq=self.Lpq,
                mu=self.mu)
            logger.timer(self, "ppRPA correlation energy: triplet", *cput0)

        self.ec = self.ec_s + self.ec_t

        return self.ec

    def energy_tot(self):
        """Get ppRPA total energy.
        Total energy = Hartree-Fock energy + ppRPA correlation energy.

        Returns:
            e_tot (double): ppRPA total energy.
        """
        mf = self._scf
        assert mf.converged
        hf_obj = mf if not isinstance(mf, KohnShamDFT) else mf.to_hf()

        dm = hf_obj.make_rdm1()
        e_hf = hf_obj.energy_nuc() + hf_obj.energy_elec(dm=dm)[0]
        ec = self.get_correlation()
        e_tot = e_hf + ec

        return e_tot, e_hf, ec
