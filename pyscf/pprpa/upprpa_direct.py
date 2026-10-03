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
from pyscf.lib import einsum, logger, StreamObject, current_memory
from pyscf.mp.ump2 import get_nocc, get_nmo
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.scf.hf import KohnShamDFT
from pyscf.pprpa.rpprpa_direct import inner_product, ij2index, \
    pprpa_orthonormalize_eigenvector, \
    pprpa_print_a_pair, diagonalize_pprpa_triplet


def upprpa_orthonormalize_eigenvector(subspace, nocc, exci, xy):
    """Orthonormalize U-ppRPA eigenvectors.
    The eigenvector is normalized as Y^2 - X^2 = 1.
    This function will rewrite input exci and xy, after calling this function,
    exci and xy will be re-ordered as [hole-hole, particle-particle].

    Args:
        subspace (str): subspace, 'aaaa', 'bbbb', or 'abab'.
        nocc (int/tuple of int): number of occupied orbitals.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nroot = xy.shape[0]

    if subspace == 'abab':
        oo_dim = int(nocc[0] * nocc[1])

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

        # change |X -Y> to |X Y>
        xy[:][:oo_dim] *= -1

    else:
        pprpa_orthonormalize_eigenvector('t', nocc, exci, xy)

    return


def get_chemical_potential(nocc, mo_energy):
    """Get chemical potential as the average between HOMO and LUMO.
    In the case there is no occupied or virtual orbital, return0.

    Args:
        nocc (list/tuple of int): number of occupied orbitals, [alpha, beta].
        mo_energy (double ndarrya/list): orbital energy.

    Returns:
        mu (double): chemical potential.
    """
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    if (nocc[0] == nocc[1] == 0) or (nocc[0] == nmo[0] and nocc[1] == nmo[1]):
        mu = 0.0
    else:
        assert nocc[0] >= nocc[1]
        if nocc[1] == 0:
            homo = mo_energy[0][nocc[0]-1]
        else:
            homo = max(mo_energy[0][nocc[0]-1], mo_energy[1][nocc[1]-1])
        lumo = min(mo_energy[0][nocc[0]], mo_energy[1][nocc[1]])
        mu = (homo + lumo) * 0.5

    return mu


def ao2mo(pprpa):
    """Get three-center density-fitting matrix in MO active space.

    Args:
        pprpa: ppRPA object.

    Returns:
        Lpq (double ndarray): three-center DF matrices in MO active space.
    """
    mf = pprpa._scf
    mo_coeff = numpy.asarray(mf.mo_coeff)
    nocc = pprpa.nocc
    nocc_act, nvir_act, nmo_act = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act

    nao = mo_coeff[0].shape[0]
    ijslice = [
        (
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0],
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0]),
        (
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1],
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1]),
    ]

    if isinstance(mf, (scf.uhf.UHF, dft.uks.UKS)):
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

        Lpq_a = None
        Lpq_b = None
        if (mem_incore + mem_now < pprpa.max_memory) or pprpa.mol.incore_anyway:
            Lpq_a = nr_e2(
                pprpa.with_df._cderi, mo_coeff[0], ijslice[0],
                aosym='s2', out=Lpq_a)
            Lpq_b = nr_e2(
                pprpa.with_df._cderi, mo_coeff[1], ijslice[1],
                aosym='s2', out=Lpq_b)
            Lpq_a = Lpq_a.reshape(naux, nmo_act[0], nmo_act[0])
            Lpq_b = Lpq_b.reshape(naux, nmo_act[1], nmo_act[1])
            return numpy.asarray([Lpq_a, Lpq_b])
        else:
            logger.warn(pprpa, 'Memory may not be enough!')
            raise NotImplementedError
    elif isinstance(mf, (pbc.scf.uhf.UHF, pbc.dft.uks.UKS)):
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
        mo_a = numpy.asarray(mo_coeff[0], order='F')
        mo_b = numpy.asarray(mo_coeff[1], order='F')
        eri_3d = []
        Lpq_a, Lpq_b = [], []

        for LpqR, _, _ in mf.with_df.sr_loop(kptijkl[:2],
                                             max_memory=0.3*max_memory,
                                             compact=False):
            tmp_a, tmp_b = None, None
            tmp_a = nr_e2(
                LpqR.reshape(-1,nao,nao), mo_a, ijslice[0], aosym='s1',
                mosym='s1', out=tmp_a)
            tmp_b = nr_e2(
                LpqR.reshape(-1,nao,nao), mo_b, ijslice[1], aosym='s1',
                mosym='s1', out=tmp_b)
            Lpq_a.append(tmp_a)
            Lpq_b.append(tmp_b)
        Lpq_a = numpy.vstack(Lpq_a).reshape(-1, nmo_act[0], nmo_act[0])
        Lpq_b = numpy.vstack(Lpq_b).reshape(-1, nmo_act[1], nmo_act[1])
        eri_3d = numpy.asarray([Lpq_a, Lpq_b])

        return eri_3d


def diagonalize_pprpa_subspace_same_spin(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize UppRPA matrix in subspace (alpha alpha, alpha alpha)
    or (beta beta, beta beta).

    See function `pprpa_direct.diagonalize_pprpa_triplet`.

    """
    exci, xy, ec = diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, mu=mu)

    return exci, xy, ec/3.0


def diagonalize_pprpa_subspace_diff_spin(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize UppRPA matrix in subspace (alpha beta, alpha beta).

    Reference:
    [1] https://doi.org/10.1063/1.4828728 (equation 14)

    Args:
        nocc(tuple of int): number of occupied orbitals, (nalpha, nbeta).
        mo_energy (list of double array): orbital energies.
        Lpq (list of double ndarray): three-center RI matrices in MO space.

    Kwarg:
        mu (double): chemical potential.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    # ===========================> A matrix <============================
    # <ab|cd>
    A = einsum(
        'Pac,Pbd->abcd', Lpq[0][:, nocc[0]:, nocc[0]:],
        Lpq[1][:, nocc[1]:, nocc[1]:], optimize=True)
    # delta_ac delta_bd (e_a + e_b - 2 * mu)
    A = A.reshape(nvir[0]*nvir[1], nvir[0]*nvir[1])
    orb_sum = numpy.asarray(
        mo_energy[0][nocc[0]:, None] + mo_energy[1][None, nocc[1]:]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    trace_A = numpy.trace(A)

    # ===========================> B matrix <============================
    # <ab|ij>
    B = einsum(
        'Pai,Pbj->abij', Lpq[0][:, nocc[0]:, :nocc[0]],
        Lpq[1][:, nocc[1]:, :nocc[1]], optimize=True)
    B = B.reshape(nvir[0]*nvir[1], nocc[0]*nocc[1])

    # ===========================> C matrix <============================
    # <ij|kl>
    C = einsum(
        'Pik,Pjl->ijkl', Lpq[0][:, :nocc[0], :nocc[0]],
        Lpq[1][:, :nocc[1], :nocc[1]], optimize=True)
    # - delta_ik delta_jl (e_i + e_j - 2 * mu)
    C = C.reshape(nocc[0]*nocc[1], nocc[0]*nocc[1])
    orb_sum = numpy.asarray(
        mo_energy[0][:nocc[0], None] + mo_energy[1][None, :nocc[1]]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)

    # ==================> whole matrix in the subspace<==================
    # C    B^T
    # B     A
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, where W is the metric matrix [[-I, 0], [0, I]]
    M[:nocc[0]*nocc[1]][:] *= -1.0

    # =====================> solve for eigenpairs <======================
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenpairs
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]
    upprpa_orthonormalize_eigenvector('abab', nocc, exci, xy)

    sum_exci = numpy.sum(exci[nocc[0]*nocc[1]:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def pprpa_print_direct_eigenvector(
        pprpa, subspace, nocc, nvir, nocc_fro, thresh, hh_state,
        pp_state, exci0, exci, xy):
    """Print components of an eigenvector.

    Args:
        pprpa (UppRPADirect object): unrestricted pprpa object.
        subspace (str): subspace, 'aaaa', 'bbbb', or 'abab'.
        nocc (int/tuple of int): number of occupied orbitals.
        nvir (int/tuple of int): number of virtual orbitals.
        nocc_fro (int/tuple of int): number of frozen occupied orbitals.
        thresh (double): threshold to print a pair.
        hh_state (int): number of interested hole-hole states.
        pp_state (int): number of interested particle-particle states.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    if subspace == 'aaaa':
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        print("\n     print U-ppRPA excitations: (alpha alpha, alpha alpha)\n")
    elif subspace == 'bbbb':
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        print("\n     print U-ppRPA excitations: (beta beta, beta beta)\n")
    elif subspace == 'abab':
        oo_dim = int(nocc[0] * nocc[1])
        vv_dim = int(nvir[0] * nvir[1])
        print("\n     print U-ppRPA excitations: (alpha beta, alpha beta)\n")
    else:
        raise ValueError("Not recognized subspace: %s." % subspace)

    if subspace == 'aaaa' or subspace == 'bbbb':
        tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
        tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    au2ev = 27.211386

    # =====================> two-electron removal <======================
    for istate in range(min(hh_state, oo_dim)):
        print("#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, subspace[:2], (exci[oo_dim-istate-1] - exci0) * au2ev,
               exci[oo_dim-istate-1] * au2ev))
        if subspace == 'aaaa' or subspace == 'bbbb':
            full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[oo_dim-istate-1][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(pprpa, is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                                   percentage=full[i, j])

            full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[oo_dim-istate-1][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(pprpa, is_pp=True, p=a+nocc_fro+nocc,
                                   q=b+nocc_fro+nocc, percentage=full[a, b])

        else:
            full = xy[oo_dim-istate-1][:oo_dim].reshape(nocc[0], nocc[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(pprpa, is_pp=False, p=i+nocc_fro[0], q=j+nocc_fro[1],
                                   percentage=full[i, j])

            full = xy[oo_dim-istate-1][oo_dim:].reshape(nvir[0], nvir[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(pprpa, is_pp=True, p=a+nocc_fro[0]+nocc[0],
                                   q=b+nocc_fro[1]+nocc[1], percentage=full[a, b])
        print("")

    # =====================> two-electron addition <=====================
    for istate in range(min(pp_state, vv_dim)):
        print("#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, subspace[:2], (exci[oo_dim+istate] - exci0) * au2ev,
               exci[oo_dim+istate] * au2ev))
        if subspace == 'aaaa' or subspace == 'bbbb':
            full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[oo_dim+istate][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(pprpa, is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                                   percentage=full[i, j])

            full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[oo_dim+istate][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(pprpa, is_pp=True, p=a+nocc_fro+nocc,
                                   q=b+nocc_fro+nocc, percentage=full[a, b])

        else:
            full = xy[oo_dim+istate][:oo_dim].reshape(nocc[0], nocc[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=False, p=i+nocc_fro[0], q=j+nocc_fro[1],
                    percentage=full[i, j])

            full = xy[oo_dim+istate][oo_dim:].reshape(nvir[0], nvir[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=True, p=a+nocc_fro[0]+nocc[0],
                    q=b+nocc_fro[1]+nocc[1], percentage=full[a, b])
        print("")

    return


def analyze_pprpa_direct(pprpa):
    """Analyze ppRPA (direct diagonalization) excited states.

    Args:
        pprpa (UppRPADirect): ppRPA object.
    """
    logger.info(pprpa, '\nanalyze U-ppRPA results.')
    nocc_fro = (
        pprpa.nocc[0] - pprpa.nocc_act[0],
        pprpa.nocc[1] - pprpa.nocc_act[1])
    oo_dim_aa = int((pprpa.nocc_act[0] - 1) * pprpa.nocc_act[0] / 2)
    oo_dim_bb = int((pprpa.nocc_act[1] - 1) * pprpa.nocc_act[1] / 2)
    oo_dim_ab = int(pprpa.nocc_act[0] * pprpa.nocc_act[1])

    exci_aa = pprpa.exci[0]
    exci_bb = pprpa.exci[1]
    exci_ab = pprpa.exci[2]

    exci0_list = []
    if exci_aa is not None:
        logger.info(pprpa, '(alpha alpha, alpha alpha) results found.')
        if pprpa.nelec == 'n-2':
            exci0_list.append(exci_aa[oo_dim_aa])
        else:
            exci0_list.append(exci_aa[oo_dim_aa - 1])
    if exci_bb is not None:
        logger.info(pprpa, '(beta beta, beta beta) results found.')
        if pprpa.nelec == 'n-2':
            exci0_list.append(exci_bb[oo_dim_bb])
        else:
            exci0_list.append(exci_bb[oo_dim_bb - 1])
    if exci_ab is not None:
        logger.info(pprpa, '(alpha beta, alpha beta) results found.')
        if pprpa.nelec == 'n-2':
            exci0_list.append(exci_ab[oo_dim_ab])
        else:
            exci0_list.append(exci_ab[oo_dim_ab - 1])

    if pprpa.nelec == 'n-2':
        exci0 = min(exci0_list)
    else:
        exci0 = max(exci0_list)

    if exci_aa is not None:
        pprpa_print_direct_eigenvector(
            pprpa, 'aaaa', pprpa.nocc_act[0], pprpa.nvir_act[0], nocc_fro[0],
            pprpa.print_thresh, pprpa.hh_state,
            pprpa.pp_state, exci0, exci_aa, pprpa.xy[0])
    if exci_bb is not None:
        pprpa_print_direct_eigenvector(
            pprpa, 'bbbb', pprpa.nocc_act[1], pprpa.nvir_act[1], nocc_fro[1],
            pprpa.print_thresh, pprpa.hh_state,
            pprpa.pp_state, exci0, exci_bb, pprpa.xy[1])
    if exci_ab is not None:
        pprpa_print_direct_eigenvector(
            pprpa, 'abab', pprpa.nocc_act, pprpa.nvir_act, nocc_fro,
            pprpa.print_thresh, pprpa.hh_state, pprpa.pp_state,
            exci0, exci_ab, pprpa.xy[2])


class UppRPADirect(StreamObject):
    def __init__(
            self, mf, nocc_act=None, nvir_act=None, hh_state=5, pp_state=5,
            nelec='n-2', print_thresh=0.1, auxbasis=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        # options
        self.nocc_act = nocc_act # number of active occupied orbitals (alpha)
        self.nvir_act = nvir_act # number of active virtual orbitals (alpha)
        self.hh_state = hh_state # number of hole-hole states to print
        self.pp_state = pp_state # number of particle-particle states to print
        self.nelec = nelec # 'n-2' for N-2 system, 'n+2' for N+2 system
        self.print_thresh = print_thresh # threshold to print component
        self.auxbasis = auxbasis # auxiliary basis set to construct Lpq

        # ======================> internal flags <=======================
        self.mu = None # chemical potential
        self.nmo_act = None # number of active orbitals
        self.mo_energy_act = None # orbital energy in active space
        self.subspace = None # subspace(s) to perform ppRPA calculations

        # =========================> results <===========================
        self.ec = [None, None, None]  # correlation energy [aaaa, bbbb, abab]
        self.exci = [None, None, None]  # two-electron addition energy [aaaa, bbbb, abab]
        self.xy = [None, None, None]  # ppRPA eigenvector [aaaa, bbbb, abab]

        # ===============================================================
        # don't modify the following attributes, they are not input options
        self._nocc = None # number of occupied orbitals
        self._nmo = None # number of molecular orbitals
        self.nvir = None # number of virtual orbitals
        self.mo_energy = numpy.asarray(self._scf.mo_energy) # MO energies
        self.Lpq = None # three-center density-fitting matrix in MO
        self.mo_occ = self._scf.mo_occ
        self.frozen = 0

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
        "Initialize and check options."
        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ["n-2", "n+2"]

        self.nvir = (self.nmo[0] - self.nocc[0], self.nmo[1] - self.nocc[1])

        # adjust active space
        if self.nocc_act is None:
            self.nocc_act = self.nocc
        else:
            if isinstance(self.nocc_act, (int, numpy.integer)):
                self.nocc_act = (self.nocc_act, self.nocc_act)
            self.nocc_act = (
                min(self.nocc_act[0], self.nocc[0]),
                min(self.nocc_act[1], self.nocc[1]))
        if self.nvir_act is None:
            self.nvir_act = self.nvir
        else:
            if isinstance(self.nvir_act, (int, numpy.integer)):
                self.nvir_act = (self.nvir_act, self.nvir_act)
            self.nvir_act = (
                min(self.nvir_act[0], self.nvir[0]),
                min(self.nvir_act[1], self.nvir[1]))

        self.nmo_act = (self.nocc_act[0] + self.nvir_act[0],
                        self.nocc_act[1] + self.nvir_act[1])
        nocc = self.nocc
        nocc_act = self.nocc_act
        nvir_act = self.nvir_act
        self.mo_energy_act = (
            self.mo_energy[0][nocc[0]-nocc_act[0]:nocc[0]+nvir_act[0]],
            self.mo_energy[0][nocc[1]-nocc_act[1]:nocc[1]+nvir_act[1]])
        if self.mu is None:
            self.mu = get_chemical_potential(self.nocc, self.mo_energy)
        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        # ====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(self.nvir_act[0] * (self.nvir_act[0] + 1) / 2)
        aaoo_dim = int(self.nocc_act[0] * (self.nocc_act[0] + 1) / 2)
        # (alpha, beta) subspace
        abvv_dim = int(self.nvir_act[0] * self.nvir_act[1])
        aboo_dim = int(self.nocc_act[0] * self.nocc_act[1])
        # (beta, beta) subspace
        bbvv_dim = int(self.nvir_act[1] * (self.nvir_act[1] + 1) / 2)
        bboo_dim = int(self.nocc_act[1] * (self.nocc_act[1] + 1) / 2)

        log.info('\n******** %s ********' % self.__class__)
        log.info('nmo = %d (%d alpha, %d beta)',
                 self.nmo[0]+self.nmo[1], self.nmo[0], self.nmo[1])
        log.info('nocc = %d (%d alpha, %d beta), nvir = %d (%d alpha, %d beta)',
                 self.nocc[0] + self.nocc[1], self.nocc[0], self.nocc[1],
                 self.nvir[0] + self.nvir[1], self.nvir[0], self.nvir[1])
        log.info('nocc_act = %d (%d alpha, %d beta)',
                 self.nocc_act[0] + self.nocc_act[1],
                 self.nocc_act[0], self.nocc_act[1])
        log.info('nvir_act = %d (%d alpha, %d beta)',
                 self.nvir_act[0] + self.nvir_act[1],
                 self.nvir_act[0], self.nvir_act[1])
        log.info('for (alpha alpha, alpha alpha) subspace:')
        log.info('  occ-occ dimension = %d vir-vir dimension = %d',
                 aaoo_dim, aavv_dim)
        log.info('for (beta beta, beta beta) subspace:')
        log.info('  occ-occ dimension = %d vir-vir dimension = %d',
                 bboo_dim, bbvv_dim)
        log.info('for (alpha beta, alpha beta) subspace:')
        log.info('  occ-occ dimension = %d vir-vir dimension = %d',
                 aboo_dim, abvv_dim)
        log.info('interested hh state = %d', self.hh_state)
        log.info('interested pp state = %d', self.pp_state)
        log.info('ground state = %s', self.nelec)
        log.info('print threshold = %.2f%%', self.print_thresh*100)
        log.info('')
        return

    def check_memory(self):
        log = logger.Logger(self.stdout, self.verbose)
        # ====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(self.nvir_act[0] * (self.nvir_act[0] + 1) / 2)
        aaoo_dim = int(self.nocc_act[0] * (self.nocc_act[0] + 1) / 2)
        aafull_dim = aavv_dim + aaoo_dim
        # (alpha, beta) subspace
        abvv_dim = int(self.nvir_act[0] * self.nvir_act[1])
        aboo_dim = int(self.nocc_act[0] * self.nocc_act[1])
        abfull_dim = abvv_dim + aboo_dim
        # (beta, beta) subspace
        bbvv_dim = int(self.nvir_act[1] * (self.nvir_act[1] + 1) / 2)
        bboo_dim = int(self.nocc_act[1] * (self.nocc_act[1] + 1) / 2)
        bbfull_dim = bbvv_dim + bboo_dim

        full_dim = max(aafull_dim, abfull_dim, bbfull_dim)

        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if mem < 1000:
            log.info("U-ppRPA needs at least %d MB memory." % mem)
        else:
            log.info("U-ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self, subspace=['aa', 'bb', 'ab']):
        """Run ppRPA direct diagonalization.

        Kwargs:
            subspace (list of string): subspace(s) to run diagonalization,
                'aa' for (alpha, alpha, alpha, alpha),
                'bb' for (beta, beta, beta, beta),
                and 'ab' for (alpha, beta, alpha, beta).
        """
        self.subspace = subspace
        self.check_parameter()
        self.check_memory()

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        if self.Lpq is None:
            self.Lpq = self.ao2mo()
        if 'aa' in subspace:
            aa_exci, aa_xy, aa_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc_act[0], self.mo_energy_act[0],
                self.Lpq[0], mu=self.mu)
        else:
            aa_exci = aa_xy = aa_ec = None

        if 'bb' in subspace:
            bb_exci, bb_xy, bb_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc_act[1], self.mo_energy_act[1],
                self.Lpq[1], mu=self.mu)
        else:
            bb_exci = bb_xy = bb_ec = None

        if 'ab' in subspace:
            ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
                self.nocc_act, self.mo_energy_act, self.Lpq, mu=self.mu)
        else:
            ab_exci = ab_xy = ab_ec = None
        logger.timer(self, "ppRPA direct: %s" % subspace, *cput0)

        self.ec = [aa_ec, bb_ec, ab_ec]
        self.exci = [aa_exci, bb_exci, ab_exci]
        self.xy = [aa_xy, bb_xy, ab_xy]

        return

    def get_correlation(self):
        """Get ppRPA correlation energy.

        Returns:
            ec (double): ppRPA correlation energy.
        """
        self.check_parameter()

        if self.Lpq is None:
            self.Lpq = self.ao2mo()
        if self.ec[0] is None:
            cput0 = (logger.process_clock(), logger.perf_counter())
            self.exci[0], self.xy[0], self.ec[0] = \
                diagonalize_pprpa_subspace_same_spin(
                self.nocc[0], self.mo_energy[0], self.Lpq[0], mu=self.mu)
            logger.timer(
                self, "ppRPA correlation energy: (alpha alpha, alpha alpha)",
                *cput0)
        if self.ec[1] is None:
            cput0 = (logger.process_clock(), logger.perf_counter())
            self.exci[1], self.xy[1], self.ec[1] = \
                diagonalize_pprpa_subspace_same_spin(
                self.nocc[1], self.mo_energy[1], self.Lpq[1], mu=self.mu)
            logger.timer(
                self, "ppRPA correlation energy: (beta beta, beta beta)",
                *cput0)
        if self.ec[2] is None:
            cput0 = (logger.process_clock(), logger.perf_counter())
            self.exci[2], self.xy[2], self.ec[2] = \
                diagonalize_pprpa_subspace_diff_spin(
                self.nocc, self.mo_energy, self.Lpq, mu=self.mu)
            logger.timer(
                self, "ppRPA correlation energy: (alpha beta, alpha beta)",
                *cput0)

        return self.ec[0] + self.ec[1] + self.ec[2]

    def energy_tot(self):
        """Get ppRPA total energy.
        Totoal energy = Hartree-Fock energy + correlation energy from ppRPA.

        Returns:
            e_tot (double); ppRPA total energy.
        """
        mf = self._scf
        assert mf.converged
        hf_obj = mf if not isinstance(mf, KohnShamDFT) else mf.to_hf()

        dm = hf_obj.make_rdm1()
        e_hf = hf_obj.energy_nuc() + hf_obj.energy_elec(dm=dm)[0]
        ec = self.get_correlation()
        e_tot = e_hf + ec

        return e_tot, e_hf, ec
