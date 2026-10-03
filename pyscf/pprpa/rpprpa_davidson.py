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

from pyscf.lib import einsum, logger, StreamObject
from pyscf.mp.mp2 import get_nocc, get_nmo

from pyscf.pprpa.rpprpa_direct import ao2mo, inner_product, ij2index, \
    get_chemical_potential, pprpa_orthonormalize_eigenvector, pprpa_print_a_pair


def kernel(pprpa):
    # initialize trial vector and product matrix
    tri_vec = numpy.zeros(shape=[pprpa.max_vec, pprpa.full_dim], dtype=numpy.double)
    tri_vec_sig = numpy.zeros(shape=[pprpa.max_vec], dtype=numpy.double)
    if pprpa.channel == "pp":
        ntri = min(pprpa.nroot * 4, pprpa.vv_dim)
    else:
        ntri = min(pprpa.nroot * 4, pprpa.oo_dim)
    tri_vec[:ntri], tri_vec_sig[:ntri] = pprpa_get_trial_vector(pprpa, ntri)
    mv_prod = numpy.zeros_like(tri_vec)

    iter = 0
    nprod = 0  # number of contracted vectors
    while iter < pprpa.max_iter:
        logger.info(
            pprpa, "\nppRPA Davidson %d-th iteration, ntri= %d , nprod= %d .",
            iter + 1, ntri, nprod)
        mv_prod[nprod:ntri] = pprpa_contraction(pprpa, tri_vec[nprod:ntri])
        nprod = ntri

        # get ppRPA matrix and metric matrix in subspace
        m_tilde = numpy.dot(tri_vec[:ntri], mv_prod[:ntri].T)
        w_tilde = numpy.zeros_like(m_tilde)
        for i in range(ntri):
            if inner_product(tri_vec[i], tri_vec[i], pprpa.oo_dim) > 0:
                w_tilde[i, i] = 1
            else:
                w_tilde[i, i] = -1

        # diagonalize subspace matrix
        alphar, _, beta, _, v_tri, _, _ = scipy.linalg.lapack.dggev(
            m_tilde, w_tilde, compute_vl=0)
        e_tri = alphar / beta
        v_tri = v_tri.T  # Fortran matrix to Python order

        if pprpa.channel == "pp":
            # sort eigenvalues and eigenvectors by ascending order
            idx = e_tri.argsort()
            e_tri = e_tri[idx]
            v_tri = v_tri[idx, :]

            # re-order all states by signs, first hh then pp
            sig = numpy.zeros(shape=[ntri], dtype=int)
            for i in range(ntri):
                if numpy.sum((v_tri[i] ** 2) * tri_vec_sig[: ntri]) > 0:
                    sig[i] = 1
                else:
                    sig[i] = -1

            hh_index = numpy.where(sig < 0)[0]
            pp_index = numpy.where(sig > 0)[0]
            e_tri_hh = e_tri[hh_index]
            e_tri_pp = e_tri[pp_index]
            e_tri[:len(hh_index)] = e_tri_hh
            e_tri[len(hh_index):] = e_tri_pp
            v_tri_hh = v_tri[hh_index]
            v_tri_pp = v_tri[pp_index]
            v_tri[:len(hh_index)] = v_tri_hh
            v_tri[len(hh_index):] = v_tri_pp

            # get only two-electron addition energy
            first_state=len(hh_index)
            pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]
        else:
            # sort eigenvalues and eigenvectors by descending order
            idx = e_tri.argsort()[::-1]
            e_tri = e_tri[idx]
            v_tri = v_tri[idx, :]

            # re-order all states by signs, first pp then hh
            sig = numpy.zeros(shape=[ntri], dtype=int)
            for i in range(ntri):
                if numpy.sum((v_tri[i] ** 2) * tri_vec_sig[:ntri]) > 0:
                    sig[i] = 1
                else:
                    sig[i] = -1

            hh_index = numpy.where(sig < 0)[0]
            pp_index = numpy.where(sig > 0)[0]
            e_tri_hh = e_tri[hh_index]
            e_tri_pp = e_tri[pp_index]
            e_tri[:len(pp_index)] = e_tri_pp
            e_tri[len(pp_index):] = e_tri_hh
            v_tri_hh = v_tri[hh_index]
            v_tri_pp = v_tri[pp_index]
            v_tri[:len(pp_index)] = v_tri_pp
            v_tri[len(pp_index):] = v_tri_hh

            # get only two-electron removal energy
            first_state=len(pp_index)
            pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]

        ntri_old = ntri
        conv, ntri = pprpa_expand_space(
            pprpa=pprpa, first_state=first_state, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
        logger.info(pprpa, "add %d new trial vectors.", ntri - ntri_old)

        iter += 1
        if conv is True:
            break

    assert conv is True, "ppRPA Davidson is not converged!"
    logger.info(
        pprpa, "\nppRPA Davidson converged in %d iterations, final subspace size = %d",
        iter, nprod)

    pprpa_orthonormalize_eigenvector(pprpa.multi, pprpa.nocc_act, pprpa.exci, pprpa.xy)

    return


# Davidson algorithm functions
def pprpa_get_trial_vector(pprpa, ntri):
    """Generate initial trial vectors in particle-particle or hole-hole channel.
    The order is determined by the pair orbital energy summation.
    The initial trial vectors are diagonal, and signatures are all 1 or -1.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        ntri (int): the number of initial trial vectors.

    Returns:
        tri_vec (double ndarray): initial trial vectors.
        tri_vec_sig (double ndarray): signature of initial trial vectors.
    """
    # for convenience, use XX as XX_act in this function
    nocc, nvir, nmo = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act
    mo_energy = pprpa.mo_energy_act

    is_singlet = 1 if pprpa.multi == "s" else 0

    max_orb_sum = 1.0e15

    class pair():
        def __init__(self):
            self.p = -1
            self.q = -1
            self.eig_sum = max_orb_sum if pprpa.channel == "pp" else -max_orb_sum

    pairs = []
    for r in range(ntri):
        t = pair()
        pairs.append(t)

    if pprpa.channel == "pp":
        # find particle-particle pairs with lowest orbital energy summation
        for r in range(ntri):
            for p in range(nocc, nmo):
                for q in range(nocc, p + is_singlet):
                    valid = True
                    for rr in range(r):
                        if pairs[rr].p == p and pairs[rr].q == q:
                            valid = False
                            break
                    if (valid is True
                        and (mo_energy[p] + mo_energy[q]) < pairs[r].eig_sum):
                        pairs[r].p, pairs[r].q = p, q
                        pairs[r].eig_sum = mo_energy[p] + mo_energy[q]

        # sort pairs by ascending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum > pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = \
                        pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = \
                        pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = \
                        p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, \
            "cannot find enough pairs for trial vectors"

        tri_vec = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)
        tri_vec_sig = numpy.zeros(shape=[ntri], dtype=numpy.double)
        tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)
        for r in range(ntri):
            p, q = pairs[r].p, pairs[r].q
            pq = ij2index(p - nocc, q - nocc, tri_row_v, tri_col_v)
            tri_vec[r, pprpa.oo_dim + pq] = 1.0
            tri_vec_sig[r] = 1.0
    else:
        # find hole-hole pairs with highest orbital energy summation
        for r in range(ntri):
            for p in range(nocc-1, -1, -1):
                for q in range(nocc-1, p - is_singlet, -1):
                    valid = True
                    for rr in range(r):
                        if pairs[rr].p == q and pairs[rr].q == p:
                            valid = False
                            break
                    if (valid is True
                        and (mo_energy[p] + mo_energy[q]) > pairs[r].eig_sum):
                        pairs[r].p, pairs[r].q = q, p
                        pairs[r].eig_sum = mo_energy[p] + mo_energy[q]

        # sort pairs by descending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum < pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = \
                        pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = \
                        pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = \
                        p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, \
            "cannot find enough pairs for trial vectors"

        tri_vec = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)
        tri_vec_sig = numpy.zeros(shape=[ntri], dtype=numpy.double)
        tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
        for r in range(ntri):
            p, q = pairs[r].p, pairs[r].q
            pq = ij2index(p, q, tri_row_o, tri_col_o)
            tri_vec[r, pq] = 1.0
            tri_vec_sig[r] = -1.0

    return tri_vec, tri_vec_sig


def pprpa_contraction(pprpa, tri_vec):
    """ppRPA contraction.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        tri_vec (double ndarray): trial vector.

    Returns:
        mv_prod (double ndarray): product between ppRPA matrix and trial vectors.
    """
    # for convenience, use XX as XX_act in this function
    nocc, nvir, nmo = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act
    mo_energy = pprpa.mo_energy_act
    Lpi = pprpa.Lpi
    Lpa = pprpa.Lpa
    naux = Lpi.shape[0]

    ntri = tri_vec.shape[0]
    mv_prod = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)

    is_singlet = 1 if pprpa.multi == "s" else 0
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    z_oo = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
    z_vv = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
    for ivec in range(ntri):
        z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][:pprpa.oo_dim]
        z_oo[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)
        z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][pprpa.oo_dim:]
        z_vv[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)

        # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
        Lpq_z = numpy.zeros(shape=[naux * nmo, nmo], dtype=numpy.double)
        Lpq_z[:, :nocc] = Lpi.reshape(naux * nmo, nocc) @ z_oo.T
        Lpq_z[:, nocc:] = Lpa.reshape(naux * nmo, nvir) @ z_vv.T

        # transpose and reshape for faster multiplication
        Lpq_z = Lpq_z.reshape(naux, nmo, nmo).transpose(1, 0, 2)
        Lpq_z = Lpq_z.reshape(nmo, naux * nmo)

        # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} \pm Lpq_{L,qr} Lpqz_{L,pr}
        # NOTE: here assuming Lpq[L,p,q] = Lpq[L,q,p] for real orbitals
        mv_prod_full = numpy.zeros(shape=[nmo, nmo], dtype=numpy.double)
        mv_prod_full[:nocc, :nocc] = Lpq_z[:nocc] @ Lpi.reshape(naux * nmo, nocc)
        mv_prod_full[nocc:, nocc:] = Lpq_z[nocc:] @ Lpa.reshape(naux * nmo, nvir)
        if pprpa.multi == "s":
            mv_prod_full += mv_prod_full.T
        else:
            mv_prod_full -= mv_prod_full.T
        mv_prod_full = mv_prod_full.T
        mv_prod_full[numpy.diag_indices(nmo)] *= 1.0 / numpy.sqrt(2)
        mv_prod[ivec][: pprpa.oo_dim] =\
            mv_prod_full[: nocc, : nocc][tri_row_o, tri_col_o]
        mv_prod[ivec][pprpa.oo_dim:] = \
            mv_prod_full[nocc:, nocc:][tri_row_v, tri_col_v]

    orb_sum_oo = (mo_energy[None, : nocc] + mo_energy[: nocc, None]) - 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = (mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
    for ivec in range(ntri):
        oz_oo = -orb_sum_oo * tri_vec[ivec][:pprpa.oo_dim]
        mv_prod[ivec][:pprpa.oo_dim] += oz_oo
        oz_vv = orb_sum_vv * tri_vec[ivec][pprpa.oo_dim:]
        mv_prod[ivec][pprpa.oo_dim:] += oz_vv

    return mv_prod


def pprpa_expand_space(
        pprpa, first_state, tri_vec, tri_vec_sig, mv_prod, v_tri):
    """Expand trial vector space in Davidson algorithm.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        first_state (int): index of first particle-particle or hole-hole state.
        tri_vec (double ndarray): trial vector.
        tri_vec_sig (int array): signature of trial vector.
        mv_prod (double ndarray): product matrix of ppRPA matrix and trial vector.
        v_tri (double ndarray): eigenvector of subspace matrix.

    Returns:
        conv (bool): if Davidson algorithm is converged.
        ntri (int): updated number of trial vectors.
    """
    nocc_act, nvir_act = pprpa.nocc_act, pprpa.nvir_act
    mo_energy_act = pprpa.mo_energy_act
    nroot = pprpa.nroot
    exci = pprpa.exci
    max_vec = pprpa.max_vec
    residue_thresh = pprpa.residue_thresh

    is_singlet = 1 if pprpa.multi == "s" else 0

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, is_singlet-1)

    # take only nRoot vectors, starting from first pp channel
    tmp = v_tri[first_state:(first_state+nroot)]

    # get the eigenvectors in the original space
    ntri = v_tri.shape[0]
    pprpa.xy = numpy.dot(tmp, tri_vec[:ntri])

    # compute residue vectors
    residue = numpy.dot(tmp, mv_prod[:ntri])
    for i in range(nroot):
        residue[i][:pprpa.oo_dim] -= -exci[i] * pprpa.xy[i][:pprpa.oo_dim]
        residue[i][pprpa.oo_dim:] += -exci[i] * pprpa.xy[i][pprpa.oo_dim:]

    # check convergence
    conv_record = numpy.zeros(shape=[nroot], dtype=bool)
    max_residue = 0
    for i in range(nroot):
        max_residue = max(max_residue, abs(numpy.max(residue[i])))
        if len(residue[i][abs(residue[i]) > residue_thresh]) == 0:
            conv_record[i] = True
        else:
            conv_record[i] = False
    nconv = len(conv_record[conv_record is True])
    logger.info(pprpa, "max residue = %.6e", max_residue)
    if nconv == nroot:
        return True, ntri

    orb_sum_oo = mo_energy_act[None, :nocc_act] + mo_energy_act[:nocc_act, None]
    orb_sum_oo -= 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = mo_energy_act[None, nocc_act:] + mo_energy_act[nocc_act:, None]
    orb_sum_vv -= 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]

    # Schmidt orthogonalization
    ntri_old = ntri
    for iroot in range(nroot):
        if conv_record[iroot] is True:
            continue

        # convert residuals
        residue[iroot][:pprpa.oo_dim] /= -(exci[iroot] - orb_sum_oo)
        residue[iroot][pprpa.oo_dim:] /= (exci[iroot] - orb_sum_vv)

        for ivec in range(ntri):
            # compute product between new vector and old vector
            inp = -inner_product(residue[iroot], tri_vec[ivec], pprpa.oo_dim)
            # eliminate parallel part
            if tri_vec_sig[ivec] < 0:
                inp = -inp
            residue[iroot] += inp * tri_vec[ivec]

        # add a new trial vector
        if len(residue[iroot][abs(residue[iroot]) > residue_thresh]) > 0:
            assert ntri < max_vec, (
                "ppRPA Davidson expansion failed! ntri %d exceeds max_vec %d!" %
                (ntri, max_vec))
            inp = inner_product(residue[iroot], residue[iroot], pprpa.oo_dim)
            tri_vec_sig[ntri] = 1 if inp > 0 else -1
            tri_vec[ntri] = residue[iroot] / numpy.sqrt(abs(inp))
            ntri = ntri + 1

    conv = True if ntri_old == ntri else False
    return conv, ntri


# analysis functions
def pprpa_davidson_print_eigenvector(pprpa, multi, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        pprpa (RppRPADavidson): ppRPA object.
        multi (string): multiplicity.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nocc = pprpa.nocc
    nocc_act, nvir_act = pprpa.nocc_act, pprpa.nvir_act
    if multi == "s":
        oo_dim = int((nocc_act + 1) * nocc_act / 2)
        is_singlet = 1
        logger.info(pprpa, "\n     print ppRPA excitations: singlet\n")
    elif multi == "t":
        oo_dim = int((nocc_act - 1) * nocc_act / 2)
        is_singlet = 0
        logger.info(pprpa, "\n     print ppRPA excitations: triplet\n")

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, is_singlet-1)

    nroot = len(exci)
    au2ev = 27.211386
    if pprpa.channel == "pp":
        for iroot in range(nroot):
            logger.info(pprpa, "#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
                  iroot + 1, multi, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev)
            if nocc_act > 0:
                full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=numpy.double)
                full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
                full = numpy.power(full, 2)
                pairs = numpy.argwhere(full > pprpa.print_thresh)
                for i, j in pairs:
                    pprpa_print_a_pair(
                        pprpa, is_pp=False, p=i, q=j, percentage=full[i, j])

            full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > pprpa.print_thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            logger.info(pprpa, "")
    else:
        for iroot in range(nroot):
            logger.info(pprpa, "#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
                        iroot + 1, multi, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev)
            full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > pprpa.print_thresh)
            for i, j in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=False, p=i, q=j, percentage=full[i, j])

            if nvir_act > 0:
                full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=numpy.double)
                full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
                full = numpy.power(full, 2)
                pairs = numpy.argwhere(full > pprpa.print_thresh)
                for a, b in pairs:
                    pprpa_print_a_pair(
                        pprpa, is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            logger.info(pprpa, "")

    return


def analyze_pprpa_davidson(pprpa):
    """Analyze ppRPA (Davidson algorithm) excited states.

    Args:
        pprpa (RppRPADavidson): ppRPA object.
    """
    logger.info(pprpa, "\nanalyze ppRPA results.")

    if pprpa.exci_s is not None and pprpa.exci_t is not None:
        logger.info(pprpa, "both singlet and triplet results found.")
        if pprpa.channel == "pp":
            exci0 = min(pprpa.exci_s[0], pprpa.exci_t[0])
        else:
            exci0 = max(pprpa.exci_s[0], pprpa.exci_t[0])
        pprpa_davidson_print_eigenvector(
            pprpa, multi="s", exci0=exci0, exci=pprpa.exci_s, xy=pprpa.xy_s)
        pprpa_davidson_print_eigenvector(
            pprpa, multi="t", exci0=exci0, exci=pprpa.exci_t, xy=pprpa.xy_t)
    else:
        if pprpa.exci_s is not None:
            logger.info(pprpa, "only singlet results found.")
            pprpa_davidson_print_eigenvector(
                pprpa, multi="s", exci0=pprpa.exci_s[0], exci=pprpa.exci_s, xy=pprpa.xy_s)
        else:
            logger.info(pprpa, "only triplet results found.")
            pprpa_davidson_print_eigenvector(
                pprpa, multi="t", exci0=pprpa.exci_t[0], exci=pprpa.exci_t, xy=pprpa.xy_t)
    return


class RppRPADavidson(StreamObject):
    def __init__(
            self, mf, nocc_act=None, nvir_act=None, channel="pp", nroot=5,
            max_vec=200, max_iter=100, residue_thresh=1.0e-7, print_thresh=0.1,
            auxbasis=None):
        # necessary input
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        # options
        self.nocc_act = nocc_act  # number of active occupied orbitals
        self.nvir_act = nvir_act  # number of active virtual orbitals
        self.channel = channel  # channel of desired states, particle-particle or hole-hole
        self.nroot = nroot  # number of desired roots
        self.max_vec = max_vec  # max size of trial vectors
        self.max_iter = max_iter  # max iteration
        self.residue_thresh = residue_thresh  # residue threshold
        self.print_thresh = print_thresh  # threshold to print component
        self.auxbasis = auxbasis  # auxiliary basis set

        # internal flags
        self.multi = None  # multiplicity
        self.is_singlet = None  # multiplicity is singlet
        self.mu = None  # chemical potential
        self.nmo_act = None  # number of active orbitals
        self.mo_energy_act = None  # orbital energy in active space
        self.oo_dim = None  # particle-particle block dimension
        self.vv_dim = None  # hole-hole block dimension
        self.full_dim = None  # full matrix dimension

        # results
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
        self.Lpi = None  # three-center density-fitting matrix in MO, [naux, nmo, nocc]
        self.Lpa = None  # three-center density-fitting matrix in MO, [naux, nmo, nvir]
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

    analyze = analyze_pprpa_davidson
    ao2mo = ao2mo

    def check_parameter(self):
        """Initialize and check options.
        """
        assert self.channel in ["pp", "hh"]
        assert self.multi in ["s", "t"]

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

        if self.multi == "s":
            self.oo_dim = int((self.nocc_act + 1) * self.nocc_act / 2)
            self.vv_dim = int((self.nvir_act + 1) * self.nvir_act / 2)
        elif self.multi == "t":
            self.oo_dim = int((self.nocc_act - 1) * self.nocc_act / 2)
            self.vv_dim = int((self.nvir_act - 1) * self.nvir_act / 2)
        self.full_dim = self.oo_dim + self.vv_dim

        self.max_vec = min(self.max_vec, self.full_dim)

        assert self.residue_thresh > 0
        assert 0.0 < self.print_thresh < 1.0

        if self.mu is None:
            self.mu = get_chemical_potential(self.nocc, self.mo_energy)

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('\n******** %s ********' % self.__class__)
        multiplicity = "singlet" if self.multi == "s" else "triplet"
        log.info('multiplicity = %s', multiplicity)
        log.info('state channel = %s' % self.channel)
        log.info('nmo = %d' % self.nmo)
        log.info('nocc = %d nvir = %d', self.nocc, self.nvir)
        log.info('nocc_act = %d nvir_act = %d', self.nocc_act, self.nvir_act)
        log.info('occ-occ dimension = %d vir-vir dimension = %d', self.oo_dim, self.vv_dim)
        log.info('full dimension = %d', self.full_dim)
        log.info('number of roots = %d', self.nroot)
        log.info('max subspace size = %d', self.max_vec)
        log.info('max iteration = %d', self.max_iter)
        log.info('residue threshold = %.3e', self.residue_thresh)
        log.info('print threshold = %.2f%%', self.print_thresh*100)
        log.info('')
        return

    def check_memory(self):
        """Check required memory.
        """
        # intermediate in contraction; mv_prod, tri_vec, xy
        mem = (3 * self.max_vec * self.full_dim) * 8 / 1.0e6
        if mem < 1000:
            logger.info(self, "ppRPA needs at least %d MB memory.", mem)
        else:
            logger.info(self, "ppRPA needs at least %.1f GB memory.", mem/1.0e3)
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
        if self.Lpi is None or self.Lpa is None:
            Lpq = self.ao2mo()
            self.Lpi = numpy.ascontiguousarray(Lpq[:, :, :self.nocc_act])
            self.Lpa = numpy.ascontiguousarray(Lpq[:, :, self.nocc_act:])
            logger.timer(self, "ppRPA integral transformation: %s" % multi, *cput0)
        self.dump_flags()
        kernel(pprpa=self)
        logger.timer(self, "ppRPA Davidson: %s" % multi, *cput0)

        if self.multi == "s":
            self.exci_s = self.exci.copy()
            self.xy_s = self.xy.copy()
        else:
            self.exci_t = self.exci.copy()
            self.xy_t = self.xy.copy()
        self.exci = self.xy = None
        return

