#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         James McClain <jdmcclain47@gmail.com>
#         Xing Zhang <zhangxing.nju@gmail.com>
#


'''
kpoint-adapted and spin-adapted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab

t2 and eris are never stored in full, only a partial
eri of size (nkpts,nocc,nocc,nvir,nvir)
'''

import numpy as np
from scipy.linalg import block_diag
import h5py

from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.mp import mp2
from pyscf.pbc.df import df
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib import kpts as libkpts
from pyscf.lib.parameters import LARGE_DENOM
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    """Computes k-point RMP2 energy.

    Args:
        mp (KMP2): an instance of KMP2
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to WITH_T2 (=True).

    Returns:
        KMP2 energy and t2 amplitudes (=None if with_t2 is False)
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mp.dump_flags()
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)

    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = mp._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6
    if with_t2:
        mem_usage += (nkpts**3 * (nocc * nvir)**2) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    # Build 3-index DF tensor Lov
    if with_df_ints:
        Lov = _init_mp_df_eris(mp)

    emp2_ss = emp2_os = 0.
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                # (ia|jb)
                if with_df_ints:
                    oovv_ij[ka] = (1./nkpts) * einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
                else:
                    orbo_i = mo_coeff[ki][:,:nocc]
                    orbo_j = mo_coeff[kj][:,:nocc]
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                                         (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]),
                                         compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]

                # Remove zero/padded elements from denominator
                eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                t2_ijab = np.conj(oovv_ij[ka]/eijab)
                if with_t2:
                    t2[ki, kj, ka] = t2_ijab
                edi = einsum('ijab,ijab', t2_ijab, oovv_ij[ka]).real * 2
                exi = -einsum('ijab,ijba', t2_ijab, oovv_ij[kb]).real
                emp2_ss += edi*0.5 + exi
                emp2_os += edi*0.5

    log.timer("KMP2", *cput0)

    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


def _init_mp_df_eris(mp):
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2) -- A KMP2 instance

    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts, nkpts, naux, nocc, nvir)
    """
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(mp.stdout, mp.verbose)

    if mp._scf.with_df._cderi is None:
        mp._scf.with_df.build()

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    mo_coeff = _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    kpts = mp.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Lov = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())

    bra_start = 0
    bra_end = nocc
    ket_start = nmo+nocc
    ket_end = ket_start + nvir
    with df.CDERIArray(mp._scf.with_df._cderi) as cderi_array:
        tao = []
        ao_loc = None
        for ki in range(nkpts):
            for kj in range(nkpts):
                Lpq_ao = cderi_array[ki,kj]

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
                Lov[ki, kj] = out.reshape(-1, nocc, nvir)

    log.timer_debug1("transforming DF-MP2 integrals", *cput0)

    return Lov


def _padding_k_idx(nmo, nocc, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.
    Args:
        nmo (Iterable): k-dependent orbital number;
        nocc (Iterable): k-dependent occupation numbers;
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    if kind not in ("split", "joint"):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == "split":
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    nocc = np.array(nocc)
    nmo = np.array(nmo)
    nvirt = nmo - nocc
    dense_o = np.amax(nocc)
    dense_v = np.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == "split":
            indexes_o.append(np.arange(k_o))
            indexes_v.append(np.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(np.concatenate((
                np.arange(k_o),
                np.arange(dense_nmo - k_v, dense_nmo),
            )))

    if kind == "split":
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind="split"):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.

    This implementation stores k-dependent Fock and other matrix in dense arrays with additional dimensions
    corresponding to k-point indexes. In case when the occupation numbers depend on the k-point index (i.e. a metal) or
    when some k-points have more Bloch basis functions than others the corresponding data structure has to be padded
    with entries that are not used (fictitious occupied and virtual degrees of freedom). Current convention stores these
    states at the Fermi level as shown in the following example.

    +----+--------+--------+--------+
    |    |  k=0   |  k=1   |  k=2   |
    |    +--------+--------+--------+
    |    | nocc=2 | nocc=3 | nocc=2 |
    |    | nvir=4 | nvir=3 | nvir=3 |
    +====+========+========+========+
    | v3 |  k0v3  |  k1v2  |  k2v2  |
    +----+--------+--------+--------+
    | v2 |  k0v2  |  k1v1  |  k2v1  |
    +----+--------+--------+--------+
    | v1 |  k0v1  |  k1v0  |  k2v0  |
    +----+--------+--------+--------+
    | v0 |  k0v0  |        |        |
    +====+========+========+========+
    |          Fermi level          |
    +====+========+========+========+
    | o2 |        |  k1o2  |        |
    +----+--------+--------+--------+
    | o1 |  k0o1  |  k1o1  |  k2o1  |
    +----+--------+--------+--------+
    | o0 |  k0o0  |  k1o0  |  k2o0  |
    +----+--------+--------+--------+

    In the above example, `get_nmo(mp, per_kpoint=True) == (6, 6, 5)`, `get_nocc(mp, per_kpoint) == (2, 3, 2)`. The
    resulting dense `get_nmo(mp) == 7` and `get_nocc(mp) == 3` correspond to padded dimensions. This function will
    return the following indexes corresponding to the filled entries of the above table:

    >>> padding_k_idx(mp, kind="split")
    ([(0, 1), (0, 1, 2), (0, 1)], [(0, 1, 2, 3), (1, 2, 3), (1, 2, 3)])

    >>> padding_k_idx(mp, kind="joint")
    [(0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 4, 5, 6)]

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    return _padding_k_idx(mp.get_nmo(per_kpoint=True), mp.get_nocc(per_kpoint=True), kind=kind)


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = np.zeros((nkpts, mp.nmo), dtype=mo_energy[0].dtype)
    for k in range(nkpts):
        result[np.ix_([k], padding_convention[k])] = mo_energy[k][frozen_mask[k]]

    return result


def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = np.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[np.ix_([k], np.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = np.array(frozen)
    nocc = np.count_nonzero(mo_occ > 0)

    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(np.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and np.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))


def get_nocc(mp, per_kpoint=False):
    '''Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    for i, moocc in enumerate(mp.mo_occ):
        if np.any(moocc % 1 != 0):
            raise RuntimeError("Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by "
                               "smearing of occupation numbers in the mean-field calculation. If so, consider "
                               "executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling "
                               "this".format(i, moocc))
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = [np.count_nonzero(mp.mo_occ[ikpt]) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (int, np.integer)):
        nocc = [(np.count_nonzero(mp.mo_occ[ikpt]) - mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, np.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nocc = []
        for ikpt in range(mp.nkpts):
            max_occ_idx = np.max(np.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = np.sum(np.array(mp.frozen) <= max_occ_idx)
            nocc.append(np.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    elif isinstance(mp.frozen[0], (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = np.max(np.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = np.sum(np.array(frozen) <= max_occ_idx)
            nocc.append(np.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(np.array(nocc) > 0), ('Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' %
           (nocc, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        nocc = np.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    '''Number of orbitals for k-point calculations.

    Number of orbitals for use in a calculation with k-points, taking into account
    frozen orbitals.

    Note:
        If `per_kpoint` is False, then the number of orbitals here is equal to max(nocc) + max(nvir),
        where each max is done over all k-points.  Otherwise the number of orbitals is returned
        as a list of number of orbitals at each k-point.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of orbitals at each k-point.
            For a description of False, see Note.

    Returns:
        nmo (int, list of int): Number of orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    if mp._nmo is not None:
        return mp._nmo

    if mp.frozen is None:
        nmo = [len(mp.mo_occ[ikpt]) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (int, np.integer)):
        nmo = [len(mp.mo_occ[ikpt]) - mp.frozen for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, np.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(np.array(nmo) > 0), ('Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' %
           (nmo, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = np.max(nocc) + np.max(np.array(nmo) - np.array(nocc))

    return nmo


def get_frozen_mask(mp):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `bool`): Boolean mask of orbitals to include.

    '''
    moidx = [np.ones(x.size, dtype=bool) for x in mp.mo_occ]
    if mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, np.integer)):
        for idx in moidx:
            idx[:mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, np.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, np.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx


def _add_padding(mp, mo_coeff, mo_energy):
    nmo = mp.nmo

    # Check if these are padded mo coefficients and energies and/or if some orbitals are frozen.
    if (mp.frozen is not None) or (not np.all([x.shape[1] == nmo for x in mo_coeff])):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if (mp.frozen is not None) or (not np.all([x.shape[0] == nmo for x in mo_energy])):
        mo_energy = padded_mo_energy(mp, mo_energy)
    return mo_coeff, mo_energy


def make_rdm1(mp, t2=None, kind="compact"):
    r"""
    Spin-traced one-particle density matrix in the MO basis representation.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Args:
        mp (KMP2): a KMP2 kernel object;
        t2 (ndarray): a t2 MP2 tensor;
        kind (str): either 'compact' or 'padded' - defines behavior for k-dependent MO basis sizes;

    Returns:
        A k-dependent single-particle density matrix.
    """
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    d_imds = _gamma1_intermediates(mp, t2=t2)
    result = []
    padding_idxs = padding_k_idx(mp, kind="joint")
    for (oo, vv), idxs in zip(zip(*d_imds), padding_idxs):
        oo += np.eye(*oo.shape)
        d = block_diag(oo, vv)
        d += d.conj().T
        if kind == "padded":
            result.append(d)
        else:
            result.append(d[np.ix_(idxs, idxs)])
    return result


def make_rdm2(mp, t2=None, kind="compact"):
    r'''
    Spin-traced two-particle density matrix in MO basis

    .. math::

        dm2[p,q,r,s] = \sum_{\sigma,\tau} <p_\sigma^\dagger r_\tau^\dagger s_\tau q_\sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if kind not in ("compact", "padded"):
        raise ValueError("The 'kind' argument should be either 'compact' or 'padded'")
    if t2 is None: t2 = mp.t2
    dm1 = mp.make_rdm1(t2, "padded")
    nmo = mp.nmo
    nocc = mp.nocc
    nkpts = mp.nkpts
    dtype = t2.dtype

    dm2 = np.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo),dtype=dtype)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                dovov = t2[ki, kj, ka].transpose(0,2,1,3) * 2 - t2[kj, ki, ka].transpose(1,2,0,3)
                dovov *= 2
                dm2[ki,ka,kj,:nocc,nocc:,:nocc,nocc:] = dovov
                dm2[ka,ki,kb,nocc:,:nocc,nocc:,:nocc] = dovov.transpose(1,0,3,2).conj()

    occidx = padding_k_idx(mp, kind="split")[0]
    for ki in range(nkpts):
        for i in occidx[ki]:
            dm1[ki][i,i] -= 2

    for ki in range(nkpts):
        for kp in range(nkpts):
            for i in occidx[ki]:
                dm2[ki,ki,kp,i,i,:,:] += dm1[kp].T * 2
                dm2[kp,kp,ki,:,:,i,i] += dm1[kp].T * 2
                dm2[kp,ki,ki,:,i,i,:] -= dm1[kp].T
                dm2[ki,kp,kp,i,:,:,i] -= dm1[kp]

    for ki in range(nkpts):
        for kj in range(nkpts):
            for i in occidx[ki]:
                for j in occidx[kj]:
                    dm2[ki,ki,kj,i,i,j,j] += 4
                    dm2[ki,kj,kj,i,j,j,i] -= 2

    if kind == "padded":
        return dm2
    else:
        idx = padding_k_idx(mp, kind="joint")
        result = []
        for kp in range(nkpts):
            for kq in range(nkpts):
                for kr in range(nkpts):
                    ks = mp.khelper.kconserv[kp, kq, kr]
                    result.append(dm2[kp,kq,kr][np.ix_(idx[kp],idx[kq],idx[kr],idx[ks])])
        return result


def _gamma1_intermediates(mp, t2=None):
    # Memory optimization should be here
    if t2 is None:
        t2 = mp.t2
    if t2 is None:
        raise NotImplementedError("Run kmp2.kernel with `with_t2=True`")
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    dtype = t2.dtype

    dm1occ = np.zeros((nkpts, nocc, nocc), dtype=dtype)
    dm1vir = np.zeros((nkpts, nvir, nvir), dtype=dtype)

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]

                dm1vir[kb] += einsum('ijax,ijay->yx', t2[ki][kj][ka].conj(), t2[ki][kj][ka]) * 2 -\
                              einsum('ijax,ijya->yx', t2[ki][kj][ka].conj(), t2[ki][kj][kb])
                dm1occ[kj] += einsum('ixab,iyab->xy', t2[ki][kj][ka].conj(), t2[ki][kj][ka]) * 2 -\
                              einsum('ixab,iyba->xy', t2[ki][kj][ka].conj(), t2[ki][kj][kb])
    return -dm1occ, dm1vir


class KMP2(mp2.MP2):
    _keys = {'kpts', 'nkpts', 'khelper'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        if isinstance(self._scf.with_df, df.GDF):
            self.with_df_ints = True
        else:
            self.with_df_ints = False

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        if isinstance(self.kpts, libkpts.KPoints):
            self.nkpts = self.kpts.nkpts
            self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts.kpts)
            #padding has to be after transformation
            self.mo_energy = self.kpts.transform_mo_energy(mf.mo_energy)
            self.mo_coeff = self.kpts.transform_mo_coeff(mo_coeff)
            self.mo_occ = self.kpts.transform_mo_occ(mo_occ)
        else:
            self.nkpts = len(self.kpts)
            self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ss = None
        self.e_corr_os = None
        self.t2 = None

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    def dump_flags(self):
        logger.info(self, "")
        logger.info(self, "******** %s ********", self.__class__)
        logger.info(self, "nkpts = %d", self.nkpts)
        logger.info(self, "nocc = %d", self.nocc)
        logger.info(self, "nmo = %d", self.nmo)
        logger.info(self, "with_df_ints = %s", self.with_df_ints)

        if self.frozen is not None:
            logger.info(self, "frozen orbitals = %s", self.frozen)
        logger.info(
            self,
            "max_memory %d MB (current use %d MB)",
            self.max_memory,
            lib.current_memory()[0],
        )
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        mo_coeff, mo_energy = _add_padding(self, mo_coeff, mo_energy)

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()

        return self.e_corr, self.t2

    to_gpu = lib.to_gpu

KRMP2 = KMP2


from pyscf.pbc import scf
scf.khf.KRHF.MP2 = lib.class_as_method(KRMP2)
scf.kghf.KGHF.MP2 = None
scf.krohf.KROHF.MP2 = None


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp

    cell = gto.Cell()
    cell.atom='''
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
    cell.build()

    # Running HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    ehf = kmf.kernel()

    mymp = mp.KMP2(kmf)
    emp2, t2 = mymp.kernel()
    print(emp2 - -0.204721432828996)
