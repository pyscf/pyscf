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
# Author: James McClain <jdmcclain47@gmail.com>
#


'''
kpoint-adapted unrestricted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab

t2 and eris are never stored in full, only a partial
eri of size (nkpts,nocc,nocc,nvir,nvir)
'''


import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp import kmp2
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import _frozen_sanity_check
from pyscf.lib.parameters import LARGE_DENOM

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    raise NotImplementedError

def padding_k_idx(mp, kind="split"):
    """For a description, see `padding_k_idx` in kmp2.py.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
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
        indexes_oa = []
        indexes_va = []
        indexes_ob = []
        indexes_vb = []
    else:
        indexesa = []
        indexesb = []

    dense_oa, dense_ob = mp.nocc
    dense_nmoa, dense_nmob = mp.nmo
    dense_va = dense_nmoa - dense_oa
    dense_vb = dense_nmob - dense_ob

    nocca_per_kpt, noccb_per_kpt = np.asarray(get_nocc(mp, per_kpoint=True))
    nmoa_per_kpt, nmob_per_kpt = np.asarray(get_nmo(mp, per_kpoint=True))

    # alpha spin
    for k_oa, k_nmoa in zip(nocca_per_kpt, nmoa_per_kpt):
        k_va = k_nmoa - k_oa
        if kind == "split":
            indexes_oa.append(np.arange(k_oa))
            indexes_va.append(np.arange(dense_va - k_va, dense_va))
        else:
            indexesa.append(np.concatenate((
                np.arange(k_oa),
                np.arange(dense_nmoa - k_va, dense_nmoa),
            )))

    # beta spin
    for k_ob, k_nmob in zip(noccb_per_kpt, nmob_per_kpt):
        k_vb = k_nmob - k_ob
        if kind == "split":
            indexes_ob.append(np.arange(k_ob))
            indexes_vb.append(np.arange(dense_vb - k_vb, dense_vb))
        else:
            indexesb.append(np.concatenate((
                np.arange(k_ob),
                np.arange(dense_nmob - k_vb, dense_nmob),
            )))

    if kind == "split":
        return [indexes_oa, indexes_va], [indexes_ob, indexes_vb]
    else:
        return indexesa, indexesb


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

    result = (np.zeros((nkpts, mp.nmo), dtype=mo_energy[0][0].dtype),
              np.zeros((nkpts, mp.nmo), dtype=mo_energy[0][0].dtype))
    for spin in [0, 1]:
        for k in range(nkpts):
            result[spin][np.ix_([k], padding_convention[k])] = mo_energy[spin][k][frozen_mask[k]]

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

    result = (np.zeros((nkpts, mo_coeff[0][0].shape[0], mp.nmo[0]), dtype=mo_coeff[0][0].dtype),
              np.zeros((nkpts, mo_coeff[1][0].shape[0], mp.nmo[1]), dtype=mo_coeff[0][0].dtype))
    for spin in [0, 1]:
        for k in range(nkpts):
            result[spin][np.ix_([k], np.arange(result[spin].shape[1]), padding_convention[spin][k])] = \
                mo_coeff[spin][k][:, frozen_mask[spin][k]]

    return result


def _is_arraylike(x):
    return isinstance(x, (tuple, list, np.ndarray))


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

    Notes:
        For specifying frozen orbitals inside mp, the following options are accepted:

            +=========================+========================================+===============================+
            | Argument (Example)      | Argument Meaning                       | Example Meaning               |
            +=========================+========================================+===============================+
            | int (1)                 | Freeze the same number of orbitals     | Freeze one (lowest) orbital   |
            |                         | regardless of spin and/or kpt          | for all kpts and spin cases   |
            +-------------------------+----------------------------------------+-------------------------------+
            | 2-tuple of list of int  | inner list: List of orbitals indices   | Freeze the orbitals [0,4] for |
            | ([0, 4], [0, 5, 6])     |   to freeze at all kpts                | spin0, and orbitals [0,5,6]   |
            |                         | outer list: Spin index                 | for spin1 at all kpts         |
            +-------------------------+----------------------------------------+-------------------------------+
            | list(2) of list of list | inner list: list of orbital indices to | Freeze orbital 0 for spin0 at |
            | ([[0,],[]],             |   freeze at each kpt for given spin    | kpt0, and freeze orbital 0,1  |
            |  [[0,1],[4]])           | outer list: spin index                 | for spin1 at kpt0 and orbital |
            |                         |                                        | 4 at kpt1                     |
            +-------------------------+----------------------------------------+-------------------------------+

    '''
    for spin in [0,1]:
        for i, moocc in enumerate(mp.mo_occ[spin]):
            if np.any(moocc % 1 != 0):
                raise RuntimeError(
                    "Fractional occupation numbers encountered @ kp={:d}: {}.  "
                    "This may have been caused by smearing of occupation numbers "
                    "in the mean-field calculation. If so, consider executing "
                    "mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior "
                    "to calling this".format(i, moocc))
    if mp._nocc is not None:
        return mp._nocc

    elif mp.frozen is None:
        nocc = [[np.count_nonzero(mp.mo_occ[0][k] > 0) for k in range(mp.nkpts)],
                [np.count_nonzero(mp.mo_occ[1][k] > 0) for k in range(mp.nkpts)]]

    elif isinstance(mp.frozen, (int, np.integer)):
        nocc = [0]*2
        for spin in [0,1]:
            nocc[spin] = [(np.count_nonzero(mp.mo_occ[spin][k] > 0) - mp.frozen) for k in range(mp.nkpts)]

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (int, np.integer))):  # case example: ([0, 4], [0, 5, 6])
        nocc = [0]*2
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            [_frozen_sanity_check(mp.frozen[spin], mp.mo_occ[spin][ikpt], ikpt) for ikpt in range(mp.nkpts)]
            nocc_spin = []
            for ikpt in range(mp.nkpts):
                max_occ_idx = np.max(np.where(mp.mo_occ[spin][ikpt] > 0))
                frozen_nocc = np.sum(np.array(mp.frozen[spin]) <= max_occ_idx)
                nocc_spin.append(np.count_nonzero(mp.mo_occ[spin][ikpt]) - frozen_nocc)
            nocc[spin] = nocc_spin

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (list, np.ndarray))):  # case example: ([[0,],[]], [[0,1],[4]])
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            nkpts = len(mp.frozen[spin])
            if nkpts != mp.nkpts:
                raise RuntimeError('Frozen list has a different number of k-points (length) than passed in'
                                   'mean-field/correlated calculation.  \n\nCalculation nkpts = %d, frozen'
                                   'list = %s (length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        nocc = [0]*2
        for spin in [0,1]:
            [_frozen_sanity_check(frozen, mo_occ, ikpt)
             for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen[spin], mp.mo_occ[spin])]
            nocc_spin = []
            for ikpt, frozen in enumerate(mp.frozen[spin]):
                max_occ_idx = np.max(np.where(mp.mo_occ[spin][ikpt] > 0))
                frozen_nocc = np.sum(np.array(frozen) <= max_occ_idx)
                nocc_spin.append(np.count_nonzero(mp.mo_occ[spin][ikpt]) - frozen_nocc)
            nocc[spin] = nocc_spin
    else:
        raise NotImplementedError('No known conversion for frozen %s' % mp.frozen)

    for spin in [0,1]:
        assert any(np.array(nocc[spin]) > 0), (
            'Must have occupied orbitals (spin=%d)! \n\nnocc %s\nfrozen %s\nmo_occ %s' %
            (spin, nocc, mp.frozen, mp.mo_occ))

    nocca, noccb = nocc
    if not per_kpoint:
        nocca = np.amax(nocca)
        noccb = np.amax(noccb)

    return nocca, noccb


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

    nmo = [0, 0]
    if isinstance(mp.frozen, (int, np.integer)):
        for spin in [0,1]:
            nmo[spin] = [len(mp.mo_occ[spin][k]) - mp.frozen for k in range(mp.nkpts)]

    elif mp.frozen is None:
        nmo = [[len(mp.mo_occ[0][k]) for k in range(mp.nkpts)],
               [len(mp.mo_occ[1][k]) for k in range(mp.nkpts)]]

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (int, np.integer))):  # case example: ([0, 4], [0, 5, 6])
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            [_frozen_sanity_check(mp.frozen[spin], mp.mo_occ[spin][ikpt], ikpt) for ikpt in range(mp.nkpts)]
            nmo[spin] = [len(mp.mo_occ[spin][ikpt]) - len(mp.frozen[spin]) for ikpt in range(mp.nkpts)]

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (list, np.ndarray))):  # case example: ([[0,],[]], [[0,1],[4]])
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            nkpts = len(mp.frozen[spin])
            if nkpts != mp.nkpts:
                raise RuntimeError('Frozen list has a different number of k-points (length) than passed in'
                                   'mean-field/correlated calculation.  \n\nCalculation nkpts = %d, frozen'
                                   'list = %s (length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        for spin in [0,1]:
            [_frozen_sanity_check(mp.frozen[spin][ikpt], mp.mo_occ[spin][ikpt], ikpt) for ikpt in range(mp.nkpts)]
            nmo[spin] = [len(mp.mo_occ[spin][ikpt]) - len(mp.frozen[spin][ikpt]) for ikpt in range(nkpts)]

    else:
        raise NotImplementedError('No known conversion for frozen %s' % mp.frozen)

    for spin in [0,1]:
        assert all(np.array(nmo[spin]) > 0), (
            'Must have a positive number of orbitals! (spin=%d)'
            '\n\nnmo %s\nfrozen %s\nmo_occ %s' % (spin, nmo, mp.frozen, mp.mo_occ))

    nmoa, nmob = nmo
    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure    that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for        occupied
        # and virtual space
        nocca, noccb = mp.get_nocc(per_kpoint=True)
        nmoa = np.amax(nocca) + np.max(np.array(nmoa) - np.array(nocca))
        nmob = np.amax(noccb) + np.max(np.array(nmob) - np.array(noccb))

    return nmoa, nmob


def get_frozen_mask(mp):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `bool`): Boolean mask of orbitals to include.

    '''
    moidx = [[np.ones(x.size, dtype=bool) for x in mp.mo_occ[s]] for s in [0,1]]

    if mp.frozen is None:
        pass

    elif isinstance(mp.frozen, (int, np.integer)):
        for spin in [0,1]:
            for idx in moidx[spin]:
                idx[:mp.frozen] = False

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (int, np.integer))):  # case example: ([0, 4], [0, 5, 6])
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            [_frozen_sanity_check(mp.frozen[spin], mp.mo_occ[spin][ikpt], ikpt) for ikpt in range(mp.nkpts)]
            for ikpt, kpt_occ in enumerate(moidx[spin]):
                kpt_occ[mp.frozen[spin]] = False

    elif (_is_arraylike(mp.frozen[0]) and
          isinstance(mp.frozen[0][0], (list, np.ndarray))):  # case example: ([[0,],[]], [[0,1],[4]])
        assert (len(mp.frozen) == 2)
        for spin in [0,1]:
            nkpts = len(mp.frozen[spin])
            if nkpts != mp.nkpts:
                raise RuntimeError('Frozen list has a different number of k-points (length) than passed in'
                                   'mean-field/correlated calculation.  \n\nCalculation nkpts = %d, frozen'
                                   'list = %s (length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        for spin in [0,1]:
            [_frozen_sanity_check(mp.frozen[spin][ikpt], mp.mo_occ[spin][ikpt], ikpt) for ikpt in range(mp.nkpts)]
            for ikpt, kpt_occ in enumerate(moidx[spin]):
                kpt_occ[mp.frozen[spin][ikpt]] = False
    else:
        raise NotImplementedError('No known conversion for frozen %s' % mp.frozen)

    return moidx


def _add_padding(mp, mo_coeff, mo_energy):
    raise NotImplementedError("Implementation needs to be checked first")
    nmo = mp.nmo

    # Check if these are padded mo coefficients and energies
    if not np.all([x.shape[0] == nmo for x in mo_coeff]):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if not np.all([x.shape[0] == nmo for x in mo_energy]):
        mo_energy = padded_mo_energy(mp, mo_energy)
    return mo_coeff, mo_energy


class KUMP2(kmp2.KMP2):
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None):
        raise NotImplementedError
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        mo_coeff, mo_energy = _add_padding(self, mo_coeff, mo_energy)

        self.e_corr, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
        logger.log(self, 'KMP2 energy = %.15g', self.e_corr)
        return self.e_corr, self.t2


from pyscf.pbc import scf
scf.kuhf.KUHF.MP2 = lib.class_as_method(KUMP2)
