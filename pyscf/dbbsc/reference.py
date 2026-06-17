#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

"""Reference-orbital selection for DBBSC ECMD."""

from dataclasses import dataclass
from enum import Enum

import numpy

from pyscf.data import elements


class ReferenceKind(Enum):
    RHF = 'rhf'
    ROHF = 'rohf'
    UHF = 'uhf'


@dataclass(frozen=True)
class ReferenceOrbitals:
    """Spin-resolved molecular orbital data used by DBBSC ECMD."""

    mol: object
    mf: object
    source: str
    mo_basis: object
    mo_occ: object
    same_orbitals: bool


def _get_scf_method(method):
    return method._scf if getattr(method, '_scf', None) is not None else method


def _reference_kind(mo_coeff=None, mo_occ=None):
    is_uhf = (
        isinstance(mo_coeff, (tuple, list))
        or getattr(mo_coeff, 'ndim', 0) == 3
        or (mo_coeff is None and (isinstance(mo_occ, (tuple, list)) or getattr(numpy.asarray(mo_occ), 'ndim', 0) == 2))
    )
    if is_uhf:
        return ReferenceKind.UHF

    is_rohf = numpy.any(abs(mo_occ - 1) < 1e-10)
    if numpy.any(~((mo_occ == 0) | (mo_occ == 1) | (mo_occ == 2))):
        raise NotImplementedError(
            'DBBSC ECMD currently requires RHF, ROHF, or UHF references; '
            'general fractional occupations are not implemented.'
        )
    if is_rohf:
        return ReferenceKind.ROHF
    return ReferenceKind.RHF


def _frozen_mask(mol, mo_occ, frozen):
    mask = numpy.zeros(mo_occ.size, dtype=bool)
    if frozen is None:
        return mask

    if isinstance(frozen, str):
        scheme = frozen.lower()
        if scheme == 'chemcore':
            frozen = elements.chemcore(mol)
        elif scheme == 'none':
            frozen = 0
        else:
            raise ValueError('Unsupported DBBSC frozen orbital scheme %r' % frozen)

    if isinstance(frozen, (bool, numpy.bool_)):
        raise TypeError('DBBSC frozen orbitals must be specified as an int, sequence, tuple, or named scheme')
    if isinstance(frozen, (int, numpy.integer)):
        mask[:frozen] = True
    else:
        mask[numpy.asarray(frozen, dtype=int)] = True
    return mask


def _spin_frozen_masks(mol, spin_occ, frozen):
    if isinstance(frozen, (tuple, list)) and len(frozen) == 2 and not isinstance(frozen[0], (int, numpy.integer)):
        return tuple(_frozen_mask(mol, occ, item) for occ, item in zip(spin_occ, frozen))
    return tuple(_frozen_mask(mol, occ, frozen) for occ in spin_occ)


def _frozen_orbital_mask(method, mol, mo_occ, frozen=None):
    if frozen is None and hasattr(method, 'get_frozen_mask'):
        return method.get_frozen_mask()
    if _reference_kind(mo_occ=mo_occ) is ReferenceKind.UHF:
        return tuple(~mask for mask in _spin_frozen_masks(mol, mo_occ, frozen))
    return ~_frozen_mask(mol, numpy.asarray(mo_occ), frozen)


def _check_active_mask(mask, size, label):
    mask = numpy.asarray(mask, dtype=bool)
    if mask.size != size:
        raise RuntimeError('Active-orbital mask shape is inconsistent with %s mo_coeff' % label)
    return mask


def _restricted_reference(method, source, reference_kind, frozen=None):
    mo_coeff = source.mo_coeff
    mo_occ = source.mo_occ

    is_rohf = reference_kind is ReferenceKind.ROHF

    mf = _get_scf_method(method)
    active = _check_active_mask(_frozen_orbital_mask(method, mf.mol, mo_occ, frozen), mo_coeff.shape[1], 'restricted')
    if is_rohf:
        occa_mask = (mo_occ > 0) & active
        occb_mask = (abs(mo_occ - 2) < 1e-10) & active
        occa = mo_coeff[:, occa_mask]
        occb = mo_coeff[:, occb_mask]
        return 'rohf', (mo_coeff, mo_coeff), (occa, occb)

    occ = mo_coeff[:, (abs(mo_occ - 2) < 1e-10) & active]
    return 'rhf', mo_coeff, occ


def _unrestricted_reference(method, source, frozen=None):
    mo_coeff = source.mo_coeff
    mo_occ = source.mo_occ

    mf = _get_scf_method(method)
    active = _frozen_orbital_mask(method, mf.mol, mo_occ, frozen)
    active = (
        _check_active_mask(active[0], mo_coeff[0].shape[1], 'UHF'),
        _check_active_mask(active[1], mo_coeff[1].shape[1], 'UHF'),
    )
    occ = (
        mo_coeff[0][:, (abs(mo_occ[0] - 1) < 1e-10) & active[0]],
        mo_coeff[1][:, (abs(mo_occ[1] - 1) < 1e-10) & active[1]],
    )
    return 'uhf', mo_coeff, occ


def _resolve_reference_orbitals(method, frozen=None):
    mf = _get_scf_method(method)
    source = method if getattr(method, 'mo_coeff', None) is not None else mf
    reference_kind = _reference_kind(getattr(source, 'mo_coeff', None), getattr(source, 'mo_occ', None))
    if reference_kind is ReferenceKind.UHF:
        ref_source, mo_basis, mo_occ = _unrestricted_reference(method, source, frozen=frozen)
    else:
        ref_source, mo_basis, mo_occ = _restricted_reference(method, source, reference_kind, frozen=frozen)
    same_orbitals = ref_source == 'rhf'
    return ReferenceOrbitals(mf.mol, mf, ref_source, mo_basis, mo_occ, same_orbitals)


def _mo_basis(reference, spin):
    return reference.mo_basis if reference.same_orbitals else reference.mo_basis[spin]


def _mo_occ(reference, spin):
    return reference.mo_occ if reference.same_orbitals else reference.mo_occ[spin]
