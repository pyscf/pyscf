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

"""
Helpers for DBBSC ECMD.
"""

from dataclasses import dataclass
from enum import Enum

import numpy

from pyscf import dft
from pyscf.data import elements


@dataclass(frozen=True)
class FunctionalSettings:
    requested_name: str
    xc: str
    xctype: str
    deriv: int
    with_lapl: bool


class ReferenceKind(Enum):
    RHF = 'rhf'
    ROHF = 'rohf'
    UHF = 'uhf'


class _GridBlockWorkspace:
    def __init__(self):
        self._arrays = {}

    def empty(self, name, shape):
        out = self._arrays.get(name)
        if out is None or out.shape != shape:
            out = numpy.empty(shape)
            self._arrays[name] = out
        return out

    def pair_values(self, name, occ_g, mo_g):
        out = self.empty(name, (occ_g.shape[0], occ_g.shape[1] * mo_g.shape[1]))
        numpy.multiply(
            occ_g[:, :, None],
            mo_g[:, None, :],
            out=out.reshape(occ_g.shape[0], occ_g.shape[1], mo_g.shape[1]),
        )
        return out


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


def _as_correlation_xc(functional):
    if not isinstance(functional, str):
        raise TypeError('DBBSC ECMD functional must be a string')
    functional = functional.strip()
    if not functional:
        raise ValueError('DBBSC ECMD functional must not be empty')

    if ',' in functional:
        if functional.count(',') != 1:
            raise ValueError('DBBSC ECMD functional must be a pure correlation expression')
        exchange_part, correlation_part = functional.split(',')
        if exchange_part.strip():
            raise ValueError('DBBSC ECMD functional must not include exchange terms')
        if not correlation_part.strip():
            raise ValueError('DBBSC ECMD functional must include a correlation term')
        return functional

    upper = functional.upper()
    if upper.startswith(('LDA_C_', 'GGA_C_', 'MGGA_C_')):
        return functional
    return ',' + functional


def _is_correlation_id(xc_id):
    prefixes = ('LDA_C_', 'GGA_C_', 'MGGA_C_')
    xc_id = int(xc_id)
    for name, value in dft.libxc.XC_CODES.items():
        if isinstance(value, (int, numpy.integer)) and int(value) == xc_id and name.startswith(prefixes):
            return True
    return False


def _validate_correlation_xc(functional, xc_code):
    try:
        hyb, fn_facs = dft.libxc.parse_xc(xc_code)
    except Exception as err:
        raise ValueError('Unknown DBBSC ECMD functional %r (%s)' % (functional, err)) from err

    if any(abs(x) > 1e-15 for x in hyb):
        raise ValueError('%s is a hybrid functional, not an ECMD correlation model' % xc_code)
    if not fn_facs:
        raise ValueError('DBBSC ECMD functional %r has no correlation terms' % functional)

    bad_ids = [int(xc_id) for xc_id, fac in fn_facs if abs(fac) > 1e-15 and not _is_correlation_id(xc_id)]
    if bad_ids:
        raise ValueError('%s contains non-correlation LibXC functionals %s' % (xc_code, bad_ids))


def _resolve_functional(functional):
    xc_code = _as_correlation_xc(functional)
    _validate_correlation_xc(functional, xc_code)
    try:
        xctype = dft.libxc.xc_type(xc_code)
    except Exception as err:
        raise ValueError('Unknown DBBSC ECMD functional %r (%s)' % (functional, err)) from err

    with_lapl = dft.libxc.needs_laplacian(xc_code)
    if xctype == 'MGGA':
        deriv = 2 if with_lapl else 1
    elif xctype == 'GGA':
        deriv = 1
    else:
        deriv = 0
    return FunctionalSettings(functional.lower(), xc_code, xctype, deriv, with_lapl)


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
