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

"""Correlation functional validation for DBBSC ECMD."""

from dataclasses import dataclass

import numpy

from pyscf import dft


@dataclass(frozen=True)
class FunctionalSettings:
    requested_name: str
    xc: str
    xctype: str
    deriv: int
    with_lapl: bool


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
