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

"""ECMD energy-density and on-top pair-density formulas."""

import numpy

from pyscf import dft
from pyscf.dbbsc.constants import (
    ECMD_BETA_PREFACTOR,
    ONTOP_CUTOFF,
    RHO_CUTOFF,
    UEG_B,
    UEG_C,
    UEG_D,
    UEG_E,
    UEG_EXPONENT,
)


def _ueg_g0(rho):
    """UEG on-top pair-distribution fit from Eq. 46 of 10.1103/PhysRevA.73.032506."""
    rs = numpy.zeros_like(rho)
    mask = rho > RHO_CUTOFF
    rs[mask] = (3.0 / (4.0 * numpy.pi * rho[mask])) ** (1.0 / 3.0)

    g0 = numpy.zeros_like(rho)
    poly = 1.0 - UEG_B * rs[mask] + UEG_C * rs[mask] ** 2 + UEG_D * rs[mask] ** 3 + UEG_E * rs[mask] ** 4
    g0[mask] = 0.5 * poly * numpy.exp(-UEG_EXPONENT * rs[mask])
    return numpy.maximum(g0, 0.0)


def _ueg_ontop_pair_density(rho, zeta):
    """UEG approximation in the text after Eq. 14b of 10.1021/acs.jpclett.9b01176."""
    n2 = rho**2 * (1.0 - zeta**2) * _ueg_g0(rho)
    return numpy.where(rho > RHO_CUTOFF, numpy.maximum(n2, 0.0), 0.0)


def _ontop_pair_density(rho, zeta, model):
    model = model.lower()
    if model == 'ueg':
        return _ueg_ontop_pair_density(rho, zeta)
    if model == 'exact':
        raise NotImplementedError('Exact on-top pair-density support is not implemented yet.')
    raise ValueError('Unknown DBBSC on-top pair-density model %r' % model)


def _correlation_eps(settings, rhoa, rhob):
    rho = (rhoa[0], rhob[0]) if settings.xctype == 'LDA' and getattr(rhoa, 'ndim', 1) > 1 else (rhoa, rhob)
    exc = dft.libxc.eval_xc(settings.xc, rho, spin=1, deriv=0)[0]
    return exc


def _ecmd_eps(settings, rho, rhoa, rhob, mu, zeta, ontop_model):
    """Evaluate Eq. 14a and Eq. 14b of 10.1021/acs.jpclett.9b01176."""
    eps_c = _correlation_eps(settings, rhoa, rhob)
    n2 = _ontop_pair_density(rho, zeta, model=ontop_model)
    mask = (rho > RHO_CUTOFF) & (n2 > ONTOP_CUTOFF) & numpy.isfinite(mu)

    beta = numpy.zeros_like(rho)
    beta[mask] = ECMD_BETA_PREFACTOR * rho[mask] * eps_c[mask] / n2[mask]
    beta = numpy.maximum(beta, 0.0)

    eps = numpy.zeros_like(rho)
    eps[mask] = eps_c[mask] / (1.0 + beta[mask] * mu[mask] ** 3)
    return eps


def _density_zeta(rhoa, rhob):
    def _rho0(rho):
        return rho[0] if getattr(rho, 'ndim', 1) > 1 else rho

    rhoa0 = _rho0(rhoa)
    rhob0 = _rho0(rhob)
    rho = rhoa0 + rhob0
    zeta = numpy.zeros_like(rho)
    mask = rho > RHO_CUTOFF
    zeta[mask] = (rhoa0[mask] - rhob0[mask]) / rho[mask]
    zeta = numpy.clip(zeta, -1.0, 1.0)
    return rho, zeta
