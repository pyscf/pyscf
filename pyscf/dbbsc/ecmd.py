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
ECMD density-based basis-set correction.

This module implements a posteriori ECMD variants of the density-based
basis-set correction (DBBSC) for single-determinant references.  The
correction uses the finite-basis effective interaction to build the local
range-separation function

    mu_B(r) = sqrt(pi) W_B(r,r) / 2

and evaluates the ECMD short-range interpolation

    eps_c,md^sr = eps_c / (1 + beta mu_B^3)

with the selected correlation energy density and on-top pair-density model in
beta.

Note: Exact on-top pair-density is not supported.

Refs:
- JPCL 10, 2931-2937 (2019); DOI:10.1021/acs.jpclett.9b01176
- JCC  45, 1247-1253 (2024); DOI:10.1002/jcc.27325
- JCTC 19, 8210-8222 (2023); DOI:10.1021/acs.jctc.3c00979
"""

from dataclasses import dataclass

import numpy

from pyscf import dft, lib
from pyscf.dbbsc.functional import _resolve_functional
from pyscf.dbbsc.grid import _dbbsc_block_size, _make_grids, _mu_from_block, _spin_values
from pyscf.dbbsc.interaction import _choose_interaction
from pyscf.dbbsc.ontop import _density_zeta, _ecmd_eps
from pyscf.dbbsc.reference import _get_scf_method, _mo_occ, _resolve_reference_orbitals
from pyscf.dbbsc.workspace import _GridBlockWorkspace
from pyscf.lib import logger


@dataclass(frozen=True)
class ECMDOptions:
    """Options controlling a DBBSC ECMD correction evaluation."""

    aux_basis: object = None
    functional: str = 'pbe'
    frozen: object = None
    grids: object = None
    ontop_model: str = 'ueg'
    max_memory: float = 2000
    verbose: object = None


def _ecmd_from_options(method, options):
    settings = _resolve_functional(options.functional)

    reference = _resolve_reference_orbitals(method, frozen=options.frozen)
    log = logger.new_logger(method, options.verbose)
    nocc_a = _mo_occ(reference, 0).shape[1]
    nocc_b = _mo_occ(reference, 1).shape[1]
    if nocc_a + nocc_b <= 1:
        log.info('E(DBBSC-ECMD-%s/%s) = 0.0', settings.xc, options.ontop_model)
        return 0.0

    grids = _make_grids(reference.mf, reference.mol, options.grids)
    interaction = _choose_interaction(
        log, reference, aux_basis=options.aux_basis, max_memory=options.max_memory, verbose=options.verbose
    )
    blksize = _dbbsc_block_size(grids, reference, interaction, max_memory=options.max_memory)

    ni = dft.numint.NumInt()
    ao_loc = reference.mol.ao_loc_nr()
    workspace = _GridBlockWorkspace()
    e_dbbsc = 0.0
    t0 = (logger.process_clock(), logger.perf_counter())
    for ao, mask, weight, _ in ni.block_loop(
        reference.mol,
        grids,
        reference.mol.nao_nr(),
        deriv=settings.deriv,
        max_memory=options.max_memory,
        blksize=blksize,
    ):
        ao0 = ao[0] if settings.deriv > 0 else ao
        rhoa, occ_ga, mo_ga = _spin_values(reference, ao, ao0, 0, settings, mask, ao_loc)
        if reference.same_orbitals:
            rhob, occ_gb, mo_gb = rhoa, occ_ga, mo_ga
        else:
            rhob, occ_gb, mo_gb = _spin_values(reference, ao, ao0, 1, settings, mask, ao_loc)

        mu = _mu_from_block(mo_ga, occ_ga, mo_gb, occ_gb, interaction, reference.same_orbitals, workspace)
        density, zeta = _density_zeta(rhoa, rhob)
        eps = _ecmd_eps(settings, density, rhoa, rhob, mu, zeta, options.ontop_model)
        e_dbbsc += numpy.dot(weight, density * eps)

    log.timer('DBBSC ECMD correction', *t0)
    e_dbbsc = float(e_dbbsc)
    log.info('E(DBBSC-ECMD-%s/%s) = %.15g', settings.xc, options.ontop_model, e_dbbsc)
    return e_dbbsc


def kernel(
    method,
    *,
    aux_basis=None,
    functional='pbe',
    frozen=None,
    grids=None,
    ontop_model='ueg',
    max_memory=2000,
    verbose=None,
):
    """Compute the DBBSC ECMD energy correction."""
    options = ECMDOptions(
        aux_basis=aux_basis,
        functional=functional,
        frozen=frozen,
        grids=grids,
        ontop_model=ontop_model,
        max_memory=max_memory,
        verbose=verbose,
    )
    return _ecmd_from_options(method, options)


energy = kernel


class ECMD(lib.StreamObject):
    """A posteriori DBBSC ECMD correction driver."""

    _keys = {
        'method',
        'functional',
        'ontop_model',
        'aux_basis',
        'frozen',
        'grids',
        'max_memory',
        'verbose',
        'stdout',
        'e_dbbsc',
    }

    def __init__(
        self,
        method,
        *,
        aux_basis=None,
        functional='pbe',
        frozen=None,
        grids=None,
        ontop_model='ueg',
        max_memory=2000,
        verbose=None,
    ):
        self.method = method
        mf = _get_scf_method(method)
        self.mol = mf.mol
        self.verbose = getattr(method, 'verbose', self.mol.verbose) if verbose is None else verbose
        self.stdout = getattr(method, 'stdout', self.mol.stdout)
        self.grids = grids
        self.functional = functional
        self.ontop_model = ontop_model
        self.aux_basis = aux_basis
        self.max_memory = (
            getattr(method, 'max_memory', getattr(mf, 'max_memory', self.max_memory))
            if max_memory is None
            else max_memory
        )
        self.frozen = frozen
        self.e_dbbsc = None

    def kernel(
        self,
        *,
        aux_basis=None,
        functional=None,
        frozen=None,
        grids=None,
        ontop_model=None,
        max_memory=None,
        verbose=None,
    ):
        """Compute the DBBSC ECMD correction and save it in ``self.e_dbbsc``."""
        options = ECMDOptions(
            aux_basis=self.aux_basis if aux_basis is None else aux_basis,
            functional=self.functional if functional is None else functional,
            frozen=self.frozen if frozen is None else frozen,
            grids=self.grids if grids is None else grids,
            ontop_model=self.ontop_model if ontop_model is None else ontop_model,
            max_memory=self.max_memory if max_memory is None else max_memory,
            verbose=self.verbose if verbose is None else verbose,
        )
        self.e_dbbsc = _ecmd_from_options(self.method, options)
