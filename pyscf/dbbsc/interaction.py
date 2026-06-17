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

"""Finite-basis effective interactions for DBBSC ECMD."""

from dataclasses import dataclass

import numpy

from pyscf import ao2mo, df
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.dbbsc.reference import _mo_basis, _mo_occ


@dataclass(frozen=True)
class FourCenterInteraction:
    """Four-center finite-basis effective interaction tensor."""

    tensor: numpy.ndarray


@dataclass(frozen=True)
class DensityFittedInteraction:
    """Density-fitted factors for the finite-basis effective interaction."""

    factors: tuple


def _interaction_eris_same(mol, mo_basis, mo_occ, max_memory=2000, verbose=None):
    nocc = mo_occ.shape[1]
    nmo = mo_basis.shape[1]
    eris = ao2mo.general(
        mol,
        (mo_occ, mo_basis, mo_occ, mo_basis),
        compact=False,
        max_memory=max_memory,
        verbose=verbose if verbose is not None else mol.verbose,
    ).reshape(nocc, nmo, nocc, nmo)
    return numpy.asarray(eris, order='C')


def _interaction_eris(reference, max_memory=2000, verbose=None):
    if reference.same_orbitals:
        return _interaction_eris_same(
            reference.mol, reference.mo_basis, reference.mo_occ, max_memory=max_memory, verbose=verbose
        )

    mo_occ_a = _mo_occ(reference, 0)
    mo_basis_a = _mo_basis(reference, 0)
    mo_occ_b = _mo_occ(reference, 1)
    mo_basis_b = _mo_basis(reference, 1)
    eris = ao2mo.general(
        reference.mol,
        (mo_occ_a, mo_basis_a, mo_occ_b, mo_basis_b),
        compact=False,
        max_memory=max_memory,
        verbose=verbose if verbose is not None else reference.mol.verbose,
    ).reshape(
        mo_occ_a.shape[1],
        mo_basis_a.shape[1],
        mo_occ_b.shape[1],
        mo_basis_b.shape[1],
    )
    return numpy.asarray(eris, order='C')


def _df_pair_descriptor(occ, basis):
    mosym, npair, moij, ijslice = _conc_mos(occ, basis, compact=False)
    return mosym, npair, moij, ijslice, occ.shape[1], basis.shape[1]


def _interaction_df(reference, aux_basis, max_memory=2000, verbose=None):
    with_df = df.DF(reference.mol, auxbasis=aux_basis)
    with_df.max_memory = max_memory
    with_df.verbose = verbose if verbose is not None else reference.mol.verbose
    with_df.build()

    mo_occ_a = _mo_occ(reference, 0)
    mo_basis_a = _mo_basis(reference, 0)
    mo_occ_b = _mo_occ(reference, 1)
    mo_basis_b = _mo_basis(reference, 1)
    desc_a = _df_pair_descriptor(mo_occ_a, mo_basis_a)
    desc_b = desc_a if reference.same_orbitals else _df_pair_descriptor(mo_occ_b, mo_basis_b)
    naux = with_df.get_naoaux()

    factor_a = numpy.empty((desc_a[4] * desc_a[5], naux))
    factor_b = factor_a if desc_b is desc_a else numpy.empty((desc_b[4] * desc_b[5], naux))

    p0 = 0
    for cderi in with_df.loop():
        cderi = numpy.asarray(cderi, order='C')
        p1 = p0 + cderi.shape[0]
        block = _ao2mo.nr_e2(cderi, desc_a[2], desc_a[3], aosym='s2', mosym=desc_a[0])
        factor_a[:, p0:p1] = block.reshape(cderi.shape[0], -1).T
        if factor_b is not factor_a:
            block = _ao2mo.nr_e2(cderi, desc_b[2], desc_b[3], aosym='s2', mosym=desc_b[0])
            factor_b[:, p0:p1] = block.reshape(cderi.shape[0], -1).T
        p0 = p1

    if p0 != naux:
        raise RuntimeError('Inconsistent density-fitting DBBSC factor size')
    return factor_a, factor_b


def _interaction_size(reference):
    mo_basis_a = _mo_basis(reference, 0)
    mo_basis_b = _mo_basis(reference, 1)
    mo_occ_a = _mo_occ(reference, 0)
    mo_occ_b = _mo_occ(reference, 1)
    return mo_occ_a.shape[1] * mo_basis_a.shape[1] * mo_occ_b.shape[1] * mo_basis_b.shape[1] * 8


def _choose_interaction(log, reference, aux_basis=None, max_memory=2000, verbose=None):
    if aux_basis is None:
        log.debug(
            'DBBSC interaction: four-center integrals, estimated size %.1f MB',
            _interaction_size(reference) / 1e6,
        )
        return FourCenterInteraction(_interaction_eris(reference, max_memory=max_memory, verbose=verbose))
    log.info('DBBSC interaction: density fitting with aux_basis = %s', aux_basis)
    factors = _interaction_df(reference, aux_basis, max_memory, verbose)
    return DensityFittedInteraction(factors)
