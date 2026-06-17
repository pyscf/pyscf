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

"""Grid-block evaluation helpers for DBBSC ECMD."""

import numpy

from pyscf import dft, lib
from pyscf.dbbsc.constants import DBBSC_BLOCK_MEMORY, MU_PREFACTOR, ONTOP_CUTOFF
from pyscf.dbbsc.interaction import DensityFittedInteraction
from pyscf.dbbsc.reference import _mo_basis, _mo_occ
from pyscf.dft.gen_grid import BLKSIZE


def _mo_values(mol, ao, mo_coeff, non0tab=None, ao_loc=None):
    shls_slice = (0, mol.nbas)
    if ao_loc is None:
        ao_loc = mol.ao_loc_nr()
    if getattr(ao, 'ndim', 2) == 2:
        return dft.numint._dot_ao_dm(mol, ao, mo_coeff, non0tab, shls_slice, ao_loc)
    return numpy.asarray([dft.numint._dot_ao_dm(mol, x, mo_coeff, non0tab, shls_slice, ao_loc) for x in ao])


def _rho_from_occ_values(occ, settings):
    if settings.xctype == 'LDA':
        return numpy.einsum('gi,gi->g', occ, occ)

    ngrids = occ.shape[1]
    if settings.xctype == 'GGA':
        rho = numpy.empty((4, ngrids))
    elif settings.with_lapl:
        rho = numpy.empty((6, ngrids))
    else:
        rho = numpy.empty((5, ngrids))

    occ0 = occ[0]
    rho[0] = numpy.einsum('gi,gi->g', occ0, occ0)
    for k in range(1, 4):
        rho[k] = 2.0 * numpy.einsum('gi,gi->g', occ0, occ[k])

    if settings.xctype == 'MGGA':
        tau = numpy.zeros(ngrids)
        for k in range(1, 4):
            tau += numpy.einsum('gi,gi->g', occ[k], occ[k])
        if settings.with_lapl:
            rho[4] = 2.0 * (numpy.einsum('gi,gi->g', occ0, occ[4] + occ[7] + occ[9]) + tau)
            rho[5] = 0.5 * tau
        else:
            rho[4] = 0.5 * tau
    return rho


def _spin_values(reference, ao, ao0, spin, settings, non0tab, ao_loc):
    occ = _mo_values(reference.mol, ao, _mo_occ(reference, spin), non0tab, ao_loc)
    rho = _rho_from_occ_values(occ, settings)
    occ_g = occ[0] if settings.deriv > 0 else occ
    mo_g = _mo_values(reference.mol, ao0, _mo_basis(reference, spin), non0tab, ao_loc)
    return rho, occ_g, mo_g


def _safe_local_interaction(n2, numer):
    w_eff = numpy.full(n2.shape, numpy.inf)
    mask = n2 > ONTOP_CUTOFF
    w_eff[mask] = numer[mask] / n2[mask]
    return MU_PREFACTOR * w_eff


def _mu_four_center_general(mo_ga, occ_ga, mo_gb, occ_gb, eris, workspace):
    rhoa = numpy.einsum('gi,gi->g', occ_ga, occ_ga)
    rhob = numpy.einsum('gi,gi->g', occ_gb, occ_gb)
    n2 = 2.0 * rhoa * rhob

    paira = workspace.pair_values('paira', occ_ga, mo_ga)
    eris = eris.reshape(paira.shape[1], -1)
    tmp = lib.dot(paira, eris, c=workspace.empty('tmp', (occ_ga.shape[0], eris.shape[1])))
    pairb = workspace.pair_values('pairb', occ_gb, mo_gb)
    numer = 2.0 * numpy.einsum('gx,gx->g', tmp, pairb)
    return _safe_local_interaction(n2, numer)


def _mu_four_center_same(mo_g, occ_g, eris, workspace):
    rho = numpy.einsum('gi,gi->g', occ_g, occ_g)
    n2 = 2.0 * rho * rho

    pair = workspace.pair_values('paira', occ_g, mo_g)
    eris = eris.reshape(pair.shape[1], pair.shape[1])
    tmp = lib.dot(pair, eris, c=workspace.empty('tmp', (occ_g.shape[0], eris.shape[1])))
    numer = 2.0 * numpy.einsum('gx,gx->g', tmp, pair)
    return _safe_local_interaction(n2, numer)


def _mu_df(mo_ga, occ_ga, mo_gb, occ_gb, factors, same_orbitals, workspace):
    if same_orbitals:
        rhoa = rhob = numpy.einsum('gi,gi->g', occ_ga, occ_ga)
    else:
        rhoa = numpy.einsum('gi,gi->g', occ_ga, occ_ga)
        rhob = numpy.einsum('gi,gi->g', occ_gb, occ_gb)
    n2 = 2.0 * rhoa * rhob

    paira = workspace.pair_values('paira', occ_ga, mo_ga)
    ya = lib.dot(paira, factors[0], c=workspace.empty('ya', (occ_ga.shape[0], factors[0].shape[1])))
    if same_orbitals and factors[0] is factors[1]:
        numer = 2.0 * numpy.einsum('gL,gL->g', ya, ya)
    else:
        pairb = workspace.pair_values('pairb', occ_gb, mo_gb)
        yb = lib.dot(pairb, factors[1], c=workspace.empty('yb', (occ_gb.shape[0], factors[1].shape[1])))
        numer = 2.0 * numpy.einsum('gL,gL->g', ya, yb)
    return _safe_local_interaction(n2, numer)


def _mu_from_block(mo_ga, occ_ga, mo_gb, occ_gb, interaction, same_orbitals, workspace):
    if isinstance(interaction, DensityFittedInteraction):
        return _mu_df(mo_ga, occ_ga, mo_gb, occ_gb, interaction.factors, same_orbitals, workspace)
    if same_orbitals:
        return _mu_four_center_same(mo_ga, occ_ga, interaction.tensor, workspace)
    return _mu_four_center_general(mo_ga, occ_ga, mo_gb, occ_gb, interaction.tensor, workspace)


def _dbbsc_block_size(grids, reference, interaction, max_memory=2000):
    mo_occ_a = _mo_occ(reference, 0)
    mo_basis_a = _mo_basis(reference, 0)
    mo_occ_b = _mo_occ(reference, 1)
    mo_basis_b = _mo_basis(reference, 1)
    npaira = mo_occ_a.shape[1] * mo_basis_a.shape[1]
    npairb = mo_occ_b.shape[1] * mo_basis_b.shape[1]
    if isinstance(interaction, DensityFittedInteraction):
        naux = interaction.factors[0].shape[1]
        words = npaira + naux
        if not reference.same_orbitals or interaction.factors[0] is not interaction.factors[1]:
            words += npairb + naux
    else:
        words = npaira * 2 if reference.same_orbitals else npaira + npairb * 2

    memory = max(1.0, min(float(max_memory), DBBSC_BLOCK_MEMORY))
    blksize = int(memory * 1e6 / (max(words, 1) * 8))
    blksize = max(BLKSIZE, (blksize // BLKSIZE) * BLKSIZE)
    ngrid = grids.coords.shape[0]
    return min(blksize, ((ngrid + BLKSIZE - 1) // BLKSIZE) * BLKSIZE)


def _make_grids(mf, mol, grids):
    if grids is None:
        grids = dft.gen_grid.Grids(mol)
        grids.level = getattr(getattr(mf, 'grids', None), 'level', grids.level)
        grids.build(with_non0tab=True)
    elif grids.coords is None:
        grids.build(with_non0tab=True)
    return grids
