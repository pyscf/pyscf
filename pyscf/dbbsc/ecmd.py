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

from pyscf import ao2mo, df, dft, lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.dbbsc.ecmd_helper import (
    ReferenceKind,
    _check_active_mask,
    _frozen_orbital_mask,
    _get_scf_method,
    _GridBlockWorkspace,
    _reference_kind,
    _resolve_functional,
)
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.lib import logger

# Density threshold below which ECMD local quantities are set to zero.
RHO_CUTOFF = 1e-15
# On-top pair-density threshold below which the local interaction is skipped.
ONTOP_CUTOFF = 1e-30
# Maximum memory, in MB, used to size ECMD integration grid blocks.
DBBSC_BLOCK_MEMORY = 256
# Prefactor in mu_B(r) = sqrt(pi) W_B(r,r) / 2.
MU_PREFACTOR = 0.5 * numpy.sqrt(numpy.pi)
# Prefactor in the ECMD short-range interpolation beta parameter.
ECMD_BETA_PREFACTOR = 3.0 / (2.0 * numpy.sqrt(numpy.pi) * (1.0 - numpy.sqrt(2.0)))


@dataclass(frozen=True)
class ReferenceOrbitals:
    """Spin-resolved molecular orbital data used by DBBSC ECMD.

    Attributes:
        mol : :class:`pyscf.gto.Mole`
            Molecule associated with the reference mean-field object.
        mf : SCF object
            Underlying SCF object.  For post-SCF methods, this is ``method._scf``.
        source : str
            Reference type label, one of ``'rhf'``, ``'rohf'``, or ``'uhf'``.
        mo_basis : ndarray or tuple of ndarrays
            MO coefficient matrices used as the finite one-particle basis.
        mo_occ : ndarray or tuple of ndarrays
            Occupied orbital coefficient matrices used to build spin densities.
        same_orbitals : bool
            Whether alpha and beta channels share the same MO basis and occupied
            orbitals.
    """

    mol: object
    mf: object
    source: str
    mo_basis: object
    mo_occ: object
    same_orbitals: bool


@dataclass(frozen=True)
class ECMDOptions:
    """Options controlling a DBBSC ECMD correction evaluation.

    Attributes:
        aux_basis : str, basis dict, or None
            Auxiliary basis for density-fitted effective interactions.  If
            ``None``, four-center integrals are used.
        functional : str
            LibXC correlation functional used for the ECMD correlation energy
            density.  Bare names such as ``'pbe'`` are interpreted as
            correlation-only expressions.
        frozen : None, int, sequence, tuple, or str
            Frozen orbital selection.  ``'chemcore'`` freezes the chemical
            core and ``'none'`` includes all orbitals.
        grids : :class:`pyscf.dft.gen_grid.Grids` or None
            Numerical integration grid.  A default DFT grid is built when
            omitted.
        ontop_model : str
            On-top pair-density model.  Currently only ``'ueg'`` is supported.
        max_memory : float
            Memory limit in MB used for integral transformations and grid block
            sizing.
        verbose : int or None
            PySCF logger verbosity.
    """

    aux_basis: object = None
    functional: str = 'pbe'
    frozen: object = None
    grids: object = None
    ontop_model: str = 'ueg'
    max_memory: float = 2000
    verbose: object = None


@dataclass(frozen=True)
class FourCenterInteraction:
    """Four-center finite-basis effective interaction tensor.

    Attributes:
        tensor : ndarray
            Effective interaction in occupied/MO-index form with shape
            ``(nocc_a, nmo_a, nocc_b, nmo_b)``.
    """

    tensor: numpy.ndarray


@dataclass(frozen=True)
class DensityFittedInteraction:
    """Density-fitted factors for the finite-basis effective interaction.

    Attributes:
        factors : tuple of ndarrays
            Alpha and beta density-fitting factors.  Identical spin channels may
            share the same array object.
    """

    factors: tuple


def _ueg_g0(rho):
    """
    Eq. 46 from 10.1103/PhysRevA.73.032506
    """
    rs = numpy.zeros_like(rho)
    mask = rho > RHO_CUTOFF
    rs[mask] = (3.0 / (4.0 * numpy.pi * rho[mask])) ** (1.0 / 3.0)

    a_hd = -0.36583
    C = 0.08193
    D = -0.01277
    E = 0.001859
    d = 0.7524
    B = -2.0 * a_hd - d

    g0 = numpy.zeros_like(rho)
    poly = 1.0 - B * rs[mask] + C * rs[mask] ** 2 + D * rs[mask] ** 3 + E * rs[mask] ** 4
    g0[mask] = 0.5 * poly * numpy.exp(-d * rs[mask])
    return numpy.maximum(g0, 0.0)


def _ueg_ontop_pair_density(rho, zeta):
    """
    In text after Eq. 14b in 10.1021/acs.jpclett.9b01176
    """
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
    """
    Eq. 14a & Eq. 14b in 10.1021/acs.jpclett.9b01176
    """
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


def _choose_interaction(log, reference, aux_basis=None, max_memory=2000, verbose=None):
    if aux_basis is None:

        def _interaction_size(reference):
            mo_basis_a = _mo_basis(reference, 0)
            mo_basis_b = _mo_basis(reference, 1)
            mo_occ_a = _mo_occ(reference, 0)
            mo_occ_b = _mo_occ(reference, 1)
            return mo_occ_a.shape[1] * mo_basis_a.shape[1] * mo_occ_b.shape[1] * mo_basis_b.shape[1] * 8

        log.debug(
            'DBBSC interaction: four-center integrals, estimated size %.1f MB',
            _interaction_size(reference) / 1e6,
        )
        return FourCenterInteraction(_interaction_eris(reference, max_memory=max_memory, verbose=verbose))
    log.info('DBBSC interaction: density fitting with aux_basis = %s', aux_basis)
    factors = _interaction_df(reference, aux_basis, max_memory, verbose)
    return DensityFittedInteraction(factors)


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


def _mu_four_center_general(mo_ga, occ_ga, mo_gb, occ_gb, eris, workspace):
    rhoa = numpy.einsum('gi,gi->g', occ_ga, occ_ga)
    rhob = numpy.einsum('gi,gi->g', occ_gb, occ_gb)
    n2 = 2.0 * rhoa * rhob

    paira = workspace.pair_values('paira', occ_ga, mo_ga)
    eris = eris.reshape(paira.shape[1], -1)
    tmp = lib.dot(paira, eris, c=workspace.empty('tmp', (occ_ga.shape[0], eris.shape[1])))
    pairb = workspace.pair_values('pairb', occ_gb, mo_gb)
    numer = 2.0 * numpy.einsum('gx,gx->g', tmp, pairb)

    w_eff = numpy.full(n2.shape, numpy.inf)
    mask = n2 > ONTOP_CUTOFF
    w_eff[mask] = numer[mask] / n2[mask]
    return MU_PREFACTOR * w_eff


def _mu_four_center_same(mo_g, occ_g, eris, workspace):
    rho = numpy.einsum('gi,gi->g', occ_g, occ_g)
    n2 = 2.0 * rho * rho

    pair = workspace.pair_values('paira', occ_g, mo_g)
    eris = eris.reshape(pair.shape[1], pair.shape[1])
    tmp = lib.dot(pair, eris, c=workspace.empty('tmp', (occ_g.shape[0], eris.shape[1])))
    numer = 2.0 * numpy.einsum('gx,gx->g', tmp, pair)

    w_eff = numpy.full(n2.shape, numpy.inf)
    mask = n2 > ONTOP_CUTOFF
    w_eff[mask] = numer[mask] / n2[mask]
    return MU_PREFACTOR * w_eff


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

    w_eff = numpy.full(n2.shape, numpy.inf)
    mask = n2 > ONTOP_CUTOFF
    w_eff[mask] = numer[mask] / n2[mask]
    return MU_PREFACTOR * w_eff


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
    aux_basis=None,
    functional='pbe',
    frozen=None,
    grids=None,
    ontop_model='ueg',
    max_memory=2000,
    verbose=None,
):
    """Compute the DBBSC ECMD energy correction.

    Args:
        method : SCF or post-SCF object
            Converged RHF, ROHF, or UHF reference, or a post-SCF object with an
            underlying ``_scf`` object and optional ``get_frozen_mask`` method.

    Kwargs:
        aux_basis : str, basis dict, or None
            Auxiliary basis for the density-fitted effective interaction.  If
            ``None``, the four-center AO-to-MO transformation is used.
        functional : str
            Correlation functional for the ECMD energy density.  Plain names are
            interpreted as correlation-only LibXC expressions, e.g. ``'pbe'``
            is treated as ``',pbe'``.
        frozen : None, int, sequence, tuple, or str
            Frozen orbital selection.  ``None`` includes all orbitals for SCF
            objects and uses the method active-space mask for post-SCF objects.
            ``'chemcore'`` freezes chemical core orbitals and ``'none'``
            includes all orbitals.
        grids : :class:`pyscf.dft.gen_grid.Grids` or None
            Numerical integration grid.  If omitted, a grid is built from the
            reference molecule.
        ontop_model : str
            On-top pair-density model.  Currently only ``'ueg'`` is supported;
            ``'exact'`` is reserved for future support.
        max_memory : float
            Memory limit in MB for integral transformations and grid blocks.
        verbose : int or None
            PySCF logger verbosity.

    Returns:
        float
            DBBSC ECMD correction energy.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.dbbsc import ecmd
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> e_dbbsc = ecmd.kernel(mf, functional='pbe')
    """
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
    """A posteriori DBBSC ECMD correction driver.

    Attributes:
        method : SCF or post-SCF object
            Reference object used to build the single-determinant densities and
            finite-basis effective interaction.
        mol : :class:`pyscf.gto.Mole`
            Molecule associated with ``method``.
        functional : str
            Correlation functional used in the ECMD interpolation.
        ontop_model : str
            On-top pair-density model.  Currently only ``'ueg'`` is supported.
        aux_basis : str, basis dict, or None
            Auxiliary basis for the density-fitted effective interaction.  If
            ``None``, four-center integrals are used.
        frozen : None, int, sequence, tuple, or str
            Frozen orbital selection.
        grids : :class:`pyscf.dft.gen_grid.Grids` or None
            Numerical integration grid.
        max_memory : float
            Memory limit in MB.
        verbose : int
            PySCF logger verbosity.

    Saved results:

        e_dbbsc : float or None
            DBBSC ECMD correction energy after :meth:`kernel` is called.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.dbbsc import ecmd
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> myecmd = ecmd.ECMD(mf, functional='pbe')
    >>> myecmd.kernel()
    >>> print(myecmd.e_dbbsc)
    """

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
        aux_basis=None,
        functional='pbe',
        frozen=None,
        grids=None,
        ontop_model='ueg',
        max_memory=2000,
        verbose=None,
    ):
        """Initialize the DBBSC ECMD correction driver.

        Args:
            method : SCF or post-SCF object
                Reference object used for the DBBSC ECMD correction.

        Kwargs:
            aux_basis : str, basis dict, or None
                Auxiliary basis for density-fitted effective interactions.
            functional : str
                Correlation functional for the ECMD energy density.
            frozen : None, int, sequence, tuple, or str
                Frozen orbital selection.
            grids : :class:`pyscf.dft.gen_grid.Grids` or None
                Numerical integration grid.
            ontop_model : str
                On-top pair-density model.
            max_memory : float or None
                Memory limit in MB.  If ``None``, inherits the method memory
                setting.
            verbose : int or None
                PySCF logger verbosity.
        """
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
        aux_basis=None,
        functional=None,
        frozen=None,
        grids=None,
        ontop_model=None,
        max_memory=2000,
        verbose=None,
    ):
        """Compute the DBBSC ECMD correction and save it in ``self.e_dbbsc``.

        Kwargs:
            aux_basis : str, basis dict, or None
                Override ``self.aux_basis`` for this call.
            functional : str or None
                Override ``self.functional`` for this call.
            frozen : None, int, sequence, tuple, or str
                Override ``self.frozen`` for this call.
            grids : :class:`pyscf.dft.gen_grid.Grids` or None
                Override ``self.grids`` for this call.
            ontop_model : str or None
                Override ``self.ontop_model`` for this call.
            max_memory : float or None
                Override ``self.max_memory`` for this call.
            verbose : int or None
                Override ``self.verbose`` for this call.

        Returns:
            None
                The computed correction is stored in ``self.e_dbbsc``.
        """
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
