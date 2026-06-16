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
Complementary auxiliary basis set (CABS).

Refs:
* JCC 127, 221106 (2007); DOI:10.1063/1.2817618
* JCP 128, 154103 (2008); DOI:10.1063/1.2889388
"""

import numpy
import scipy.linalg

from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lib import logger
from pyscf.scf import hf


def find_cabs(mol, auxmol, lindep=1e-8):
    """Project an auxiliary basis to the complement of the orbital basis."""
    cabs_mol = gto.conc_mol(mol, auxmol)
    nao = mol.nao_nr()
    s = cabs_mol.intor_symmetric('int1e_ovlp')

    ls12 = scipy.linalg.solve(s[:nao, :nao], s[:nao, nao:], assume_a='pos')
    s[nao:, nao:] -= s[nao:, :nao].dot(ls12)
    w, v = scipy.linalg.eigh(s[nao:, nao:])
    c2 = v[:, w > lindep] / numpy.sqrt(w[w > lindep])
    c1 = ls12.dot(c2)
    return cabs_mol, numpy.vstack((-c1, c2))


def make_cabs_auxmol(mol, auxbasis):
    """Build a basis-only Mole object for the CABS basis.

    The auxiliary functions must sit on the molecular centers, but they should
    not add another copy of the nuclear attraction operator when the OBS and
    CABS spaces are concatenated for one-electron matrix elements.
    """
    auxmol = mol.copy()
    auxmol.basis = auxbasis
    auxmol.build(False, False)
    auxmol._atm[:, gto.CHARGE_OF] = 0
    auxmol._ecpbas = auxmol._ecpbas[:0]
    return auxmol


def _as_cabs_auxmol(mol, auxmol_or_basis):
    if isinstance(auxmol_or_basis, gto.MoleBase):
        auxmol = auxmol_or_basis
        if not auxmol._built:
            auxmol.build(False, False)
        if (
            auxmol.natm == mol.natm
            and numpy.linalg.norm(auxmol.atom_coords() - mol.atom_coords()) < 1e-10
            and numpy.linalg.norm(auxmol.atom_charges()) > 1e-12
        ):
            return make_cabs_auxmol(mol, auxmol._basis)
        return auxmol
    return make_cabs_auxmol(mol, auxmol_or_basis)


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
            raise ValueError(f'Unsupported CABS frozen orbital scheme {frozen!r}')

    if isinstance(frozen, (bool, numpy.bool_)):
        raise TypeError('CABS frozen orbitals must be specified as an int, sequence, tuple, or named scheme')
    if isinstance(frozen, (int, numpy.integer)):
        mask[:frozen] = True
    else:
        mask[numpy.asarray(frozen, dtype=int)] = True
    return mask


def _active_masks(mol, mo_occ, frozen):
    frozen_mask = _frozen_mask(mol, mo_occ, frozen)
    occidx = (mo_occ > 0) & ~frozen_mask
    viridx = (mo_occ == 0) & ~frozen_mask
    return occidx, viridx


def _spin_masks(mol, spin_occ, frozen):
    if isinstance(frozen, (tuple, list)) and len(frozen) == 2 and not isinstance(frozen[0], (int, numpy.integer)):
        return tuple(_active_masks(mol, occ, frz) for occ, frz in zip(spin_occ, frozen))
    return tuple(_active_masks(mol, occ, frozen) for occ in spin_occ)


def _embed_dm(dm, nao, nca):
    dm = numpy.asarray(dm)
    dm_ext = numpy.zeros(dm.shape[:-2] + (nca, nca), dtype=dm.dtype)
    dm_ext[..., :nao, :nao] = dm
    return dm_ext


def _get_jk(mf, cabs_mol, dm):
    if getattr(mf, 'with_df', None) is not None:
        dfmf = scf.RHF(cabs_mol).density_fit(auxbasis=mf.with_df.auxbasis)
        dfmf.with_df.max_memory = mf.with_df.max_memory
        dfmf.with_df.stdout = mf.with_df.stdout
        dfmf.with_df.verbose = mf.with_df.verbose
        return dfmf.get_jk(cabs_mol, dm, hermi=1)
    return hf.get_jk(cabs_mol, dm, hermi=1)


def _unrestricted_focks(mf, cabs_mol, dm):
    vj, vk = _get_jk(mf, cabs_mol, dm)
    hcore = mf.get_hcore(cabs_mol)
    vj_tot = vj[0] + vj[1]
    return hcore + vj_tot - vk[0], hcore + vj_tot - vk[1]


def _extended_projector(mo_coeff, cabs_coeff):
    nao, nmo = mo_coeff.shape
    nca = cabs_coeff.shape[0]
    pcoeff = numpy.zeros((nca, nmo + cabs_coeff.shape[1]))
    pcoeff[:nao, :nmo] = mo_coeff
    pcoeff[:, nmo:] = cabs_coeff
    return pcoeff


def _cabs_singles_from_fock(fock, pcoeff, mo_occ, mo_energy, occidx, viridx):
    nmo = mo_occ.size
    if not numpy.any(occidx):
        return 0.0

    # Diagonalize the external space formed by orbital-basis virtual MOs and CABS.
    # The MO-virtual block is zero for canonical RHF/UHF, but gives the ROHF/non-canonical singles contribution,
    # and MolPro separates those contributions.
    extidx = numpy.r_[numpy.where(viridx)[0], numpy.arange(nmo, pcoeff.shape[1])]

    fock_p = pcoeff.T.dot(fock).dot(pcoeff)
    e_cabs, u_cabs = scipy.linalg.eigh(fock_p[numpy.ix_(extidx, extidx)])
    fia = fock_p[numpy.ix_(occidx, extidx)].dot(u_cabs)
    denom = mo_energy[occidx, None] - e_cabs
    return numpy.einsum('i,ia,ia,ia->', mo_occ[occidx], fia, fia, 1.0 / denom)


def energy_singles(mf, auxmol_or_basis, frozen='chemcore', lindep=1e-8):
    r"""CABS singles correction to the Hartree-Fock reference energy.

    For a closed-shell reference this evaluates

    .. math::
        E_\mathrm{CABS} = 2 \sum_{iA}
            \frac{|F_{iA}|^2}{\epsilon_i - \epsilon_A}

    where ``A`` denotes canonical orbitals in the external space formed by
    the virtual MOs of the orbital basis and CABS. For UHF and ROHF references
    the same expression is evaluated for the alpha and beta Fock matrices with
    spin occupations as prefactors.

    Args:
        mf : SCF object
            Converged molecular HF object.
        auxmol_or_basis : Mole, str, list, tuple, or dict
            CABS/OptRI basis as a Mole object or in the usual Mole.basis format.
            If a normal charged Mole is supplied on the same centers as ``mf``.
        frozen : None, int, sequence, tuple, or str
            Frozen orbital selection. ``'chemcore'`` (default) freezes the chemical
            core. ``None`` or ``0`` includes all orbitals. An integer freezes
            the lowest orbitals and a sequence freezes explicit MO indices. For
            UHF, a flat sequence is applied to both spins; a nested two-item
            sequence gives separate alpha and beta frozen orbitals.
        lindep : float
            Linear-dependence threshold in the CABS projection.
    """
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    is_uhf = isinstance(mo_coeff, (tuple, list)) or getattr(mo_coeff, 'ndim', 0) == 3
    is_rohf = not is_uhf and numpy.any(mo_occ == 1)

    if not is_uhf:
        valid_occ = (mo_occ == 0) | (mo_occ == 2)
        if is_rohf:
            valid_occ |= mo_occ == 1
        if numpy.any(~valid_occ):
            raise NotImplementedError('CABS singles for general fractional-occupation references is not implemented.')

    if is_rohf:
        spin_coeff = (mo_coeff, mo_coeff)
        spin_occ = (mo_occ > 0, mo_occ == 2)
        spin_energy = (mo_energy.mo_ea, mo_energy.mo_eb)
        spin_masks = _spin_masks(mol, spin_occ, frozen)
    elif is_uhf:
        spin_coeff = mo_coeff
        spin_occ = mo_occ
        spin_energy = mo_energy
        spin_masks = _spin_masks(mol, spin_occ, frozen)
    else:
        occidx, viridx = _active_masks(mol, mo_occ, frozen)

    auxmol = _as_cabs_auxmol(mol, auxmol_or_basis)
    cabs_mol, cabs_coeff = find_cabs(mol, auxmol, lindep)
    if cabs_coeff.shape[1] == 0:
        logger.info(mf, 'CABS singles correction = 0')
        return 0.0
    nao = mol.nao_nr()
    nca = cabs_mol.nao_nr()

    if is_rohf or is_uhf:
        dm = _embed_dm(mf.make_rdm1(), nao, nca)
        focks = _unrestricted_focks(mf, cabs_mol, dm)
        e_cabs = 0.0
        for fock, coeff, occ, energy, (occidx, viridx) in zip(focks, spin_coeff, spin_occ, spin_energy, spin_masks):
            pcoeff = _extended_projector(coeff, cabs_coeff)
            e_cabs += _cabs_singles_from_fock(fock, pcoeff, occ, energy, occidx, viridx)
    else:
        pcoeff = _extended_projector(mo_coeff, cabs_coeff)
        dm = _embed_dm(mf.make_rdm1(), nao, nca)
        vj, vk = _get_jk(mf, cabs_mol, dm)
        fock = mf.get_hcore(cabs_mol) + vj - vk * 0.5
        e_cabs = _cabs_singles_from_fock(fock, pcoeff, mo_occ, mo_energy, occidx, viridx)

    logger.info(mf, 'CABS singles correction = %.15g', e_cabs)
    return e_cabs


energy_cabs_singles = energy_singles
