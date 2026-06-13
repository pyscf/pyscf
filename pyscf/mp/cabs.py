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

from pyscf import gto
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


def _cabs_singles_from_fock(fock, cabs_coeff, mo_coeff, mo_occ, mo_energy):
    occidx = mo_occ > 0
    if not numpy.any(occidx):
        return 0.0

    fcabs = numpy.dot(fock, cabs_coeff)
    e_cabs, u_cabs = scipy.linalg.eigh(numpy.dot(cabs_coeff.T, fcabs))
    fia = numpy.dot(numpy.dot(mo_coeff[:, occidx].T, fcabs[: mo_coeff.shape[0]]), u_cabs)
    denom = mo_energy[occidx, None] - e_cabs
    return numpy.einsum('i,ia,ia,ia->', mo_occ[occidx], fia, fia, 1.0 / denom)


def energy_singles(mf, auxmol_or_basis, lindep=1e-8):
    r"""CABS singles correction to the Hartree-Fock reference energy.

    For a closed-shell reference this evaluates

    .. math::
        E_\mathrm{CABS} = 2 \sum_{iA}
            \frac{|F_{iA}|^2}{\epsilon_i - \epsilon_A}

    where ``A`` denotes canonical orbitals in the complementary auxiliary
    basis space. For UHF references the same expression is evaluated for the
    alpha and beta Fock matrices with spin occupations as prefactors.

    Args:
        mf : SCF object
            Converged molecular HF object.
        auxmol_or_basis : Mole, str, list, tuple, or dict
            CABS/OptRI basis as a Mole object or in the usual Mole.basis format.
            If a normal charged Mole is supplied on the same centers as ``mf``,
            it is converted to ghost centers to avoid doubled nuclei.
        lindep : float
            Linear-dependence threshold in the CABS projection.
    """
    mol = mf.mol
    auxmol = _as_cabs_auxmol(mol, auxmol_or_basis)
    cabs_mol, cabs_coeff = find_cabs(mol, auxmol, lindep)
    if cabs_coeff.shape[1] == 0:
        logger.info(mf, 'CABS singles correction = 0')
        return 0.0
    nao = mol.nao_nr()
    nca = cabs_mol.nao_nr()

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    is_uhf = isinstance(mo_coeff, (tuple, list)) or getattr(mo_coeff, 'ndim', 0) == 3
    if is_uhf:
        moa, mob = mo_coeff
        dma, dmb = mf.make_rdm1()
        dm = numpy.zeros((2, nca, nca))
        dm[0, :nao, :nao] = dma
        dm[1, :nao, :nao] = dmb
        vj, vk = hf.get_jk(cabs_mol, dm, hermi=1)
        hcore = mf.get_hcore(cabs_mol)
        focka = hcore + vj[0] + vj[1] - vk[0]
        fockb = hcore + vj[0] + vj[1] - vk[1]
        e_cabs = _cabs_singles_from_fock(focka, cabs_coeff, moa, mo_occ[0], mo_energy[0])
        e_cabs += _cabs_singles_from_fock(fockb, cabs_coeff, mob, mo_occ[1], mo_energy[1])
    else:
        if numpy.any((mo_occ != 0) & (mo_occ != 2)):
            raise NotImplementedError(
                'CABS singles for ROHF/general open-shell references is not implemented. Use a UHF reference.'
            )
        dm = numpy.zeros((nca, nca))
        dm[:nao, :nao] = mf.make_rdm1()
        vj, vk = hf.get_jk(cabs_mol, dm, hermi=1)
        fock = mf.get_hcore(cabs_mol) + vj - vk * 0.5
        e_cabs = _cabs_singles_from_fock(fock, cabs_coeff, mo_coeff, mo_occ, mo_energy)

    logger.info(mf, 'CABS singles correction = %.15g', e_cabs)
    return e_cabs


energy_cabs_singles = energy_singles
