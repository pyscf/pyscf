#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
#
# Author: Nikolay A. Bogdanov <n.bogdanov@inbox.lv>
#

from functools import reduce
import numpy
import libmsym as msym
from pyscf.lib import param
from pyscf.lib import logger
from pyscf.data import elements
from pyscf import symm
from pyscf.lo.orth import vec_lowdin


def sph_mval(label):
    '''Get integer l_z value of given spherical AO label'''
    m_list = {'': 0,
              'x': 1, 'y': -1, 'z': 0,
              'xy': -2, 'yz': -1, 'z^2': 0, 'xz': 1, 'x2-y2': 2}
    try:
        m = int(label)
    except(ValueError):
        m = m_list[label]
    return m


def basis_info(ao_labels):
    info = [[id, int(nl[:-1]), param.ANGULARMAP[nl[-1]], sph_mval(m_label)]
            for (id, elem, nl, m_label) in ao_labels]
    return info


def msym_basis(ao_labels, msym_atoms):
    '''Generate list of basis functions for libmsym from pyscf data

    Given pyscf Mole object mol use as follows:
    msym_basis(mol.ao_labels(fmt=False), msym_atoms(mol._atom))
    '''
    basis_functions = [msym.RealSphericalHarmonic(element=msym_atoms[id],
                                                  n=int(nl[:-1]),
                                                  l=param.ANGULARMAP[nl[-1]],
                                                  m=sph_mval(m_label))
                       for (id, elem, nl, m_label) in ao_labels]
    return basis_functions


def msym_atoms(_atom):
    '''Generate list of atoms for libmsym from pyscf mol._atom'''
    atoms = [msym.Element(coordinates=coord,
                          charge=elements.charge(elem))
             for elem, coord in _atom]
    return atoms


def log_symmetry_info(mol):
    if mol.topgroup == mol.groupname:
        logger.info(mol, 'point group symmetry = %s', mol.topgroup)
    else:
        logger.info(mol, 'point group symmetry = %s, use subgroup %s',
                    mol.topgroup, mol.groupname)
    for ir in range(mol.symm_orb.__len__()):
        logger.info(mol, 'num. orbitals of irrep %s = %d',
                    mol.irrep_name[ir], mol.symm_orb[ir].shape[1])


def gen_mol_msym(mol, tol=1e-14, verbose=True):
    mol_msym = mol.copy()
    if verbose:
        logger.info(mol_msym, '\n*** Use libmsym to generate symmetry-adapted AOs ***')
        logger.info(mol_msym, 'tolerance used to detect symmetry = %s', tol)
    prop_atoms = mol_msym.format_atom(mol_msym._atom,
                                      mol_msym._symm_orig,
                                      mol_msym._symm_axes,
                                      'Angstrom')
    atoms_msym_fmt = msym_atoms(prop_atoms)
    with msym.Context(elements=atoms_msym_fmt,
                      basis_functions=msym_basis(mol_msym.ao_labels(fmt=False),
                                                 atoms_msym_fmt)) as ctx:
        ctx.set_thresholds(zero=tol, geometry=tol, angle=tol,
                           equivalence=tol, eigfact=tol, permutation=tol,
                           orthogonalization=tol)
        mol_msym.groupname = ctx.find_symmetry()
        # character_table = ctx.character_table.table
        irrep_name = [irrep.name for irrep in ctx.character_table.symmetry_species]
        irrep_dim = [irrep.dim for irrep in ctx.character_table.symmetry_species]
        (salc_mo_coeff, salc_symm_id, salc_partner_func) = ctx.salcs
        salc_mo_coeff = salc_mo_coeff.T
        salc_partner_id = [pf.dim for pf in salc_partner_func]
        # partners_by_irrep = [(id, numpy.sort(numpy.unique(salc_partner_id[salc_symm_id==id])))
        #                      for id in numpy.unique(salc_symm_id)]
        labels = numpy.array([irrep_name[idx]
                              + ("_" + str(pf))*(irrep_dim[idx] > 1)  # add partner id for dim > 1
                              for idx, pf in zip(salc_symm_id, salc_partner_id)])
        mol_msym.irrep_name = list(numpy.unique(labels))
        mol_msym.irrep_id = list(range(len(mol_msym.irrep_name)))
        mol_msym.symm_orb = [salc_mo_coeff[:, labels == i_label] for i_label in mol_msym.irrep_name]
        # in-place rotate SALCs to the original geometry
        l_idx = symm.basis.ao_l_dict(mol_msym)
        Ds = symm.basis._momentum_rotation_matrices(mol_msym, mol_msym._symm_axes)
        for c_ir in mol_msym.symm_orb:
            nso = c_ir.shape[1]
            for l, idx in l_idx.items():
                c = c_ir[idx].reshape(-1,Ds[l].shape[1],nso)
                c_ir[idx] = numpy.einsum('nm,smp->snp', Ds[l], c).reshape(-1,nso)
    if verbose:
        log_symmetry_info(mol_msym)
    return mol_msym


def symmetrize_space_pyscf(mol, mo, s=None, orthonormalize=False):
    '''Symmetrize MOs with pyscf function but symmetry generated by libmsym'''
    mol_msym = gen_mol_msym(mol)
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    mo_msym = symm.symmetrize_space(mol_msym, mo, s=s)
    if orthonormalize:
        mo_msym = vec_lowdin(mo_msym, s=s)
    return mo_msym


def symmetrize_space_libmsym(mol, mo, tol=1e-14, check=True, orthonormalize=False):
    '''Symmetrize MOs with libmsym itself'''
    if check:
        s = mol.intor_symmetric('int1e_ovlp')
        nmo = mo.shape[1]
        # s_mo = numpy.dot(s, mo_msym)
        max_non_orth = abs(reduce(numpy.dot, (mo.conj().T, s, mo)) - numpy.eye(nmo)).max()
        if max_non_orth > tol:
            if orthonormalize:
                print('Input orbitals are not orthogonalized: '+str(max_non_orth))
                mo = vec_lowdin(mo, s=s)
                max_non_orth = abs(reduce(numpy.dot, (mo.conj().T, s, mo)) - numpy.eye(nmo)).max()
                print("Input orbitals were orthonormalized: " + str(max_non_orth))
            else:
                raise ValueError('Input orbitals are not orthogonalized: '+str(max_non_orth))
    # mol_msym = gen_mol_msym(mol)
    my_msym_atoms = msym_atoms(mol._atom)
    with msym.Context(elements=my_msym_atoms,
                      basis_functions=msym_basis(mol.ao_labels(fmt=False), my_msym_atoms)) as ctx:
        ctx.set_thresholds(zero=tol, geometry=tol, angle=tol,
                           equivalence=tol, eigfact=tol, permutation=tol,
                           orthogonalization=tol)
        # print(ctx.find_symmetry())
        sym_mo_coeff, sym_symm_id, symm_partner_func = ctx.symmetrize_wavefunctions(mo.T)
    mo_msym = sym_mo_coeff.T
    if check:
        nmo = mo_msym.shape[1]
        max_non_orth = abs(
            reduce(numpy.dot, (mo_msym.conj().T, s, mo_msym)) - numpy.eye(nmo)).max()
        if max_non_orth > tol:
            if orthonormalize:
                mo_msym = vec_lowdin(mo_msym, s=s)
                max_non_orth = abs(
                    reduce(numpy.dot, (mo_msym.conj().T, s, mo_msym)) - numpy.eye(nmo)).max()
            else:
                raise ValueError('Output orbitals are not orthogonalized: ' + str(max_non_orth))
    return mo_msym


symmetrize_space = symmetrize_space_pyscf
