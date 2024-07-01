#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
#         Zhi-Hao Cui <zhcui0408@gmail.com>

'''
Intrinsic Atomic Orbitals
ref. JCTC, 9, 4834
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import __config__
from pyscf.lo.orth import vec_lowdin
from pyscf.data.elements import is_ghost_atom

# Alternately, use ANO for minao
# orthogonalize iao with coefficients obtained by
#     vec_lowdin(iao_coeff, mol.intor('int1e_ovlp'))
MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

def iao(mol, orbocc, minao=MINAO, kpts=None, lindep_threshold=1e-8):
    '''Intrinsic Atomic Orbitals. [Ref. JCTC, 9, 4834]

    For large basis sets which are close to being linearly dependent,
    the Cholesky decomposition can fail. In this case a canonical orthogonalization
    with threshold `lindep_threshold` is used.

    Args:
        mol : molecule or cell object
        orbocc : 2D array
            occupied orbitals
        minao : str, optional
            reference basis set for IAOs
        kpts : 2D ndarray, optional
            k-points, for cell objects only
        lindep_threshold : float, optional
            threshold for canonical orthogonalization

    Returns:
        non-orthogonal IAO orbitals.  Orthogonalize them as C (C^T S C)^{-1/2},
        eg using :func:`orth.lowdin`

        >>> orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        >>> c = iao(mol, orbocc)
        >>> numpy.dot(c, orth.lowdin(reduce(numpy.dot, (c.T,s,c))))
    '''
    if mol.has_ecp() and minao == 'minao':
        logger.warn(mol, 'ECP/PP is used. MINAO is not a good reference AO basis in IAO.')

    pmol = reference_mol(mol, minao)
    # For PBC, we must use the pbc code for evaluating the integrals lest the
    # pbc conditions be ignored.
    has_pbc = getattr(mol, 'dimension', 0) > 0
    if has_pbc:
        from pyscf.pbc import gto as pbcgto
        s1 = mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s2 = pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s12 = pbcgto.cell.intor_cross('int1e_ovlp', mol, pmol, kpts=kpts)
    else:
        #s1 is the one electron overlap integrals (coulomb integrals)
        s1 = mol.intor_symmetric('int1e_ovlp')
        #s2 is the same as s1 except in minao
        s2 = pmol.intor_symmetric('int1e_ovlp')
        #overlap integrals of the two molecules
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)

    def make_iaos(s1, s2, s12, mo):
        """Make IAOs for a molecule or single k-point"""
        s21 = s12.conj().T
        # s2 is overlap in minimal reference basis and should never be singular:
        s2cd = scipy.linalg.cho_factor(s2)
        ctild = scipy.linalg.cho_solve(s2cd, numpy.dot(s21, mo))
        try:
            s1cd = scipy.linalg.cho_factor(s1)
            p12 = scipy.linalg.cho_solve(s1cd, s12)
            ctild = scipy.linalg.cho_solve(s1cd, numpy.dot(s12, ctild))
        # s1 can be singular in large basis sets: Use canonical orthogonalization in this case:
        except numpy.linalg.LinAlgError:
            x = scf.addons.canonical_orth_(s1, lindep_threshold)
            p12 = numpy.linalg.multi_dot((x, x.conj().T, s12))
            ctild = numpy.dot(p12, ctild)
        # If there are no occupied orbitals at this k-point, all but the first term will vanish:
        if mo.shape[-1] == 0:
            return p12
        ctild = vec_lowdin(ctild, s1)
        ccs1 = numpy.linalg.multi_dot((mo, mo.conj().T, s1))
        ccs2 = numpy.linalg.multi_dot((ctild, ctild.conj().T, s1))
        #a is the set of IAOs in the original basis
        a = (p12 + 2*numpy.linalg.multi_dot((ccs1, ccs2, p12))
             - numpy.dot(ccs1, p12) - numpy.dot(ccs2, p12))
        return a

    # Molecules and Gamma-point only solids
    if s1[0].ndim == 1:
        iaos = make_iaos(s1, s2, s12, orbocc)
    # Solid with multiple k-points
    else:
        iaos = []
        for k in range(len(kpts)):
            iaos.append(make_iaos(s1[k], s2[k], s12[k], orbocc[k]))
        iaos = numpy.asarray(iaos)
    return iaos

def reference_mol(mol, minao=MINAO):
    '''Create a molecule which uses reference minimal basis'''
    pmol = mol.copy()
    atoms = list(gto.format_atom(pmol.atom, unit=1))
    # remove ghost atoms
    pmol.atom = [atom for atom in atoms if not is_ghost_atom(atom[0])]
    if len(pmol.atom) != len(atoms):
        logger.info(mol, 'Ghost atoms found in system. '
                    'Current IAO does not support ghost atoms. '
                    'They are removed from IAO reference basis.')
    if getattr(pmol, 'rcut', None) is not None:
        pmol.rcut = None
    pmol.build(False, False, basis=minao)
    return pmol


def fast_iao_mullikan_pop(mol, dm, iaos, verbose=logger.DEBUG):
    '''
    Args:
        mol : the molecule or cell object

        iaos : 2D array
            (orthogonal or non-orthogonal) IAO orbitals

    Returns:
        mullikan population analysis in the basis IAO
    '''
    pmol = reference_mol(mol)
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        ovlpS = mol.pbc_intor('int1e_ovlp')
    else:
        ovlpS = mol.intor_symmetric('int1e_ovlp')

# Transform DM in big basis to IAO basis
# |IAO> = |big> C
# DM_{IAO} = C^{-1} DM (C^{-1})^T = S_{IAO}^{-1} C^T S DM S C S_{IAO}^{-1}
    cs = numpy.dot(iaos.T.conj(), ovlpS)
    s_iao = numpy.dot(cs, iaos)
    iao_inv = numpy.linalg.solve(s_iao, cs)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = reduce(numpy.dot, (iao_inv, dm, iao_inv.conj().T))
        return scf.hf.mulliken_pop(pmol, dm, s_iao, verbose)
    else:
        dm = [reduce(numpy.dot, (iao_inv, dm[0], iao_inv.conj().T)),
              reduce(numpy.dot, (iao_inv, dm[1], iao_inv.conj().T))]
        return scf.uhf.mulliken_pop(pmol, dm, s_iao, verbose)

del (MINAO)
