#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>

'''
Intrinsic Atomic Orbitals
ref. JCTC, 9, 4834
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf import gto
from pyscf.pbc import gto as pbcgto


# Alternately, use ANO for minao
# orthogonalize iao by orth.lowdin(c.T*mol.intor(ovlp)*c)

def iao(mol, orbocc, minao='minao'):
    '''Intrinsic Atomic Orbitals. [Ref. JCTC, 9, 4834]

    Args:
        orbocc : 2D array
            occupied orbitals

    Returns:
        non-orthogal IAO orbitals.  Orthogonalize them as C (C^T S C)^{-1/2},
        eg using :func:`orth.lowdin`

        >>> orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        >>> c = iao(mol, orcc)
        >>> numpy.dot(c, orth.lowdin(reduce(numpy.dot, (c.T,s,c))))
    '''
    if mol.has_ecp():
        logger.warn('ECP/PP is used. MINAO is not a good reference AO basis in IAO.')

    #for PBC, we must use the pbc code for evaluating the integrals lest the pbc conditions be ignored
    if isinstance(mol, pbcgto.Cell):
        pmol = mol.copy()
        pmol.rcut = None
        pmol.build(False, False, basis=minao)
        s1 = mol.pbc_intor('int1e_ovlp_sph', hermi=1)
        s2 = pmol.pbc_intor('int1e_ovlp_sph', hermi=1)
        s12 = pbcgto.cell.intor_cross('int1e_ovlp_sph', mol, pmol)
    else:
        pmol = mol.copy()
        pmol.build(False, False, basis=minao)
#s1 is the one electron overlap integrals (coulomb integrals)
        s1 = mol.intor_symmetric('int1e_ovlp')
#s2 is the same as s1 except in minao 
        s2 = pmol.intor_symmetric('int1e_ovlp')
#overlap integrals of the two molecules 
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)
#transpose of overlap
    s21 = s12.T
    s1cd = scipy.linalg.cho_factor(s1)
    s2cd = scipy.linalg.cho_factor(s2)

    p12 = scipy.linalg.cho_solve(s1cd, s12)

    ctild = scipy.linalg.cho_solve(s2cd, numpy.dot(s21, orbocc))
    ctild = scipy.linalg.cho_solve(s1cd, numpy.dot(s12, ctild))
    ccs1 = reduce(numpy.dot, (orbocc, orbocc.T, s1))
    ccs2 = reduce(numpy.dot, (ctild, ctild.T, s1))
#a is the set of IAOs in the original basis
    a = (p12 + reduce(numpy.dot, (ccs1, ccs2, p12)) * 2
        - numpy.dot(ccs1, p12) - numpy.dot(ccs2, p12))
    return a

