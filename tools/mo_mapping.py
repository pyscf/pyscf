#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf.lib import logger
from pyscf import lo

def mo_map(mol1, mo1, mol2, mo2, base=1, tol=.5):
    '''Given two orbitals, based on their overlap <i|j>, search all
    orbital-pairs which have significant overlap.

    Returns:
        Two lists.  First list is the orbital-pair indices, second is the
        overlap value.
    '''
    s = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    s = reduce(numpy.dot, (mo1.T, s, mo2))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        logger.info(mol1, '<mo-1|mo-2>  %d  %d  %12.8f',
                    i+base, j+base, s[i,j])
    return idx, s

def mo_1to1map(s):
    '''Given <i|j>, search for the 1-to-1 mapping between i and j.

    Returns:
        a list [similar-j-to-i for i in <bra|]
    '''
    s1 = abs(s)
    like_input = []
    for i in range(s1.shape[0]):
        k = numpy.argmax(s1[i])
        like_input.append(k)
        s1[:,k] = 0
    return like_input

def mo_comps(test_or_idx, mol, mo_coeff, cart=False, orth_method='meta_lowdin'):
    '''Pick particular AOs based on the given test function to compute the AO
    contributions for each MO

    Args:
        test_or_idx : filter function or 1D array
            If test_or_idx is a list, it is considered as the AO indices.
            If it's a function,  the AO indices are the items for which the
            function value is true.

    Kwargs:
        cart : bool
            whether the orbital coefficients are based on cartesian basis.
        orth_method : str
            The localization method to generated orthogonal AO upon which the AO
            contribution are computed.  It can be one of 'meta_lowdin',
            'lowdin' or 'nao'.

    Returns:
        A list of float to indicate the total contributions (normalized to 1) of
        localized AOs

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.tools import mo_mapping
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    >>> mf = scf.RHF(mol).run()
    >>> comp = mo_mapping.mo_comps(lambda x: 'F 2s' in x, mol, mf.mo_coeff)
    >>> print('MO-id    components')
    >>> for i in enumerate(comp):
    ...     print('%-3d      %.10f' % (i, comp[i]))
    MO-id    components
    0        0.0000066344
    1        0.8796915532
    2        0.0590259826
    3        0.0000000000
    4        0.0000000000
    5        0.0435028851
    6        0.0155889103
    7        0.0000000000
    8        0.0000000000
    9        0.0000822361
    10       0.0021017982
    '''
    if cart:
        s = mol.intor_symmetric('cint1e_ovlp_cart')
        preao = lo.orth.pre_orth_ao(mol)
        tc2s = []
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            tc2s.append(gto.cart2sph(l))
        tc2s = scipy.linalg.block_diag(*tc2s)
        lao = lo.orth.orth_ao(mol, orth_method)
        lao = numpy.dot(tc2s, lao)
    else:
        s = mol.intor_symmetric('cint1e_ovlp_sph')
        lao = lo.orth.orth_ao(mol, orth_method)

    if callable(test_or_idx):
        idx = [i for i,x in enumerate(mol.spheric_labels(1)) if test(x)]
    else:
        idx = test_or_idx
    idx = numpy.asarray(idx)
    mo1 = reduce(numpy.dot, (lao[:,idx].T, s, mo_coeff))
    s1 = numpy.einsum('ki,ki->i', mo1, mo1)
    return s1

