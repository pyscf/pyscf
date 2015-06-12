#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf.lib import logger
from pyscf import lo

def mo_map(mol1, mo1, mol2, mo2, base=1, tol=.5):
    s = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    s = reduce(numpy.dot, (mo1.T, s, mo2))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        logger.info(mol1, '<mo-1|mo-2>  %d  %d  %12.8f',
                    i+base, j+base, s[i,j])
    return idx, s

def mo_1to1map(s):
    ''' given <i|j>, [similar-j-to-i for i in <bra|]
    '''
    s1 = abs(s)
    like_input = []
    for i in range(s1.shape[0]):
        k = numpy.argmax(s1[i])
        like_input.append(k)
        s1[:,k] = 0
    return like_input

def mo_comps(test, mol, mo_coeff, cart=False, orth_method='meta_lowdin'):
    '''Pick particular AOs based on the given test function
    compute the AO components
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

    idx = [i for i,x in enumerate(mol.spheric_labels(1)) if test(x)]
    idx = numpy.array(idx)
    mo1 = reduce(numpy.dot, (lao[:,idx].T, s, mo_coeff))
    s1 = numpy.einsum('ki,ki->i', mo1, mo1)
    return s1


