'''
@author Linus Bjarne Dittmer

'''

from functools import reduce

import numpy
import numpy.linalg
import scipy

import pyscf
import pyscf.scf

import pyscf.soscf.newton_ah


def coupleSOSCF(soscf):
    '''
    Method used for coupling an instance of M3SOSCF to the Sigma utils. This serves the purpose of automatically storing important variables for Sigma Utils inside the M3SOSCF instance as to avoid mixing.

    Arguments:
        soscf: pyscf.scf.m3soscf.M3SOSCF
            The M3SOSCF class that is to be coupled.
    '''

    #soscf._int2e = pyscf.ao2mo.addons.restore('s1', pyscf.ao2mo.kernel(soscf._mf.mol, soscf._moCoeffs), soscf._mf.mol.nao_nr())
    #soscf._int2e = pyscf.ao2mo.kernel(soscf._mf.mol, soscf._moCoeffs)
    pass

def decoupleSOSCF(soscf):
    '''
    Method used for decoupling an instance of M3SOSCF to the Sigma utils. This serves the purpose of freeing storage space.

    Arguments:
        soscf: pyscf.scf.m3soscf.M3SOSCF
            The M3SOSCF class that is to be decoupled.
    '''

    soscf._int2e = None



def getCanonicalBasis(size):
    '''
    Returns the canonical basis for an anti-hermitian matrix.

    Arguments:
        matrix: 2D array
            A square matrix that determines the size of the basis

    Returns:
        basis: 3D array
            The canonical basis for an anti-hermitian matrix of size (n,n). Tensor of shape ( n(n-1)/2, n, n ). Contraction over the first axis with a vector of size ( n(n-1)/2, ) yields an anti-hermitian matrix.

    Examples:
    >>> n = 3
    >>> basis = getCanonicalBasis(numpy.zeros((n,n)))
    >>> print(basis)
    [[[ 0. -1.  0.]
      [ 1.  0.  0.]
      [ 0.  0.  0.]]
     [[ 0.  0. -1.]
      [ 0.  0.  0.]
      [ 1.  0.  0.]]
     [[ 0.  0.  0.]
      [ 0.  0. -1.]
      [ 0.  1.  0.]]]
    >>> coeffs = numpy.arange(1, n+1)
    >>> ah_matrix = numpy.einsum('kij,k->ij', basis, coeffs)
    >>> print(ah_matrix)
    [[ 0. -1. -2.]
     [ 1.  0. -3.]
     [ 2.  3.  0.]]
    '''
    
    numberOfVariables = int(0.5 * size * (size-1))
    basis = numpy.zeros((numberOfVariables, size, size))
    counter = 0
    for i in range(1, size):
        for j in range(0, i):
            basis[counter][i][j] = 1
            basis[counter][j][i] = -1

            counter += 1

    return basis


def vectorToMatrix(vector, assm=None):
    s = int(0.5 + (0.25 + 2 * len(vector))**0.5)
    if not type(assm) is numpy.ndarray:
        assm = getVectorMatrixIndices(s)
    mat = numpy.zeros((s, s))
    mat[assm[:,0],assm[:,1]] = vector

    return mat - mat.conj().T

def matrixToVector(matrix, assm=None):
    #s = int(len(matrix) * (len(matrix)-1) * 0.5)
    if not type(assm) is numpy.ndarray:
        assm = getVectorMatrixIndices(len(matrix))
    #vec = numpy.zeros(s)
    vec = matrix[assm[:,0],assm[:,1]]
    return vec
    


def getVectorMatrixIndices(size):
    numberOfVariables = int(0.5 * size * (size-1))
    assignment = numpy.zeros((numberOfVariables, 2), dtype=numpy.int32)
    counter = 0
    for i in range(1, size):
        for j in range(0, i):
            assignment[counter,0] = i
            assignment[counter,1] = j
            counter += 1


    return assignment

def contractVectorToVOSpace(subconverger, vector):
    occ_num = int(round(subconverger._m3._mf.mol.nelectron / 2))
    assm = getVectorMatrixIndices(subconverger._m3.getDegreesOfFreedom())
    v_reda = []
    
    for i in range(len(vector)):
        if assm[i,0] >= occ_num and assm[i,1] < occ_num:
            v_reda.append(vector[i])

    return numpy.array(v_reda)

def decontractVectorFromVOSpace(subconverger, vector):
    occ_num = int(round(subconverger._m3._mf.mol.nelectron / 2))
    assm = getVectorMatrixIndices(subconverger._m3.getDegreesOfFreedom())
    counter = 0
    v_dred = numpy.zeros(subconverger._m3.getDegreesOfFreedom())

    for i in range(len(v_dred)):
        if assm[i,0] >= occ_num and assm[i,1] < occ_num:
            v_dred[i] = vector[counter]
            counter += 1

    return v_dred




