'''
@author Linus Bjarne Dittmer

'''

import numpy
import numpy.linalg

import pyscf
import pyscf.scf


def vectorToMatrix(vector, assm=None):
    s = int(0.5 + (0.25 + 2 * len(vector))**0.5)
    if not type(assm) is numpy.ndarray:
        assm = getVectorMatrixIndices(s)
    mat = numpy.zeros((s, s))
    mat[assm[:,0],assm[:,1]] = vector

    return mat - mat.conj().T

def matrixToVector(matrix, assm=None):
    if not type(assm) is numpy.ndarray:
        assm = getVectorMatrixIndices(len(matrix))
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
    occ_num = int(round(subconverger._m3.mf.mol.nelectron / 2))
    assm = getVectorMatrixIndices(subconverger._m3.getDegreesOfFreedom())
    v_reda = []
    
    for i in range(len(vector)):
        if assm[i,0] >= occ_num and assm[i,1] < occ_num:
            v_reda.append(vector[i])

    return numpy.array(v_reda)

def decontractVectorFromVOSpace(subconverger, vector):
    occ_num = int(round(subconverger._m3.mf.mol.nelectron / 2))
    assm = getVectorMatrixIndices(subconverger._m3.getDegreesOfFreedom())
    counter = 0
    v_dred = numpy.zeros(subconverger._m3.getDegreesOfFreedom())

    for i in range(len(v_dred)):
        if assm[i,0] >= occ_num and assm[i,1] < occ_num:
            v_dred[i] = vector[counter]
            counter += 1

    return v_dred




