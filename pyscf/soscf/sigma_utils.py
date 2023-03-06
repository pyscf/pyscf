'''
@author Linus Bjarne Dittmer

'''

import numpy
import numpy.linalg

def vec_to_matrix(vector, assm=None):
    s = int(0.5 + (0.25 + 2 * len(vector))**0.5)
    if not isinstance(assm, numpy.ndarray):
        assm = get_vec_matrix_indices(s)
    mat = numpy.zeros((s, s))
    mat[assm[:,0],assm[:,1]] = vector
    return mat - mat.conj().T

def matrix_to_vec(matrix, assm=None):
    if not isinstance(assm, numpy.ndarray):
        assm = get_vec_matrix_indices(len(matrix))
    vec = matrix[assm[:,0],assm[:,1]]
    return vec

def get_vec_matrix_indices(size):
    numberOfVariables = int(0.5 * size * (size-1))
    assignment = numpy.zeros((numberOfVariables, 2), dtype=numpy.int32)
    counter = 0
    for i in range(1, size):
        for j in range(0, i):
            assignment[counter,0] = i
            assignment[counter,1] = j
            counter += 1
    return assignment
