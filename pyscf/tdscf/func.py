# TODO: refactor the code before adding to FEATURES list by PySCF-1.5 release
# 1. code style
#   * Indent: 8 -> 4
#   * Function/method should be all lowercase
#   * Use either double quote or single quote, not mix
#

import numpy as np


def TransMat(M,U,inv = 1):
        if inv == 1:
                # U.t() * M * U
                Mtilde = np.dot(np.dot(U.T.conj(),M),U)
        elif inv == -1:
                # U * M * U.t()
                Mtilde = np.dot(np.dot(U,M),U.T.conj())
        return Mtilde

def TrDot(A,B):
        C = np.trace(np.dot(A,B))
        return C

def MatrixPower(A,p,PrintCondition=False):
        ''' Raise a Hermitian Matrix to a possibly fractional power. '''
        u,s,v = np.linalg.svd(A)
        if (PrintCondition):
                print "MatrixPower: Minimal Eigenvalue =", np.min(s)
        for i in range(len(s)):
                if (abs(s[i]) < np.power(10.0,-14.0)):
                        s[i] = np.power(10.0,-14.0)
        return np.dot(u,np.dot(np.diag(np.power(s,p)),v))
