#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

import os
import tempfile
from functools import reduce
import numpy
import pyscf.lib.logger as log


class DIIS:
# J. Mol. Struct. 114, 31-34
# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
# DIIS try to minimize the change of the input vectors. It rotates the vectors
# to minimize the error in the least square sense.
    '''
diis.space is the maximum of the allowed space
diis.min_space is the minimal number of vectors to store before damping'''
    def __init__(self, dev):
        self.verbose = dev.verbose
        self.stdout = dev.stdout
        self._vec_stack = []
        self.conv_tol = 1e-6
        self.space = 6
        self.min_space = 1

    def push_vec(self, x):
        self._vec_stack.append(x)
        if self._vec_stack.__len__() > self.space:
            self._vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self._vec_stack[idx+1] - self._vec_stack[idx]

    def get_vec(self, idx):
        return self._vec_stack[idx+1]

    def get_num_diis_vec(self):
        return self._vec_stack.__len__() - 1

    def update(self, x):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)

        nd = self.get_num_diis_vec()
        if nd <= self.min_space:
            return x

        H = numpy.ones((nd+1,nd+1), x.dtype)
        H[0,0] = 0
        G = numpy.zeros(nd+1, x.dtype)
        G[0] = 1
        for i in range(nd):
            dti = self.get_err_vec(i)
            for j in range(i+1):
                dtj = self.get_err_vec(j)
                H[i+1,j+1] = numpy.dot(numpy.array(dti).ravel(), \
                                       numpy.array(dtj).ravel())
                H[j+1,i+1] = H[i+1,j+1].conj()

        try:
            c = numpy.linalg.solve(H, G)
        except numpy.linalg.linalg.LinAlgError:
            log.warn(self, 'singularity in diis')
            #c = pyscf.lib.solve_lineq_by_SVD(H, G)
            ## damp diagonal elements to avoid singularity
            #for i in range(H.shape[0]):
            #    H[i,i] = H[i,i] + 1e-9
            #c = numpy.linalg.solve(H, G)
            for i in range(1,nd):
                H[i,i] = H[i,i] + 1e-11
            c = numpy.linalg.solve(H, G)
            #c = numpy.linalg.solve(H[:nd,:nd], G[:nd])
        log.debug1(self, 'diis-c %s', c)

        x = numpy.zeros_like(x)
        for i, ci in enumerate(c[1:]):
            x += self.get_vec(i) * ci
        return x

class DIISLarge(DIIS):
    def __init__(self, dev, filename=None):
        import h5py
        DIIS.__init__(self, dev)
        if filename is None:
            self._tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
            filename = self._tmpfile.name
        self.diisfile = h5py.File(filename, 'w')
        self._count = 0
        self._head = 0
        self._is_tmpfile_reused = False

    def push_vec(self, x):
        key = 'x%d' % (self._count % self.space)
        if self._count < self.space:
            self.diisfile[key] = x
        else:
            self.diisfile[key][:] = x
        self._count += 1

    def get_err_vec(self, idx):
        if self._count < self.space:
            return numpy.array(self.diisfile['x%d'%(idx+1)]) \
                 - numpy.array(self.diisfile['x%d'%idx])
        else:
# the previous vector may refer to the last one in circular store
            last_id = (idx+self.space-1) % self.space
            return numpy.array(self.diisfile['x%d'%idx]) \
                 - numpy.array(self.diisfile['x%d'%last_id])

    def get_vec(self, idx):
        if self._count < self.space:
            return numpy.array(self.diisfile['x%d'%(idx+1)])
        else:
            return numpy.array(self.diisfile['x%d'%idx])

    def get_num_diis_vec(self):
        return len(self.diisfile.keys()) - 1


# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class SCF_DIIS(DIIS):
    def __init__(self, dev):
        DIIS.__init__(self, dev)
        self.err_vec_stack = []
        self.start_cycle = 3
        self.space = 8

    def clear_diis_space(self):
        self._vec_stack = []
        self.err_vec_stack = []

    def push_err_vec(self, s, d, f):
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = sdf.T.conj() - sdf
        log.debug1(self, 'diis-norm(errvec) = %g', numpy.linalg.norm(errvec))

        self.err_vec_stack.append(errvec)
        if self.err_vec_stack.__len__() > self.space:
            self.err_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self.err_vec_stack[idx]

    def get_vec(self, idx):
        return self._vec_stack[idx]

    def get_num_diis_vec(self):
        return self._vec_stack.__len__()

    def update(self, s, d, f):
        self.push_err_vec(s, d, f)
        return DIIS.update(self, f)

#TODO
def with_inits(inits, DiisClass):
    def fn(*args):
        adiis = DiisClass(args)
        adiis._vec_stack = inits
        return adiis
    return fn


if __name__ == '__main__':
    c = DIIS()
