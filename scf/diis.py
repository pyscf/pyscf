#!/usr/bin/env python
# -*- coding: utf-8
#
# File: diis.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

import os
import cPickle as pickle
import tempfile

import numpy
import pyscf.lib
import pyscf.lib.logger as log

__author__ = "Qiming Sun <osirpt.sun@gmail.com>"
__version__ = "$ 0.1 $"


class DIIS:
# J. Mol. Struct. 114, 31-34
# PCCP, 4, 11
# GEDIIS JCTC, 2, 835
# C2DIIS IJQC, 45, 31
# DIIS try to minimize the change of the input vectors. It rotates the vectors
# to make the error vector as small as possible.
    def __init__(self, dev):
        self.verbose = dev.verbose
        self.fout = dev.fout
        self.diis_vec_stack = []
        self.threshold = 1e-6
        self.diis_space = 6
        self.diis_start_cycle = 2
        self._head = 0

    def push_vec(self, x):
        self.diis_vec_stack.append(x)
        if self.diis_vec_stack.__len__() > self.diis_space:
            self.diis_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self.diis_vec_stack[idx+1] - self.diis_vec_stack[idx]

    def get_vec(self, idx):
        return self.diis_vec_stack[idx+1]

    def get_num_diis_vec(self):
        return self.diis_vec_stack.__len__() - 1

    def update(self, x):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)

        nd = self.get_num_diis_vec()
        if nd < self.diis_start_cycle:
            if self.diis_start_cycle >= self.diis_space:
                self.diis_start_cycle = self.diis_space - 1
            return x

        H = numpy.ones((nd+1,nd+1), x.dtype)
        H[-1,-1] = 0
        G = numpy.zeros(nd+1, x.dtype)
        G[-1] = 1
        for i in range(nd):
            dti = self.get_err_vec(i)
            for j in range(i+1):
                dtj = self.get_err_vec(j)
                H[i,j] = numpy.dot(numpy.array(dti).flatten(), \
                                   numpy.array(dtj).flatten())
                H[j,i] = H[i,j].conj()

        # solve  H*x = G
        try:
            c_GH = numpy.linalg.solve(H, G)
        except numpy.linalg.linalg.LinAlgError:
            # damp diagonal elements to prevent singular
            for i in range(H.shape[0]):
                H[i,i] = H[i,i] + 1e-8
            c_GH = numpy.linalg.solve(H, G)
        #c_GH = pyscf.lib.solve_lineq_by_SVD(H, G)
        log.debug(self, 'diis-c %s', c_GH)

        x = numpy.zeros_like(x)
        for i, c in enumerate(c_GH[:-1]):
            x += self.get_vec(i) * c
        return x

class DIISLarge(DIIS):
    def __init__(self, dev, filename=None):
        import h5py
        DIIS.__init__(self, dev)
        if filename is None:
            tmp = tempfile.mktemp('.h5')
        else:
            tmp = filename
        self.diistmpfile = h5py.File(tmp, 'w')
        self._is_tmpfile_reused = False

    def __del__(self):
        try:
            self.diistmpfile.close()
        except:
            pass

    def push_vec(self, x):
        try:
            self.diistmpfile['/diis_vec/x%d' % self._head] = x
        except:
            self.diistmpfile['/diis_vec/x%d' % self._head][:] = x
        if self._head == self.diis_space:
            self._head = 0
            self._is_tmpfile_reused = True
        else:
            self._head += 1

    def get_err_vec(self, idx):
        if self._is_tmpfile_reused and idx == self._head:
            # x_head is not x_{head+1}-x_head
            return numpy.array(self.diistmpfile['/diis_vec/x0']) \
                    - numpy.array(self.diistmpfile['/diis_vec/x%d'%self.diis_space])
        else:
            return numpy.array(self.diistmpfile['/diis_vec/x%d'%(idx+1)]) \
                    - numpy.array(self.diistmpfile['/diis_vec/x%d'%idx])

    def get_vec(self, idx):
        if self._is_tmpfile_reused and idx == self._head:
            return numpy.array(self.diistmpfile['/diis_vec/x0'])
        else:
            return numpy.array(self.diistmpfile['/diis_vec/x%d'%(i+1)])

    def get_num_diis_vec(self):
        if self._is_tmpfile_reused:
            return self.diis_space
        else:
            return self._head - 1


# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class SCF_DIIS(DIIS):
    def __init__(self, dev):
        DIIS.__init__(self, dev)
        self.err_vec_stack = []

    def clear_diis_space(self):
        self.diis_vec_stack = []
        self.err_vec_stack = []

    def push_err_vec(self, s, d, f):
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = sdf.T.conj() - sdf
        log.debug(self, 'diis-norm(errvec) = %g', numpy.linalg.norm(errvec))

        self.err_vec_stack.append(errvec)
        if self.err_vec_stack.__len__() > self.diis_space:
            self.err_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self.err_vec_stack[idx]

    def get_vec(self, idx):
        return self.diis_vec_stack[idx]

    def get_num_diis_vec(self):
        return self.diis_vec_stack.__len__()

    def update(self, s, d, f):
        self.push_err_vec(s, d, f)
        return DIIS.update(self, f)

class DIISDamping(DIIS):
# Based on the given vector, minmize the previous error vectors.  Then mix the
# given vector with the optmized previous vector.
#TESTME
    def get_err_vec(self, idx, vec_ref):
        return self.diis_vec_stack[idx+1] - vec_ref

    def get_vec(self, idx):
        return self.diis_vec_stack[idx+1]

    def get_num_diis_vec(self):
        return self.diis_vec_stack.__len__() - 1

    def update(self, x, factor=.5):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)

        nd = self.get_num_diis_vec()
        if nd + 1 < self.diis_start_cycle:
            if self.diis_start_cycle >= self.diis_space:
                self.diis_start_cycle = self.diis_space - 1
            return x

        H = numpy.ones((nd,nd), x.dtype)
        H[-1,-1] = 0
        G = numpy.zeros(nd, x.dtype)
        G[-1] = 1
        for i in range(nd-1):
            dti = self.get_err_vec(i, x)
            for j in range(i+1):
                dtj = self.get_err_vec(j, x)
                H[i,j] = numpy.dot(numpy.array(dti).flatten(), \
                                   numpy.array(dtj).flatten())
                H[j,i] = H[i,j].conj()

        c_GH = self.diis_lineq(H, G)

        t = numpy.zeros_like(x)
        for i, c in enumerate(c_GH[:-1]):
            t += self.get_vec(i) * c
        x = x * factor + t * (1-factor)
        return x


if __name__ == '__main__':
    c = DIIS()
