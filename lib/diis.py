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
import h5py
import pyscf.lib.logger as log


# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
class DIIS:
    '''diis.space is the maximum of the allowed space
    diis.min_space is the minimal number of vectors to store before damping'''
    def __init__(self, dev, filename=None):
        self.verbose = dev.verbose
        self.stdout = dev.stdout
        self.space = 6
        self.min_space = 1
        self._vec = []
        self._err_vec = []
        if not isinstance(filename, str):
            self._tmpfile = tempfile.NamedTemporaryFile()
            self.filename = self._tmpfile.name
        else:
            self.filename = filename
        self._diisfile = h5py.File(self.filename)
        self._count = 0
        self._H = None

    def __del__(self):
        self._diisfile.close()

    def push_err_vec(self, x):
        if len(self._err_vec) >= self.space:
            i = self._err_vec.pop(0)
            del(self._diisfile[i])
        self._err_vec.append('e%d'%self._count)
        self._diisfile[self._err_vec[-1]] = x
        self._count += 1

    def push_vec(self, x):
        if self._H is None:
            self._H = numpy.ones((self.space+1,self.space+1), x.dtype)
            self._H[0,0] = 0

        if len(self._vec) >= self.space:
            if (len(self._err_vec) > 0 or
                len(self._vec) >= self.space): # one more vector to compute error vector
                i = self._vec.pop(0)
                del(self._diisfile[i])
                self._H[1:self.space,1:self.space] = self._H[2:,2:]
        self._vec.append('x%d'%self._count)
        self._diisfile[self._vec[-1]] = x
        self._count += 1

    def get_err_vec(self, idx):
        if len(self._err_vec) == 0:
            return (numpy.array(self._diisfile[str(self._vec[idx+1])])
                    - numpy.array(self._diisfile[str(self._vec[idx])]))
        else:
            return numpy.array(self._diisfile[str(self._err_vec[idx])])

    def get_vec(self, idx):
        if len(self._err_vec) == 0:
            return numpy.array(self._diisfile[str(self._vec[idx+1])])
        else:
            return numpy.array(self._diisfile[str(self._vec[idx])])

    def get_num_vec(self):
        if len(self._err_vec) == 0:
            return len(self._vec) - 1
        else:
            assert(len(self._vec) == len(self._err_vec))
            return len(self._vec)

    def update(self, x):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        G = numpy.zeros(nd+1, x.dtype)
        G[0] = 1
        dt = self.get_err_vec(nd-1)
        for i in range(nd):
            di = self.get_err_vec(i)
            self._H[nd,i+1] = numpy.dot(numpy.array(dt).ravel(),
                                        numpy.array(di).ravel())
            self._H[i+1,nd] = self._H[nd,i+1].conj()

        try:
            c = numpy.linalg.solve(self._H[:nd+1,:nd+1], G)
        except numpy.linalg.linalg.LinAlgError:
            log.warn(self, 'singularity in diis')
            H = self._H[:nd+1,:nd+1].copy()
            for i in range(1,nd):
                H[i,i] += 1e-10
            c = numpy.linalg.solve(H, G)
        log.debug1(self, 'diis-c %s', c)

        x = numpy.zeros_like(x)
        for i, ci in enumerate(c[1:]):
            x += self.get_vec(i) * ci
        return x

#class CDIIS
#class EDIIS
#class GDIIS


if __name__ == '__main__':
    c = DIIS()
