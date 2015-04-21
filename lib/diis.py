#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

import os, sys
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf.lib import logger


INCORE_SIZE = 1e7
BLOCK_SIZE  = int(20e6) # ~ 160/320 MB
# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
class DIIS:
    '''diis.space is the maximum of the allowed space
    diis.min_space is the minimal number of vectors to store before damping'''
    def __init__(self, dev=None, filename=None):
        if dev is not None:
            self.verbose = dev.verbose
            self.stdout = dev.stdout
        else:
            self.verbose = logger.INFO
            self.stdout = sys.stdout
        self.space = 6
        self.min_space = 1
        if not isinstance(filename, str):
            self._tmpfile = tempfile.NamedTemporaryFile()
            self.filename = self._tmpfile.name
            self._buffer = {}
            self._diisfile = h5py.File(self.filename)
            self._head = 0
            self._H = None
            self._xopt = None
            self._err_vec_touched = False
        else:
            self.filename = filename
            raise RuntimeError('TODO: initialized from given file %s' % filename)

    def __del__(self):
        self._diisfile.close()

    def push_err_vec(self, x):
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        key = 'e%d' % self._head
        if x.size < INCORE_SIZE:
            self._buffer[key] = x.ravel()
        elif key in self._diisfile:
            self._diisfile['e%d'%self._head][:] = x.ravel()
        else:
            self._diisfile['e%d'%self._head] = x.ravel()
        self._head += 1

    def push_vec(self, x):
        x = x.ravel()
        if self._H is None:
            self._H = numpy.ones((self.space+1,self.space+1), x.dtype)
            self._H[0,0] = 0

# we assumed that push_err_vec is called in advance, otherwise assuming
# generate error vector based on the given trial vectors
        if self._err_vec_touched:
            key = 'x%d' % (self._head - 1)
            if x.size < INCORE_SIZE:
                self._buffer[key] = x
            elif key in self._diisfile:
                self._diisfile[key][:] = x
            else:
                self._diisfile[key] = x
        elif self._xopt is None: # pass the first trial vectors
            self._xopt = x
        else:
            if self._head >= self.space:
                self._head = 0
            if x.size < INCORE_SIZE:
                self._buffer['e%d'%self._head] = x - self._xopt
                self._buffer['x%d'%self._head] = x
            else:
                if 'x%d' % self._head not in self._diisfile:
                    self._diisfile.create_dataset('e%d'%self._head, (x.size,), x.dtype)
                    self._diisfile.create_dataset('x%d'%self._head, (x.size,), x.dtype)
                self._diisfile['x%d'%self._head][:] = x
                for p0,p1 in prange(0, x.size, BLOCK_SIZE):
                    self._diisfile['e%d'%self._head][p0:p1] = x[p0:p1] - self._xopt[p0:p1]
            self._head += 1

    def get_err_vec(self, idx):
        if self._buffer:
            return self._buffer['e%d'%idx]
        else:
            return self._diisfile['e%d'%idx]

    def get_vec(self, idx):
        if self._buffer:
            return self._buffer['x%d'%idx]
        else:
            return self._diisfile['x%d'%idx]

    def get_num_vec(self):
        key = 'x%d'%(self.space-1)
        if self._buffer:
            if key in self._buffer:
                return self.space
            else:
                return self._head
        elif key in self._diisfile:
            return self.space
        else:
            return self._head

    def update(self, x, xerr=None):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        G = numpy.zeros(nd+1, x.dtype)
        G[0] = 1
        dt = numpy.array(self.get_err_vec(self._head-1), copy=False)
        for i in range(nd):
            tmp = 0
            for p0,p1 in prange(0, x.size, BLOCK_SIZE):
                tmp += numpy.dot(dt[p0:p1].conj(), self.get_err_vec(i)[p0:p1])
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conj()
        dt = None

        try:
            c = numpy.linalg.solve(self._H[:nd+1,:nd+1], G)
        except numpy.linalg.linalg.LinAlgError:
            logger.warn(self, 'singularity in diis')
            H = self._H[:nd+1,:nd+1].copy()
            for i in range(1,nd):
                H[i,i] += 1e-10
            c = numpy.linalg.solve(H, G)
        logger.debug1(self, 'diis-c %s', c)

        self._xopt = None
        self._xopt = x.ravel() * c[1]
        for i, ci in enumerate(c[2:]):
            for p0,p1 in prange(0, x.size, BLOCK_SIZE):
                self._xopt[p0:p1] += self.get_vec(i+1)[p0:p1] * ci
        return self._xopt.reshape(x.shape)

#class CDIIS
#class EDIIS
#class GDIIS

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    c = DIIS()
