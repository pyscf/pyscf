#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

import numpy
import scipy.linalg
import ctf
import sys
from pyscf.lib import logger

"""DIIS solver for CTF, adapted from pyscf.lib.diis, all vectors stored in memory"""

class DIIS(object):
    def __init__(self, dev=None):
        if dev is not None:
            self.verbose = dev.verbose
            self.stdout = dev.stdout
        else:
            self.verbose = 4
            self.stdout = sys.stdout
        self.space = 6
        self.min_space = 1
##################################################
# don't modify the following private variables, they are not input options
        self._diisfile = None
        self._buffer = {}
        self._bookkeep = [] # keep the ordering of input vectors
        self._head = 0
        self._H = None
        self._xprev = None
        self._err_vec_touched = False

    def _store(self, key, value):
        self._buffer[key] = value

    def push_err_vec(self, xerr):
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        key = 'e%d' % self._head
        self._store(key, xerr.ravel())

    def push_vec(self, x):
        x = x.ravel()

        while len(self._bookkeep) >= self.space:
            self._bookkeep.pop(0)

        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            key = 'x%d' % (self._head)
            self._store(key, x)
            self._head += 1

        elif self._xprev is None:
# If push_err_vec is not called in advance, the error vector is generated
# as the diff of the current vec and previous returned vec (._xprev)
# So store the first trial vec as the previous returned vec
            self._xprev = x
            self._store('xprev', x)

        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d'%self._head
            xkey = 'x%d'%self._head
            self._store(xkey, x)
            self._store(ekey, x - ctf.astensor(self._xprev))
            self._head += 1

    def get_err_vec(self, idx):
        return self._buffer['e%d'%idx]

    def get_vec(self, idx):
        return self._buffer['x%d'%idx]

    def get_num_vec(self):
        return len(self._bookkeep)

    def update(self, x, xerr=None):
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x
        dt = ctf.astensor(self.get_err_vec(self._head-1))
        #_H small, dt and xnew, x, are all large
        if self._H is None:
            self._H = numpy.zeros((self.space+1,self.space+1), x.dtype)
            self._H[0,1:] = self._H[1:,0] = 1
        for i in range(nd):
            tmp = 0
            dti = self.get_err_vec(i)
            tmp += ctf.dot(dt.conj(), dti)
            tmp = tmp.to_nparray()
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conj()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = self.extrapolate(nd)

            self._store('xprev', xnew)
        return xnew.reshape(x.shape)

    def extrapolate(self, nd=None):

        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError('No vector found in DIIS object.')

        h = self._H[:nd+1,:nd+1]
        g = numpy.zeros(nd+1, h.dtype)
        g[0] = 1

        w, v = scipy.linalg.eigh(h)
        if numpy.any(abs(w)<1e-14):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w)>1e-14
            c = numpy.dot(v[:,idx]*(1./w[idx]), numpy.dot(v[:,idx].T.conj(), g))
        else:
            try:
                c = numpy.linalg.solve(h, g)
            except numpy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = ctf.zeros(xi.size, dtype=c.dtype)
            xnew += xi * ci
        return xnew
