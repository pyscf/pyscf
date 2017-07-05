#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

import sys
import tempfile
import numpy
import scipy.linalg
import h5py
from . import parameters
from . import logger
from . import misc


INCORE_SIZE = 1e7
BLOCK_SIZE  = int(20e6) # ~ 160/320 MB
# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
class DIIS(object):
    '''Direct inversion in the iterative subspace method.

    Attributes:
        space : int
            DIIS subspace size. The maximum number of the vectors to be stored.
        min_space
            The minimal size of subspace before DIIS extrapolation.

    Functions:
        update(x, xerr=None) :
            If xerr the error vector is given, this function will push the target
            vector and error vector in the DIIS subspace, and use the error vector
            to extrapolate the vector and return the extrapolated vector.
            If xerr is None, this function will take the difference between
            the current given vector and the last given vector as the error
            vector to extrapolate the vector.

    Examples:

    >>> from pyscf import gto, scf, lib
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol)
    >>> h = mf.get_hcore()
    >>> s = mf.get_ovlp()
    >>> e, c = mf.eig(h, s)
    >>> occ = mf.get_occ(e, c)
    >>> # DIIS without error vector
    >>> adiis = lib.diis.DIIS()
    >>> for i in range(7):
    ...     dm = mf.make_rdm1(c, occ)
    ...     f = h + mf.get_veff(mol, dm)
    ...     if i > 1:
    ...         f = adiis.update(f)
    ...     e, c = mf.eig(f, s)
    ...     print('E_%d = %.12f' % (i, mf.energy_tot(dm, h, mf.get_veff(mol, dm))))
    E_0 = -1.050329433306
    E_1 = -1.098566175145
    E_2 = -1.100103795287
    E_3 = -1.100152104615
    E_4 = -1.100153706922
    E_5 = -1.100153764848
    E_6 = -1.100153764878

    >>> # Take Hartree-Fock gradients as the error vector
    >>> adiis = lib.diis.DIIS()
    >>> for i in range(7):
    ...     dm = mf.make_rdm1(c, occ)
    ...     f = h + mf.get_veff(mol, dm)
    ...     if i > 1:
    ...         f = adiis.update(f, mf.get_grad(c, occ, f))
    ...     e, c = mf.eig(f, s)
    ...     print('E_%d = %.12f' % (i, mf.energy_tot(dm, h, mf.get_veff(mol, dm))))
    E_0 = -1.050329433306
    E_1 = -1.098566175145
    E_2 = -1.100103795287
    E_3 = -1.100152104615
    E_4 = -1.100153763813
    E_5 = -1.100153764878
    E_6 = -1.100153764878
    '''
    def __init__(self, dev=None, filename=None):
        if dev is not None:
            self.verbose = dev.verbose
            self.stdout = dev.stdout
        else:
            self.verbose = logger.INFO
            self.stdout = sys.stdout
        self.space = 6
        self.min_space = 1

##################################################
# don't modify the following private variables, they are not input options
        self.filename = filename
        self._diisfile = None
        self._buffer = {}
        self._bookkeep = [] # keep the ordering of input vectors
        self._head = 0
        self._H = None
        self._xprev = None
        self._err_vec_touched = False

    def _store(self, key, value):
        if value.size < INCORE_SIZE:
            self._buffer[key] = value

        # save the error vector if filename is given, this file can be used to
        # restore the DIIS state
        if value.size >= INCORE_SIZE or isinstance(self.filename, str):
            if self._diisfile is None:
                self._diisfile = misc.H5TmpFile(self.filename)
            if key in self._diisfile:
                self._diisfile[key][:] = value
            else:
                self._diisfile[key] = value
# to avoid "Unable to find a valid file signature" error when reopen from crash
            self._diisfile.flush()

    def push_err_vec(self, xerr):
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        key = 'e%d' % self._head
        self._store(key, xerr.ravel())

    def push_vec(self, x):
        x = x.ravel()

        if len(self._bookkeep) >= self.space:
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

        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d'%self._head
            xkey = 'x%d'%self._head
            self._store(xkey, x)
            if x.size < INCORE_SIZE:
                self._buffer[ekey] = x - self._xprev
                if isinstance(self.filename, str):
                    self._store(ekey, self._buffer[ekey])
            else:
                if ekey not in self._diisfile:
                    self._diisfile.create_dataset(ekey, (x.size,), x.dtype)
                for p0,p1 in prange(0, x.size, BLOCK_SIZE):
                    self._diisfile[ekey][p0:p1] = x[p0:p1] - self._xprev[p0:p1]
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
        return len(self._bookkeep)

    def update(self, x, xerr=None):
        '''Extrapolate vector 

        * If xerr the error vector is given, this function will push the target
        vector and error vector in the DIIS subspace, and use the error vector
        to extrapolate the vector and return the extrapolated vector.
        * If xerr is None, this function will take the difference between
        the current given vector and the last given vector as the error
        vector to extrapolate the vector.
        '''
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        dt = numpy.array(self.get_err_vec(self._head-1), copy=False)
        if self._H is None:
            self._H = numpy.zeros((self.space+1,self.space+1), dt.dtype)
            self._H[0,1:] = self._H[1:,0] = 1
        for i in range(nd):
            tmp = 0
            dti = self.get_err_vec(i)
            for p0,p1 in prange(0, dt.size, BLOCK_SIZE):
                tmp += numpy.dot(dt[p0:p1].conj(), dti[p0:p1])
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conjugate()
        dt = None
        h = self._H[:nd+1,:nd+1]
        g = numpy.zeros(nd+1, x.dtype)
        g[0] = 1

        #try:
        #    c = numpy.linalg.solve(h, g)
        #except numpy.linalg.linalg.LinAlgError:
        #    logger.warn(self, ' diis singular')
        if 1:
            w, v = scipy.linalg.eigh(h)
            idx = abs(w)>1e-14
            c = numpy.dot(v[:,idx]*(1/w[idx]), numpy.dot(v[:,idx].T.conj(), g))
        logger.debug1(self, 'diis-c %s', c)

        if self._xprev is None:
            xnew = numpy.zeros_like(x.ravel())
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = numpy.zeros_like(x.ravel())

        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            for p0,p1 in prange(0, x.size, BLOCK_SIZE):
                xnew[p0:p1] += xi[p0:p1] * ci
        return xnew.reshape(x.shape)

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

