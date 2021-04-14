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
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#         George Booth <george.booth@kcl.ac.uk>
#

'''
Functions for tuning the chemical potential.
'''

import numpy as np
from scipy import optimize
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__


def _objective(x, se, fock, nelec, occupancy=2, buf=None):
    ''' Objective function for the minimization
    '''

    w, v = se.eig(fock, chempot=x, out=buf)

    chempot, error = binsearch_chempot((w,v), se.nphys, nelec,
                                       occupancy=occupancy)

    return error**2


def _gradient(x, se, fock, nelec, occupancy=2, buf=None):
    ''' Gradient function for the minimization
    '''

    w, v = se.eig(fock, chempot=x, out=buf)

    chempot, error = binsearch_chempot((w,v), se.nphys, nelec,
                                       occupancy=occupancy)

    nocc = np.sum(w < chempot)
    nphys = se.nphys

    h1 = -np.dot(v[nphys:,nocc:].conj().T, v[nphys:,:nocc])
    zai = -h1 / lib.direct_sum('i,a->ai', w[:nocc], -w[nocc:])

    c_occ = np.dot(v[:nphys,nocc:], zai)
    d_rdm1 = np.dot(v[:nphys,:nocc], c_occ.conj().T) * 4

    ne = np.trace(d_rdm1).real
    d = occupancy * error * ne

    return error**2, d


def binsearch_chempot(fock, nphys, nelec, occupancy=2):
    ''' Finds a chemical potential which best agrees with the number
        of physical electrons and abides by the Aufbau principal via
        a binary search.

    Args:
        fock : 2D array or tuple of arrays
            Fock matrix to diagonalise, may be the physical Fock matrix
            or extended Fock matrix. Can also be the output of
            :func:`np.linalg.eigh` for this matrix, i.e. a tuple of the
            eigenvalues and eigenvectors.
        nphys : int
            Number of physical degrees of freedom
        nelec : int
            Number of physical electrons

    Kwargs:
        occupancy : int
            Occupancy of the states, i.e. 2 for RHF and 1 for UHF.
            Default 2.

    Returns:
        chemical potential, and the error in the number of electrons
    '''

    if isinstance(fock, tuple):
        w, v = fock
    else:
        w, v = np.linalg.eigh(fock)

    nmo = v.shape[-1]
    sum0 = sum1 = 0.0

    for i in range(nmo):
        n = occupancy * np.dot(v[:nphys,i].conj().T, v[:nphys,i]).real
        sum0, sum1 = sum1, sum1 + n

        if i > 0:
            if sum0 <= nelec and nelec <= sum1:
                break

    if abs(sum0 - nelec) < abs(sum1 - nelec):
        homo = i-1
        error = nelec - sum0
    else:
        homo = i
        error = nelec - sum1

    lumo = homo+1
    chempot = 0.5 * (w[homo] + w[lumo])

    return chempot, error


def minimize_chempot(se, fock, nelec, occupancy=2, x0=0.0, tol=1e-6, maxiter=200, jac=True):
    ''' Finds a set of auxiliary energies and chemical potential on
        the physical space which best satisfy the number of electrons.

    Args:
        se : AuxiliarySpace
            Auxiliary space
        fock : 2D array
            phys : 2D array
                Physical space (1p + 1h), typically the Fock matrix
        nelec : int
            Number of physical electrons

    Kwargs:
        occupancy : int
            Occupancy of the states, i.e. 2 for RHF and 1 for UHF.
            Default 2.
        x0 : float
            Initial guess for :attr:`chempot`. Default 0.0
        tol : float
            Convergence threshold (units are the same as :attr:`nelec`).
            Default 1e-6.
        maxiter : int
            Maximum number of iterations. Default 200.
        jac : bool
            If True, use gradient. Default True.

    Returns:
        AuxiliarySpace object with altered :attr:`energy` and
        :attr:`chempot`, and the SciPy :attr:`OptimizeResult` object.
    '''

    tol = tol**2  # we minimize the squared error
    dtype = np.result_type(se.energy.dtype, se.coupling.dtype, fock.dtype)
    buf = np.zeros((se.nphys+se.naux, se.nphys+se.naux), dtype=dtype)
    fargs = (se, fock, nelec, occupancy, buf)

    options = dict(maxiter=maxiter, ftol=tol, xtol=tol, gtol=tol)
    kwargs = dict(x0=x0, method='TNC', jac=jac, options=options)
    fun = _objective if not jac else _gradient

    opt = optimize.minimize(fun, args=fargs, **kwargs)

    se.energy -= opt.x
    se.chempot = binsearch_chempot(se.eig(fock), se.nphys, nelec,
                                   occupancy=occupancy)[0]

    return se, opt
