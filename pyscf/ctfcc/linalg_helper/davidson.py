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


import time
import numpy
import sys
import ctf

from pyscf.lib import logger
from pyscf.ctfcc import mpi_helper


'''Davidson solver for EOM, a ctf replica from pyscf.pbc.linalg.davidson'''

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

def eigs(matvec, vecsize, nroots, x0=None, Adiag=None, guess=False, verbose=4):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''
    def matvec_args(vec, args=None):
        return matvec(vec)

    nroots = min(nroots, vecsize)
    conv, e, c = davidson(matvec, vecsize, nroots, x0, Adiag, verbose)
    return conv, e, c


def davidson(mult_by_A, N, neig, x0=None, Adiag=None, verbose=4, **kwargs):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested
    """
    log = logger.Logger(sys.stdout, verbose)

    cput1 = (time.clock(), time.time())

    Mmin = min(neig,N)
    Mmax = min(N,2000)

    tol = kwargs.get('conv_tol', 1e-6)

    def mult(arg):
        return mult_by_A(arg)

    if Adiag is None:
        Adiag = ctf.zeros(N, dtype=numpy.complex)
        for i in range(N):
            test = ctf.zeros(N,dtype=numpy.complex)
            if rank==0:
                test.write([i],[1.0])
            else:
                test.write([], [])
            val = mult(test)[i]
            if rank==0:
                Adiag.write([i], val)
            else:
                Adiag.write([],[])

    idx = mpi_helper.argsort(Adiag.real(), Mmin)
    lamda_k_old = 0
    lamda_k = 0
    target = 0
    conv = False
    if x0 is not None:
        assert (x0.shape == (Mmin, N) )
        b = x0.copy()
        Ab = tuple([mult(b[m]) for m in range(Mmin)])
        Ab = ctf.vstack(Ab).transpose()
    evals = numpy.zeros(neig,dtype=numpy.complex)
    evecs = []

    for istep,M in enumerate(range(Mmin,Mmax+1)):
        if M == Mmin:
            b = ctf.zeros((N,M))
            if rank==0:
                ind = [i*M+m for m,i in zip(range(M),idx)]
                fill = numpy.ones(len(ind))
                b.write(ind, fill)
            else:
                b.write([],[])
            Ab = tuple([mult(b[:,m]) for m in range(M)])
            Ab = ctf.vstack(Ab).transpose()
        else:
            Ab = ctf.hstack((Ab, mult(b[:,M-1]).reshape(N,-1)))

        Atilde = ctf.dot(b.conj().transpose(),Ab)
        Atilde = Atilde.to_nparray()

        lamda, alpha = diagonalize_asymm(Atilde)
        lamda_k_old, lamda_k = lamda_k, lamda[target]
        alpha_k = ctf.astensor(alpha[:,target])
        if M == Mmax:
            break
        q = ctf.dot( Ab-lamda_k*b, alpha_k)
        qnorm = ctf.norm(q)
        log.info('davidson istep = %d  root = %d  E = %.15g  dE = %.9g  residual = %.6g',
                 istep, target, lamda_k.real, (lamda_k - lamda_k_old).real, qnorm)
        cput1 = log.timer('davidson iter', *cput1)

        if ctf.norm(q) < tol:
            evecs.append(ctf.dot(b,alpha_k))
            evals[target] = lamda_k
            if target == neig-1:
                conv = True
                break
            else:
                target += 1
        eps = 1e-10
        xi = q/(lamda_k-Adiag+eps)
        bxi,R = ctf.qr(ctf.hstack((b,xi.reshape(N,-1))))
        nlast = bxi.shape[-1] - 1
        b = ctf.hstack((b,bxi[:,nlast].reshape(N,-1))) #can not replace nlast with -1, (inconsistent between numpy and ctf)
    evecs = ctf.vstack(tuple(evecs))
    return conv, evals.real, evecs

def diagonalize_asymm(H):
    E,C = numpy.linalg.eig(H)
    idx = E.real.argsort()
    E = E[idx]
    C = C[:,idx]
    return E,C
