#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restricted coupled pertubed Hartree-Fock solver
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger


def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''
    Args:
        fvind : function
            Given density matrix, compute (ij|kl)D_{lk}*2 - (ij|kl)D_{jk}

    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
    '''
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field independent basis. First order overlap matrix is zero'''
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)
    mo1base = h1 * -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v *= e_ai
        return v.ravel()
    mo1 = lib.krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)
    return mo1.reshape(h1.shape), None

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]

    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir

    s1 = s1.reshape(-1,nmo,nocc)
    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1*e_i
    mo_e1 = hs[:,occidx,:].copy()

    mo1base[:,viridx] *= -e_ai
    mo1base[:,occidx] = -s1[:,occidx] * .5

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(-1,nmo,nocc)
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.ravel()
    mo1 = lib.krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1 = mo1.reshape(mo1base.shape)
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1.reshape(h1.shape)).reshape(-1,nmo,nocc)
    mo1[:,viridx] = mo1base[:,viridx] - v1mo[:,viridx]*e_ai

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    mo_e1 += mo1[:,occidx] * lib.direct_sum('i-j->ij', e_i, e_i)
    mo_e1 += v1mo[:,occidx,:]

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        return mo1.reshape(h1.shape), mo_e1.reshape(nocc,nocc)

if __name__ == '__main__':
    numpy.random.seed(1)
    nd = 3
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    a = numpy.random.random((nocc*nvir,nocc*nvir))
    a = a + a.T
    def fvind(x):
        v = numpy.dot(a,x[:,nocc:].reshape(-1,nocc*nvir).T)
        v1 = numpy.zeros((nd,nmo,nocc))
        v1[:,nocc:] = v.T.reshape(nd,nvir,nocc)
        return v1
    mo_energy = numpy.sort(numpy.random.random(nmo)) * 10
    mo_occ = numpy.zeros(nmo)
    mo_occ[:nocc] = 2
    e_i = mo_energy[mo_occ>0]
    e_a = mo_energy[mo_occ==0]
    e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)
    h1 = numpy.random.random((nd,nmo,nocc))
    h1[:,:nocc,:nocc] = h1[:,:nocc,:nocc] + h1[:,:nocc,:nocc].transpose(0,2,1)
    s1 = numpy.random.random((nd,nmo,nocc))
    s1[:,:nocc,:nocc] = s1[:,:nocc,:nocc] + s1[:,:nocc,:nocc].transpose(0,2,1)

    x = solve(fvind, mo_energy, mo_occ, h1, s1, max_cycle=30)[0]
    print(numpy.linalg.norm(x)-6.272581531366389)
    hs = h1.reshape(-1,nmo,nocc) - s1.reshape(-1,nmo,nocc)*e_i
    print(abs(hs[:,nocc:] + fvind(x)[:,nocc:]+x[:,nocc:]/e_ai).sum())

################
    xref = solve(fvind, mo_energy, mo_occ, h1, s1*0, max_cycle=30)[0][:,mo_occ==0]
    def fvind(x):
        return numpy.dot(a,x.reshape(nd,nocc*nvir).T).T.reshape(nd,nvir,nocc)
    h1 = h1[:,nocc:]
    x0 = numpy.linalg.solve(numpy.diag(1/e_ai.ravel())+a, -h1.reshape(nd,-1).T).T.reshape(nd,nvir,nocc)
    x1 = solve(fvind, mo_energy, mo_occ, h1, max_cycle=30)[0]
    print(abs(x0-x1).sum())
    print(abs(xref-x1).sum())
    print(abs(h1 + fvind(x1)+x1/e_ai).sum())
