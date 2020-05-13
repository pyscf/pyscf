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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Unrestricted coupled pertubed Hartree-Fock solver
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

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = numpy.count_nonzero(occidxa)
    noccb = numpy.count_nonzero(occidxb)
    nvira = mo_occ[0].size - nocca
    nvirb = mo_occ[1].size - noccb
    e_ai = numpy.hstack(((mo_energy[0][viridxa,None]-mo_energy[0][occidxa]).ravel(),
                         (mo_energy[1][viridxb,None]-mo_energy[1][occidxb]).ravel()))
    e_ai = 1 / e_ai
    mo1base = numpy.hstack((h1[0].reshape(-1,nvira*nocca),
                            h1[1].reshape(-1,nvirb*noccb)))
    mo1base *= -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(mo1base.shape)).reshape(mo1base.shape)
        v *= e_ai
        return v.ravel()
    mo1 = lib.krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)

    if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
        mo1 = (mo1[:nocca*nvira].reshape(nvira,nocca),
               mo1[nocca*nvira:].reshape(nvirb,noccb))
    else:
        mo1 = mo1.reshape(mo1base.shape)
        mo1_a = mo1[:,:nvira*nocca].reshape(-1,nvira,nocca)
        mo1_b = mo1[:,nvira*nocca:].reshape(-1,nvirb,noccb)
        mo1 = (mo1_a, mo1_b)
    return mo1, None

# h1 shape is (:,nvir+nocc,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = numpy.count_nonzero(occidxa)
    noccb = numpy.count_nonzero(occidxb)
    nmoa, nmob = mo_occ[0].size, mo_occ[1].size
    eai_a = mo_energy[0][viridxa,None] - mo_energy[0][occidxa]
    eai_b = mo_energy[1][viridxb,None] - mo_energy[1][occidxb]
    s1_a = s1[0].reshape(-1,nmoa,nocca)
    nset = s1_a.shape[0]
    s1_b = s1[1].reshape(nset,nmob,noccb)
    hs_a = mo1base_a = h1[0].reshape(nset,nmoa,nocca) - s1_a * mo_energy[0][occidxa]
    hs_b = mo1base_b = h1[1].reshape(nset,nmob,noccb) - s1_b * mo_energy[1][occidxb]
    mo_e1_a = hs_a[:,occidxa].copy()
    mo_e1_b = hs_b[:,occidxb].copy()

    mo1base_a[:,viridxa]/= -eai_a
    mo1base_b[:,viridxb]/= -eai_b
    mo1base_a[:,occidxa] = -s1_a[:,occidxa] * .5
    mo1base_b[:,occidxb] = -s1_b[:,occidxb] * .5

    eai_a = 1. / eai_a
    eai_b = 1. / eai_b
    mo1base = numpy.hstack((mo1base_a.reshape(nset,-1), mo1base_b.reshape(nset,-1)))

    def vind_vo(mo1):
        v = fvind(mo1).reshape(mo1base.shape)
        v1a = v[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
        v1b = v[:,nmoa*nocca:].reshape(nset,nmob,noccb)
        v1a[:,viridxa] *= eai_a
        v1b[:,viridxb] *= eai_b
        v1a[:,occidxa] = 0
        v1b[:,occidxb] = 0
        return v.ravel()
    mo1 = lib.krylov(vind_vo, mo1base.ravel(),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1).reshape(mo1base.shape)
    v1a = v1mo[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    v1b = v1mo[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    mo1_b = mo1[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1_a[:,viridxa] = mo1base_a[:,viridxa] - v1a[:,viridxa] * eai_a
    mo1_b[:,viridxb] = mo1base_b[:,viridxb] - v1b[:,viridxb] * eai_b

    mo_e1_a += mo1_a[:,occidxa] * (mo_energy[0][occidxa,None] - mo_energy[0][occidxa])
    mo_e1_b += mo1_b[:,occidxb] * (mo_energy[1][occidxb,None] - mo_energy[1][occidxb])
    mo_e1_a += v1mo[:,:nmoa*nocca].reshape(nset,nmoa,nocca)[:,occidxa]
    mo_e1_b += v1mo[:,nmoa*nocca:].reshape(nset,nmob,noccb)[:,occidxb]

    if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
        mo1_a, mo1_b = mo1_a[0], mo1_b[0]
        mo_e1_a, mo_e1_b = mo_e1_a[0], mo_e1_b[0]
    return (mo1_a, mo1_b), (mo_e1_a, mo_e1_b)

