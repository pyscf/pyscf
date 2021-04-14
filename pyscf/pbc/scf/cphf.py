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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
Restricted coupled pertubed Hartree-Fock solver
Modified from pyscf.scf.cphf
'''


import numpy as np
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
    t0 = (logger.process_clock(), logger.perf_counter())

    nkpt = len(h1)
    moloc = np.zeros([nkpt+1], dtype=int)
    for k in range(nkpt):
        moloc[k+1] = moloc[k] + h1[k].size

    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_ai = [1 / lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]
    mo1base = []
    for k in range(nkpt):
        mo1base.append((h1[k] * -e_ai[k]).ravel())
    mo1base = np.hstack(mo1base)

    def vind_vo(mo1):
        mo1 = mo1.flatten()
        tmp = []
        for k in range(nkpt):
            tmp.append(mo1[moloc[k]:moloc[k+1]].reshape(h1[k].shape))
        v = fvind(tmp)
        for k in range(nkpt):
            v[k] *= e_ai[k]
            v[k] = v[k].ravel()
        return np.hstack(v)

    _mo1 = lib.krylov(vind_vo, mo1base,
                      tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log).flatten()
    log.timer('krylov solver in CPHF', *t0)
    mo1 = []
    for k in range(nkpt):
        mo1.append(_mo1[moloc[k]:moloc[k+1]].reshape(h1[k].shape))
    return mo1, None

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
    t0 = (logger.process_clock(), logger.perf_counter())

    nkpt = len(h1)
    ncomp = h1[0].shape[0]
    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_ai = [1 / lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]
    nocc = np.zeros([nkpt], dtype=int)
    nvir = np.zeros([nkpt], dtype=int)
    nmo  = np.zeros([nkpt], dtype=int)
    moloc = np.zeros([nkpt+1], dtype=int)
    for k in range(nkpt):
        nvir_k, nocc_k = e_ai[k].shape
        nmo_k = nvir_k + nocc_k
        nvir[k] = nvir_k
        nocc[k] = nocc_k
        nmo[k]  = nmo_k
        moloc[k+1] = moloc[k] + nmo_k * nocc_k * ncomp

    mo1base = []
    _mo1base = []
    mo_e1 = []
    for k in range(nkpt):
        mo1base.append(h1[k] - s1[k] * e_i[k])
        mo_e1.append(mo1base[k][:,occidx[k],:].copy())
        mo1base[k][:,viridx[k]] *= -e_ai[k]
        mo1base[k][:,occidx[k]] = -s1[k][:,occidx[k]] * .5
        _mo1base.append(mo1base[k].ravel())
    _mo1base = np.hstack(_mo1base)

    def vind_vo(mo1):
        mo1 = mo1.ravel()
        tmp = []
        for k in range(nkpt):
            tmp.append(mo1[moloc[k]:moloc[k+1]].reshape(-1,nmo[k],nocc[k]))
        v = fvind(tmp)
        for k in range(nkpt):
            v[k][:,viridx[k],:] *= e_ai[k]
            v[k][:,occidx[k],:] = 0
            v[k] = v[k].ravel()
        return np.hstack(v)
    _mo1 = lib.krylov(vind_vo, _mo1base,
                      tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1 = []
    for k in range(nkpt):
        mo1.append(_mo1[moloc[k]:moloc[k+1]].reshape(-1,nmo[k],nocc[k]))
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1)
    for k in range(nkpt):
        mo1[k][:,viridx[k]] = mo1base[k][:,viridx[k]] - \
                             v1mo[k][:,viridx[k]]*e_ai[k]

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo + u1*lib.direct_sum('i-j->ij',e_i,e_i)
    for k in range(nkpt):
        mo_e1[k] += mo1[k][:,occidx[k]] * lib.direct_sum('i-j->ij', e_i[k], e_i[k])
        mo_e1[k] += v1mo[k][:,occidx[k]]

    return mo1, mo_e1
