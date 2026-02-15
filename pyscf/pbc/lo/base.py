#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#         Gengzhi Yang <genzyang17@gmail.com>
#

'''
Base classes for k-point orbital rotations

ref. [To be updated]
'''

import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.soscf import kciah
from pyscf.lo import cholesky_mos
from pyscf.lo import boys
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.tools import mo_mapping
from pyscf import __config__


def _align_phase(mo, mo0):
    ovlp = lib.dot(mo.conj().T, mo0)
    l, _, r = numpy.linalg.svd(ovlp)
    return lib.dot(l, r)

def wannierization(cell, kpts, mo_coeff, kmesh=None, refcell_only=False):
    mo_coeff = numpy.asarray(mo_coeff, order='C')
    if kmesh is None: kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)
    nkpts,nao,nmo = mo_coeff.shape
    Nao = nkpts*nao

    if refcell_only:
        W = lib.einsum('Rk,kmi,k->Rmi', phase, mo_coeff, phase[0].conj()).reshape(Nao,nmo)
    else:
        W = lib.einsum('Rk,kmi,Sk->RmSi', phase, mo_coeff, phase.conj()).reshape(Nao,nkpts*nmo)

    return W


class KptsOrbitalLocalizer(lib.StreamObject, kciah.SubspaceCIAHOptimizerMixin):

    conv_tol = getattr(__config__, 'pbc_lo_base_conv_tol', 1e-6)
    conv_tol_grad = getattr(__config__, 'pbc_lo_base_conv_tol_grad', None)
    max_cycle = getattr(__config__, 'pbc_lo_base_max_cycle', 100)
    max_iters = getattr(__config__, 'pbc_lo_base_max_iters', 20)
    max_stepsize = getattr(__config__, 'pbc_lo_base_max_stepsize', .05)
    ah_trust_region = getattr(__config__, 'pbc_lo_base_ah_trust_region', 3)
    ah_start_tol = getattr(__config__, 'pbc_lo_base_ah_start_tol', 1e9)
    ah_max_cycle = getattr(__config__, 'pbc_lo_base_ah_max_cycle', 40)
    init_guess = getattr(__config__, 'pbc_lo_base_init_guess', 'atomic')
    algorithm = getattr(__config__, 'pbc_lo_base_init_guess', 'ciah')
    maximize = getattr(__config__, 'pbc_lo_base_init_guess', False)

    _keys = {
        'conv_tol', 'conv_tol_grad', 'max_cycle', 'max_iters',
        'max_stepsize', 'ah_trust_region', 'ah_start_tol',
        'ah_max_cycle', 'init_guess', 'algorithm', 'cell', 'mo_coeff', 'kpts'
    }

    def __init__(self, cell, mo_coeff, kpts):
        if isinstance(kpts, KPoints):
            if not kpts.time_reversal:
                raise NotImplementedError('k-point symmetry not implemented')
            mo_coeff = remove_trs_mo(mo_coeff, kpts)
            kpts = kpts.kpts
            logger.warn(cell, 'Time-reversal symmetry will be ignored')

        self.kpts = kpts
        ''' One gauge-fixed complex rotation (type 1)
            + (Nk - 1) general complex rotations (type 2)
        '''
        rtypes = [1] + [2] * (len(kpts)-1)
        kciah.SubspaceCIAHOptimizerMixin.__init__(self, mo_coeff[0].shape[1], rtypes)

        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.mo_coeff = numpy.asarray(mo_coeff)

    def rotate_orb(self, u=None):
        if u is None:
            return self.mo_coeff
        else:
            return numpy.asarray([lib.dot(xk, uk) for xk,uk in zip(self.mo_coeff, u)])

    def dump_flags(self, verbose=None):
        boys.OrbitalLocalizer.dump_flags(self, verbose)

    def get_init_guess(self, key='atomic'):
        ''' Generate initial guess for localization.

            The initial guess is first generated for the first k-point. Other k-points

        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        cell = self.cell
        kpts = self.kpts
        mo_coeff = self.mo_coeff

        mo0 = mo_coeff[0]
        if isinstance(key, str) and key.lower().startswith('atom'):
            u00 = boys.atomic_init_guess(cell, mo0, kpt=kpts[0])
        elif isinstance(key, str) and key.lower().startswith('cho'):
            mo_init = cholesky_mos(mo0)
            S = cell.pbc_intor('int1e_ovlp', kpt=kpts[0])
            u00 = numpy.linalg.multi_dot([mo0.T, S, mo_init])
        else:
            return self.identity_rotation()

        # diabatization: align phase of MO[k] to MO[0]
        mo0 = lib.dot(mo_coeff[0], u00)

        return self.diabatization(mo0)

    def diabatization(self, mo0):
        mo_coeff = self.mo_coeff
        return numpy.asarray([_align_phase(mok, mo0) for mok in mo_coeff])

    def get_wannier_function(self, u=None, refcell_only=False):
        mo_coeff = self.rotate_orb(u)

        return wannierization(self.cell, self.kpts, mo_coeff, refcell_only=refcell_only)

    def sort_orb(self, u):
        u = numpy.asarray(u)
        u00 = u.sum(axis=0) / len(self.kpts)
        sorted_idx = mo_mapping.mo_1to1map(u00)
        return self.rotate_orb([uk[:,sorted_idx] for uk in u])

    kernel = boys.kernel


def get_kmesh(cell, kpts, tol=1e-6, nmax=1000):
    scaled_kpts = cell.get_scaled_kpts(kpts-kpts[0])
    nks = numpy.arange(1,nmax+1)
    ks = lib.einsum('kx,n->xnk', scaled_kpts, nks)
    mask = numpy.all(abs(ks - numpy.round(ks)) < tol, axis=-1)
    if not numpy.all(numpy.any(mask, axis=-1)):
        raise RuntimeError('Input kmesh is either too large or not a (shifted) regular mesh.')

    kmesh = [numpy.where(mask[i])[0][0]+1 for i in range(3)]

    return kmesh

def remove_trs_mo(mo_coeff_trs, kpts):
    assert( len(mo_coeff_trs) == kpts.nkpts_ibz )

    mo_coeff_trs = numpy.asarray(mo_coeff_trs, order='C')
    kpairs = [numpy.where(kpts.bz2ibz==q)[0] for q in range(kpts.nkpts_ibz)]
    mo_coeff = numpy.empty((kpts.nkpts,*mo_coeff_trs[0].shape), dtype=mo_coeff_trs.dtype)

    for q,kpair in enumerate(kpairs):
        if len(kpair) == 1:
            k = kpair[0]
            mo_coeff[k] = mo_coeff_trs[q]
        else:
            k1, k2 = kpair
            mo_coeff[k1] = mo_coeff_trs[q].conj()
            mo_coeff[k2] = mo_coeff_trs[q]

    return mo_coeff

def _pack_bz2ibz(g, kpts_symm, sign=1):
    G = numpy.zeros((kpts_symm.nkpts_ibz, *g[0].shape), dtype=numpy.complex128)
    for q in range(kpts_symm.nkpts_ibz):
        idx = numpy.where(kpts_symm.bz2ibz==q)[0]
        if idx.size == 1:
            G[q] = g[idx[0]]
        else:
            G[q] = g[idx[0]] + sign * g[idx[1]].conj()
    return G

def _unpack_ibz2bz(G, kpts_symm, sign=1):
    g = numpy.zeros((kpts_symm.nkpts, *G[0].shape), dtype=numpy.complex128)
    for q in range(kpts_symm.nkpts_ibz):
        idx = numpy.where(kpts_symm.bz2ibz==q)[0]
        g[idx[0]] = G[q]
        if idx.size == 2:
            g[idx[1]] = sign * G[q].conj()
    return g


class KptsOrbitalLocalizerReal(KptsOrbitalLocalizer):

    def __init__(self, cell, mo_coeff, kpts):
        if isinstance(kpts, KPoints):
            if not kpts.time_reversal:
                raise NotImplementedError('k-point symmetry not implemented')
            mo_coeff = remove_trs_mo(mo_coeff, kpts)
            kpts_symm = kpts
        else:
            if not gamma_point(kpts[0]):
                raise ValueError('Input k-mesh is not Gamma-centered.')
            kmesh = get_kmesh(cell, kpts)
            kpts_symm = cell.make_kpts(kmesh, time_reversal_symmetry=True)

        self.kpts = kpts_symm.kpts
        self.kpts_symm = kpts_symm

        # Time-reversal invariant k-points => real rotation (type 0)
        # Time-reversal paired    k-points => complex rotation (type 2)
        rtypes = numpy.zeros(kpts_symm.nkpts_ibz, dtype=int)
        for q in range(kpts_symm.nkpts_ibz):
            idx = numpy.where(kpts_symm.bz2ibz==q)[0]
            if idx.size == 2:
                rtypes[q] = 2

        kciah.SubspaceCIAHOptimizerMixin.__init__(self, mo_coeff[0].shape[1], rtypes)

        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.mo_coeff = numpy.asarray(mo_coeff)

    def rotate_orb(self, u=None):
        if u is None:
            return self.mo_coeff
        else:
            u = _unpack_ibz2bz(u, self.kpts_symm)
            return numpy.asarray([lib.dot(xk, uk) for xk,uk in zip(self.mo_coeff, u)])

    def diabatization(self, mo0):
        u0 = []
        for q in range(self.kpts_symm.nkpts_ibz):
            idx = numpy.where(self.kpts_symm.bz2ibz==q)[0]
            u0.append( _align_phase(self.mo_coeff[idx[0]], mo0) )
        return numpy.asarray(u0)
