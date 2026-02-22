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
K-point Pipek-Mezey localization

ref: Yang and Ye, arxiv:2602.12382
'''

import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.lo import boys
from pyscf.lo import iao
from pyscf.pbc.lo.base import KptsOrbitalLocalizer, KptsOrbitalLocalizerReal
from pyscf.pbc.lo.base import _pack_bz2ibz, _unpack_ibz2bz, get_kmesh
from pyscf.lo.stability import stability_newton
from pyscf.pbc.lo.stability import pipek_stability_jacobi
from pyscf.pbc.tools import k2gamma
from pyscf import __config__


def gen_proj_op(cell, mo_coeff, kpts, method='meta_lowdin', proj_data=None, verbose=None):
    ''' Construct atomic projection operators and their action in k-point space.

    Args:
        cell : Cell
            PySCF Cell object.
        mo_coeff : ndarray of shape (nkpts, norb, norb)
            Bloch-orbital coefficients at each k-point.
        kpts : ndarray of shape (nkpts, 3)
            k-points.
        method : str
            Atomic population / projection scheme. It can be 'mulliken', 'lowdin',
            'meta_lowdin', 'iao', 'becke'.
        proj_data : object or None
            Precomputed projection data obtained from :func:`get_proj_data`. If provided,
            it must be consistent with ``method``.
        verbose : int or None
            Verbosity level.

    Returns:
        projk0 : ndarray of shape (Natm, nkpts, norb, norb)
            Atomic projection matrix elements in mixed (k, 0) representation,
            ``projk0[x, k, i, j] = < phi_{k i} | P_x | w_{0 j} >``,
            where ``phi_{k i}`` denotes Bloch orbital *i* at k-point *k* and ``w_{0 j}``
            denotes Wannier orbital *j* in the home cell.
        popk : ndarray of shape (Natm, nkpts, norb)
            Diagonal atomic populations,
            ``popk[x, k, i] = < phi_{k i} | P_x | phi_{k i} >``.
        proj_op : callable
            A function ``proj_op(v)`` that applies the atomic projector(s) to a vector ``v``.
    '''
    method = method.lower().replace('_', '-')
    mo_coeff = numpy.asarray(mo_coeff)
    nkpts,nao,nmo = mo_coeff.shape
    Nmo = nkpts*nmo

    if proj_data is None:
        proj_data = get_proj_data(cell, mo_coeff, method, kpts)

    def get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom):
        Natm = len(offset_nr_by_atom)
        nproj = proj_coeff.shape[3]
        Nproj = nkpts*nproj

        proj_coeff = proj_coeff.reshape(nkpts,nao,Nproj)
        kcsc = numpy.asarray([
            lib.dot(mo_coeff[k].conj().T, proj_coeff[k]) for k in range(nkpts)
        ])

        scsc = numpy.asarray(kcsc.sum(axis=0).conj().T, order='C')  # Sx,i
        kcsc = numpy.asarray(kcsc.reshape(Nmo,Nproj).conj().T, order='C')    # Sx,ki

        projk0 = numpy.empty((Natm,Nmo,nmo), dtype=numpy.complex128)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            lib.dot(kcsc[p0:p1].conj().T, scsc[p0:p1], c=projk0[i])
        projk0 = projk0.reshape(Natm,nkpts,nmo,nmo)

        popk = numpy.empty((Natm,Nmo), dtype=numpy.float64)
        kcsc2 = abs(kcsc)**2.   # Sx,ki
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            popk[i] = kcsc2[p0:p1].sum(axis=0)
        popk = popk.reshape(Natm,nkpts,nmo)

        def proj_op(x):
            kcscx = lib.dot(kcsc, x.reshape(Nmo,nmo))    # Sx,j
            Px = numpy.empty((Natm, Nmo, nmo), dtype=numpy.complex128)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(kcsc[p0:p1].conj().T, kcscx[p0:p1], c=Px[i])    # ki|m, m|j -> ki|j
            Px = Px.reshape(Natm,nkpts,nmo,nmo)

            return Px

        return projk0, popk, proj_op

    def get_proj_op_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom):
        Natm = len(offset_nr_by_atom)
        nproj = proj_coeff.shape[3]
        Nproj = nkpts*nproj

        proj_coeff = proj_coeff.reshape(nkpts,nao,Nproj)
        projtild_coeff = projtild_coeff.reshape(nkpts,nao,Nproj)
        kcsc = numpy.asarray([
            lib.dot(mo_coeff[k].conj().T, proj_coeff[k]) for k in range(nkpts)
        ])
        kcsctild = numpy.asarray([
            lib.dot(mo_coeff[k].conj().T, projtild_coeff[k]) for k in range(nkpts)
        ])

        scsc = numpy.asarray(kcsc.sum(axis=0).conj().T, order='C')  # Sx,i
        kcsc = numpy.asarray(kcsc.reshape(Nmo,Nproj).conj().T, order='C')    # Sx,ki
        scsctild = numpy.asarray(kcsctild.sum(axis=0).conj().T, order='C')  # Sx,i
        kcsctild = numpy.asarray(kcsctild.reshape(Nmo,Nproj).conj().T, order='C')   # Sx,ki

        projk0 = numpy.empty((Natm,Nmo,nmo), dtype=numpy.complex128)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            lib.dot(kcsc[p0:p1].conj().T, scsctild[p0:p1], c=projk0[i])
            lib.dot(kcsctild[p0:p1].conj().T, scsc[p0:p1], c=projk0[i], beta=1)
        projk0 *= 0.5
        projk0 = projk0.reshape(Natm,nkpts,nmo,nmo)

        popk = numpy.empty((Natm,Nmo), dtype=numpy.float64)
        kcsc2 = (kcsc.conj()*kcsctild).real
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            popk[i] = kcsc2[p0:p1].sum(axis=0)
        popk = popk.reshape(Natm,nkpts,nmo)

        def proj_op(x):
            kcscx = lib.dot(kcsc, x.reshape(Nmo,nmo))    # Sx,j
            kcsctildx = lib.dot(kcsctild, x.reshape(Nmo,nmo))    # Sx,j
            Px = numpy.empty((Natm, Nmo, nmo), dtype=numpy.complex128)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(kcsc[p0:p1].conj().T, kcsctildx[p0:p1], c=Px[i])
                lib.dot(kcsctild[p0:p1].conj().T, kcscx[p0:p1], c=Px[i], beta=1)
            Px *= 0.5
            Px = Px.reshape(Natm,nkpts,nmo,nmo)

            return Px

        return projk0, popk, proj_op

    if method == 'mulliken':
        raise NotImplementedError

    elif method == 'becke':
        raise NotImplementedError

    elif method in ('lowdin', 'meta-lowdin'):
        proj_coeff, offset_nr_by_atom = proj_data
        projk0, popk, proj_op = get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo'):
        proj_coeff, offset_nr_by_atom = proj_data
        projk0, popk, proj_op = get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method == 'iao-biorth':
        proj_coeff, projtild_coeff, offset_nr_by_atom = proj_data
        projk0, popk, proj_op = get_proj_op_biorth(mo_coeff, proj_coeff, projtild_coeff,
                                                   offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return projk0, popk, proj_op

def atomic_pops(cell, mo_coeff, kpts, mode='kk', method='meta_lowdin', proj_data=None,
                verbose=None):
    '''Compute atomic population matrices in various k/Wannier representations.

    Kwargs:
        method : str
            Atomic population / projection scheme. Supported values include
            'mulliken', 'lowdin', 'meta_lowdin', 'iao', and 'becke'.
        mode : str
            Representation used for evaluating projection matrix elements:
                - 'kk': < phi_{k i} | P_x | phi_{k' j} >
                - 'k0': < phi_{k i} | P_x | w_{0 j} >
                - '0k': < w_{0 i} | P_x | phi_{k j} >
                - '00': < w_{0 i} | P_x | w_{0 j} >

            Here ``phi_{k i}`` denotes Bloch orbital *i* at k-point *k*, and ``w_{0 i}``
            denotes Wannier orbital *i* in the home cell. Default is 'kk'.

    Returns:
        proj : ndarray
            Atomic projection matrix. The shape depends on ``mode``:
                - 'kk': (Natm, nkpts, norb, nkpts, norb)
                - 'k0': (Natm, nkpts, norb, norb)
                - '0k': (Natm, norb, nkpts, norb)
                - '00': (Natm, norb, norb)

            The dtype follows the result type of ``mo_coeff`` and the underlying
            projection coefficients.

    Note:
        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    method = method.lower().replace('_', '-')
    mo_coeff = numpy.asarray(mo_coeff)
    nkpts,nao,nmo = mo_coeff.shape
    Nao, Nmo = nkpts*nao, nkpts*nmo
    kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)

    if proj_data is None:
        proj_data = get_proj_data(cell, mo_coeff, method, kpts)

    def proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom):
        nproj = proj_coeff.shape[-1]
        Nproj = nkpts*nproj

        proj_coeff = proj_coeff.reshape(nkpts,nao,Nproj)

        # computing csc
        if mode in ['kk','k0','0k']:
            kcsc = numpy.asarray([
                lib.dot(mo_coeff[k].conj().T, proj_coeff[k]) for k in range(nkpts)
            ])
            if '0' in mode:
                scsc = numpy.ascontiguousarray(kcsc.sum(axis=0).conj().T)       # Sx,0i
            kcsc = numpy.ascontiguousarray(kcsc.reshape(Nmo,Nproj).conj().T)    # Sx,ki
        elif mode == '00':
            scsc = lib.dot(proj_coeff.reshape(Nao,Nproj).conj().T,
                           mo_coeff.reshape(-1,nmo))                            # Sx,0i
        else:
            raise ValueError('Unknown mode %s' % str(mode))

        # computing proj matrix
        if mode == 'kk':
            proj = numpy.empty((scell.natm,Nmo,Nmo), dtype=kcsc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(kcsc[p0:p1].conj().T, kcsc[p0:p1], c=proj[i])
            proj = proj.reshape(scell.natm,nkpts,nmo,nkpts,nmo)

        elif mode in ['0k','k0']:
            if mode == '0k':
                proj = numpy.empty((scell.natm,nmo,Nmo), dtype=kcsc.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    lib.dot(scsc[p0:p1].conj().T, kcsc[p0:p1], c=proj[i])
                proj = proj.reshape(scell.natm,nmo,nkpts,nmo)
            else:
                proj = numpy.empty((scell.natm,Nmo,nmo), dtype=kcsc.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    lib.dot(kcsc[p0:p1].conj().T, scsc[p0:p1], c=proj[i])
                proj = proj.reshape(scell.natm,nkpts,nmo,nmo)

        elif mode == '00':
            proj = numpy.empty((scell.natm,nmo,nmo), dtype=scsc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(scsc[p0:p1].conj().T, scsc[p0:p1], c=proj[i])

        else:
            raise ValueError('Unknown mode %s' % str(mode))

        return proj

    def proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom):
        nproj = proj_coeff.shape[-1]
        Nproj = nkpts*nproj

        proj_coeff = proj_coeff.reshape(nkpts,nao,Nproj)
        projtild_coeff = projtild_coeff.reshape(nkpts,nao,Nproj)

        # computing csc
        if mode in ['kk','k0','0k']:
            kcsc = numpy.asarray([
                lib.dot(mo_coeff[k].conj().T, proj_coeff[k]) for k in range(nkpts)
            ])
            kcsctild = numpy.asarray([
                lib.dot(mo_coeff[k].conj().T, projtild_coeff[k]) for k in range(nkpts)
            ])
            if '0' in mode:
                scsc = numpy.ascontiguousarray(kcsc.sum(axis=0).conj().T)               # Sx,0i
                scsctild = numpy.ascontiguousarray(kcsctild.sum(axis=0).conj().T)       # Sx,0i
            kcsc = numpy.ascontiguousarray(kcsc.reshape(Nmo,Nproj).conj().T)            # Sx,ki
            kcsctild = numpy.ascontiguousarray(kcsctild.reshape(Nmo,Nproj).conj().T)    # Sx,ki
        elif mode == '00':
            scsc = lib.dot(proj_coeff.reshape(Nao,Nproj).conj().T,
                           mo_coeff.reshape(-1,nmo))                                    # Sx,0i
            scsctild = lib.dot(projtild_coeff.reshape(Nao,Nproj).conj().T,
                               mo_coeff.reshape(-1,nmo))                                # Sx,0i
        else:
            raise ValueError('Unknown mode %s' % str(mode))

        # computing proj matrix
        if mode == 'kk':
            proj = numpy.empty((scell.natm,Nmo,Nmo), dtype=kcsc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(kcsc[p0:p1].conj().T, kcsctild[p0:p1], c=proj[i])
                lib.dot(kcsctild[p0:p1].conj().T, kcsc[p0:p1], c=proj[i], beta=1)
            proj *= 0.5
            proj = proj.reshape(scell.natm,nkpts,nmo,nkpts,nmo)

        elif mode in ['0k','k0']:
            if mode == '0k':
                proj = numpy.empty((scell.natm,nmo,Nmo), dtype=kcsc.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    lib.dot(scsc[p0:p1].conj().T, kcsctild[p0:p1], c=proj[i])
                    lib.dot(scsctild[p0:p1].conj().T, kcsc[p0:p1], c=proj[i], beta=1)
                proj *= 0.5
                proj = proj.reshape(scell.natm,nmo,nkpts,nmo)
            else:
                proj = numpy.empty((scell.natm,Nmo,nmo), dtype=kcsc.dtype)
                for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                    lib.dot(kcsc[p0:p1].conj().T, scsctild[p0:p1], c=proj[i])
                    lib.dot(kcsctild[p0:p1].conj().T, scsc[p0:p1], c=proj[i], beta=1)
                proj *= 0.5
                proj = proj.reshape(scell.natm,nkpts,nmo,nmo)

        elif mode == '00':
            proj = numpy.empty((scell.natm,nmo,nmo), dtype=scsc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(scsc[p0:p1].conj().T, scsctild[p0:p1], c=proj[i])
                lib.dot(scsctild[p0:p1].conj().T, scsc[p0:p1], c=proj[i], beta=1)
            proj *= 0.5

        else:
            raise ValueError('Unknown mode %s' % str(mode))

        return proj

    if method == 'mulliken':
        raise NotImplementedError

    elif method == 'becke':
        raise NotImplementedError

    elif method in ('lowdin', 'meta-lowdin'):
        proj_coeff, offset_nr_by_atom = proj_data

        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method == 'iao-biorth':
        proj_coeff, projtild_coeff, offset_nr_by_atom = proj_data

        proj = proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo'):
        proj_coeff, offset_nr_by_atom = proj_data

        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj

def get_proj_data(cell, mo_coeff, method, kpts, minao=None):
    ''' Precompute data for atomic projectors
    '''
    if method is None:  # allow customized population method to skip this precompute
        return None

    method = method.lower().replace('_', '-')

    mo_coeff = numpy.asarray(mo_coeff)
    nkpts,nao,nmo = mo_coeff.shape
    kmesh = get_kmesh(cell, kpts)
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh=kmesh)

    if method == 'mulliken':
        proj_data = None

    elif method in ('lowdin', 'meta-lowdin'):
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        proj_coeff = numpy.asarray([orth.orth_ao(cell, method, 'ANO', s=s[k],
                                    adjust_phase=False) for k in range(nkpts)])
        proj_coeff = lib.einsum('kmn,knx->kmx', s, proj_coeff)
        proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
        offset_nr_by_atom = scell.offset_nr_by_atom()
        proj_data = (proj_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo', 'iao-biorth'):
        if minao is None: minao = 'minao'
        s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        iao_coeff = iao.iao(cell, mo_coeff, kpts=kpts, minao=minao)
        iao_scell = iao.reference_mol(scell, minao=minao)
        offset_nr_by_atom = iao_scell.offset_nr_by_atom()

        if method == 'iao-biorth':
            ovlp = lib.einsum('kmx,kmn,kny->kxy', iao_coeff.conj(), s, iao_coeff)
            iaotild_coeff = numpy.asarray([numpy.linalg.solve(ovlp[k],
                                           iao_coeff[k].conj().T).conj().T
                                           for k in range(nkpts)], order='C')
            iao_coeff = lib.einsum('kmn,knx->kmx', s, iao_coeff)
            iaotild_coeff = lib.einsum('kmn,knx->kmx', s, iaotild_coeff)
            proj_coeff = lib.einsum('kmx,Sk->kmSx', iao_coeff, phase.conj()) / nkpts**0.5
            projtild_coeff = lib.einsum('kmx,Sk->kmSx', iaotild_coeff,
                                        phase.conj()) / nkpts**0.5
            proj_data = (proj_coeff, projtild_coeff, offset_nr_by_atom)
        else:
            proj_coeff = numpy.asarray([orth.vec_lowdin(iao_coeff[k], s[k])
                                        for k in range(nkpts)])
            proj_coeff = lib.einsum('kmn,knx->kmx', s, proj_coeff)
            proj_coeff = lib.einsum('kmx,Sk->kmSx', proj_coeff, phase.conj()) / nkpts**0.5
            proj_data = (proj_coeff, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj_data

def gen_g_hop(mlo, u=None):
    ''' Get matrix form of grad, hop, and h_diag without packing.
    '''
    exponent = mlo.exponent
    nkpts = len(mlo.kpts)
    norb = mlo.norb

    mo_coeff = mlo.rotate_orb(u)

    projk0, popk, proj_op = gen_proj_op(mlo.cell, mo_coeff, mlo.kpts, method=mlo.pop_method,
                                        proj_data=mlo._proj_data, verbose=mlo.verbose)
    projk0 = numpy.ascontiguousarray(projk0.transpose(1,2,3,0)) # ki,0j,x
    popk = numpy.ascontiguousarray(popk.transpose(1,2,0))       # ki,x

    pop0 = numpy.ascontiguousarray(lib.einsum('kiix->ix', projk0.real)) # i,x
    pop0exp1 = pop0**(exponent-1)
    pop0exp2 = pop0**(exponent-2)

    # gradient
    g = get_grad(mlo, u, projk0)

    # hessian diagonal
    if getattr(mlo, 'kpts_symm', None) is None:
        h_diag = numpy.zeros((nkpts,norb,norb), dtype=numpy.complex128)

        # disconnected terms
        g1 = lib.einsum('kijx,jx->kij', projk0.real**2, pop0exp2)
        g2 = lib.einsum('kijx,jx->kij', projk0.imag**2, pop0exp2)
        h_diag += -4 * exponent * (exponent-1) * (g1 + g2 * 1j)

        # connected terms
        g1 = lib.einsum('kiix,ix->ki', projk0.real, pop0exp1)
        g2 = lib.einsum('kjx,ix->kij', popk, pop0exp1)
        h_diag += 2 * exponent * (g1[:,:,None] - g2) * (1 + 1j)
    else:
        kpts_symm = mlo.kpts_symm

        projK0 = _pack_bz2ibz(projk0, kpts_symm)
        popK = _pack_bz2ibz(popk, kpts_symm)
        h_diag = numpy.zeros((kpts_symm.nkpts_ibz,norb,norb), dtype=numpy.complex128)

        # disconnected terms
        g1 = lib.einsum('kijx,jx->kij', projK0.real**2, pop0exp2)
        g2 = lib.einsum('kijx,jx->kij', projK0.imag**2, pop0exp2)
        h_diag += -4 * exponent * (exponent-1) * (g1 + g2 * 1j)

        # connected terms
        g1 = lib.einsum('kiix,ix->ki', projK0.real, pop0exp1)
        g2 = lib.einsum('kjx,ix->kij', popK, pop0exp1)
        h_diag += 2 * exponent * (g1[:,:,None] - g2) * (1 + 1j)

        projK0 = popK = None

    for hk in h_diag:
        numpy.fill_diagonal(hk, numpy.diag(hk)*0.5)
        hk += hk.T

    # hessian vector product
    Gk = lib.einsum('kijx,jx->kij', projk0, pop0exp1)

    def h_op(x):
        ''' input `x` is (nkpts,norb,norb)
            output `hx` is (nkpts,norb,norb)
        '''

        hx = numpy.zeros(x.shape, dtype=numpy.complex128)

        projx = numpy.ascontiguousarray(proj_op(x).transpose(1,2,3,0))  # ki,j,x

        # disconnected
        j0 = pop0exp2 * numpy.ascontiguousarray(lib.einsum('kjjx->jx', projx.real))
        j1 = lib.einsum('kijx,jx->kij', projk0, j0)
        hx += -4 * exponent * (exponent-1) * j1

        # connected symmetric
        hx += -2 * exponent * lib.einsum('kijx,jx->kij', projx, pop0exp1)
        projx = None

        # connected asymmetric
        hx += -exponent * (lib.einsum('kim,kjm->kij', Gk, x.conj())
                           + lib.einsum('kmi,kmj->kij', x.conj(), Gk))

        for hxk in hx:
            numpy.fill_diagonal(hxk, numpy.diag(hxk)*0.5)
            hxk -= hxk.conj().T

        return hx

    return g, h_op, h_diag

def get_grad(mlo, u=None, projk0=None):
    ''' Get matrix form of grad without packing.
        Output `g` is (nkpts,norb,norb)
    '''
    if projk0 is None:
        mo_coeff = mlo.rotate_orb(u)
        projk0 = mlo.atomic_pops(mlo.cell, mo_coeff, mode='k0')
        projk0 = numpy.ascontiguousarray(projk0.transpose(1,2,3,0))

    exponent = mlo.exponent
    pop0exp1 = numpy.ascontiguousarray(lib.einsum('kiix->ix', projk0.real))**(exponent-1)
    g = -lib.einsum('kijx,jx->kij', projk0, pop0exp1)

    for gk in g:
        numpy.fill_diagonal(gk, numpy.diag(gk)*0.5)
        gk -= gk.conj().T

    return 2 * exponent * g


class KptsPipekMezey(KptsOrbitalLocalizer):
    '''The Pipek-Mezey localization optimizer that maximizes the orbital
    population for periodic systems using complex rotations.

    Args:
        cell : Cell object

        mo_coeff : size (Nk,n,n) numpy.array
            The orbital space to localize for PM localization.
            When initializing the localization optimizer ``bopt = KPM(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

        kpts : size (Nk,3) numpy.array or KPoints object
            k-points corresponding to the input orbitals.

            If ``kpts`` has no symmetry (type = numpy.array), ``mo_coeff`` must have
            len(mo_coeff) = len(kpts)

            If ``kpts`` has time-reversal symmetry (TRS), ``mo_coeff`` must have
            len(mo_coeff) = kpts.nkpts_ibz. In this case, TRS will be removed from
            ``mo_coeff``.

    Attributes for PM class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        pop_method : str
            How the orbital population is calculated, see JCTC 10, 642
            (2014) for discussion. Options are:
            - 'meta-lowdin' (default) as defined in JCTC 10, 3784 (2014)
            - 'mulliken' original Pipek-Mezey scheme, JCP 90, 4916 (1989)
            - 'lowdin' Lowdin charges, JCTC 10, 642 (2014)
            - 'iao' or 'ibo' intrinsic atomic orbitals with symmetric (i.e., Lowdin)
              orthogonalization, JCTC 9, 4384 (2013)
            - 'iao-biorth' biorthogonalized IAOs, JPCA 128, 8570 (2024)
            - 'becke' Becke charges, JCTC 10, 642 (2014)
            The IAO and Becke charges do not depend explicitly on the
            basis set, and have a complete basis set limit [JCTC 10,
            642 (2014)].
        exponent : int
            The power to define norm. It can be any integer >= 2. Default 2.
        algorithm : str
            Algorithm for maximizing the PM metric function. Currently support
            'ciah' and 'bfgs'. Default 'ciah'.
        minao : str or basis
            MINAO for constructing IAO. This switch only affects calculations with
            `pop_method` = 'iao'/'ibo'/'iao-biorth'. Default 'minao'.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''


    pop_method = getattr(__config__, 'pbc_lo_kpipek_Kpipek_pop_method', 'meta_lowdin')
    conv_tol = getattr(__config__, 'pbc_lo_kpipek_Kpipek_conv_tol', 1e-6)
    exponent = getattr(__config__, 'pbc_lo_kpipek_Kpipek_exponent', 2)  # any integer >= 2
    minao = getattr(__config__, 'pbc_lo_kpipek_Kpipek_minao', 'minao')  # allow user defined MINAO

    _keys = {'pop_method', 'conv_tol', 'exponent', '_proj_data'}

    def __init__(self, cell, mo_coeff, kpts, pop_method=None):
        KptsOrbitalLocalizer.__init__(self, cell, mo_coeff, kpts)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self._proj_data = None

    def dump_flags(self, verbose=None):
        KptsOrbitalLocalizer.dump_flags(self, verbose)
        logger.info(self, 'pop_method = %s',self.pop_method)
        logger.info(self, 'exponent = %s',self.exponent)

    def get_proj_data(self, cell=None, mo_coeff=None, method=None, kpts=None, minao=None):
        if cell is None: cell = self.cell
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if method is None: method = self.pop_method.lower().replace('_', '-')
        if kpts is None: kpts = self.kpts
        if minao is None: minao = self.minao

        log = logger.new_logger(self, verbose=self.verbose-1)
        cput0 = (logger.process_clock(), logger.perf_counter())

        proj_data = get_proj_data(cell, mo_coeff, method, kpts, minao=minao)

        log.timer('get_proj_data', *cput0)

        return proj_data

    def gen_g_hop(self, u=None):
        g, h_op0, h_diag = gen_g_hop(self, u)

        g = self.pack_uniq_var(g)
        h_diag = self.pack_uniq_var(h_diag)
        def h_op(x):
            x = self.unpack_uniq_var(x)
            hx = h_op0(x)
            return self.pack_uniq_var(hx)
        return g, h_op, h_diag


    def get_grad(self, u=None, projk0=None):
        g = get_grad(self, u, projk0)
        return self.pack_uniq_var(g)

    def cost_function(self, u=None):
        mo_coeff = self.rotate_orb(u)
        proj00 = self.atomic_pops(self.cell, mo_coeff, mode='00')
        return (lib.einsum('xii->xi', proj00.real)**self.exponent).sum()

    @lib.with_doc(atomic_pops.__doc__)
    def atomic_pops(self, cell, mo_coeff, kpts=None, mode='kk', method=None,
                    proj_data=None, verbose=None):
        if kpts is None: kpts = self.kpts
        if method is None: method = self.pop_method
        if proj_data is None: proj_data = self._proj_data
        if verbose is None: verbose = self.verbose

        return atomic_pops(self.cell, mo_coeff, self.kpts, mode=mode, method=method,
                           proj_data=proj_data, verbose=verbose)

    def kernel(self, mo_coeff=None, callback=None, verbose=None):
        self._proj_data = self.get_proj_data()
        mo_coeff = boys.kernel(self, mo_coeff, callback, verbose)
        self._proj_data = None

        return mo_coeff

    def stability_jacobi(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = pipek_stability_jacobi(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res

    def stability(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_newton(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res

KPM = KPipek = KPipekMezey = KptsPipekMezey


class KptsPipekMezeyReal(KptsOrbitalLocalizerReal,KptsPipekMezey):

    def __init__(self, cell, mo_coeff, kpts, pop_method=None):
        KptsOrbitalLocalizerReal.__init__(self, cell, mo_coeff, kpts)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self._proj_data = None

    def gen_g_hop(self, u=None):
        g, h_op0, h_diag = gen_g_hop(self, u)

        kpts_symm = self.kpts_symm
        g = self.pack_uniq_var(_pack_bz2ibz(g, kpts_symm))
        h_diag = self.pack_uniq_var(h_diag)
        def h_op(x):
            x = _unpack_ibz2bz(self.unpack_uniq_var(x), kpts_symm)
            hx = h_op0(x)
            return self.pack_uniq_var(_pack_bz2ibz(hx, kpts_symm))

        return g, h_op, h_diag

    def get_grad(self, u=None, projk0=None):
        g = get_grad(self, u, projk0)
        return self.pack_uniq_var(_pack_bz2ibz(g, self.kpts_symm))

KPMReal = KPipekReal = KPipekMezeyReal = KptsPipekMezeyReal



if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.lo.tools import findiff_grad, findiff_hess

    cell = gto.Cell()
    cell.atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    cell.a = numpy.eye(3) * 5
    cell.basis = 'ccpvdz'
    cell.build()

    kpts = cell.make_kpts([3,1,1])

    mf = scf.KRHF(cell, kpts=kpts).density_fit().run()

    log = logger.new_logger(cell, verbose=6)

    mo = [mok[:,:cell.nelectron//2] for mok in mf.mo_coeff]
    mlo = KPM(cell, mo, kpts)

    # Validate gradient and Hessian against finite difference
    g, h_op, hdiag = mlo.gen_g_hop()

    h = numpy.zeros((mlo.pdim,mlo.pdim))
    x0 = mlo.zero_uniq_var()
    for i in range(mlo.pdim):
        x0[i] = 1
        h[:,i] = h_op(x0)
        x0[i] = 0

    def func(x):
        u = mlo.extract_rotation(x)
        f = mlo.cost_function(u)
        if mlo.maximize:
            return -f
        else:
            return f

    def fgrad(x):
        u = mlo.extract_rotation(x)
        return mlo.get_grad(u)

    g_num = findiff_grad(func, x0)
    h_num = findiff_hess(fgrad, x0)
    hdiag_num = numpy.diag(h_num)

    log.info('Grad  error: %.3e', abs(g-g_num).max())
    log.info('Hess  error: %.3e', abs(h-h_num).max())
    log.info('Hdiag error: %.3e', abs(hdiag-hdiag_num).max())

    # localization + stability check using CIAH
    mlo.verbose = 4
    mlo.algorithm = 'ciah'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    # localization + Jacobi-based stability check using BFGS
    mlo.algorithm = 'bfgs'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability_jacobi(return_status=True)
        if stable:
            break
        mlo.kernel(mo)
