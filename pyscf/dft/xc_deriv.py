#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

'''
Transform XC functional derivatives between different representations
'''

import math
import itertools
from functools import lru_cache
import ctypes
import numpy as np
from pyscf import lib

libdft = lib.load_library('libdft')


def transform_vxc(rho, vxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,N]
            GGA : [2,4,N]
            MGGA: [2,5,N]
        * spin unpolarized
            LDA : [1,N]
            GGA : [4,N]
            MGGA: [5,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fr = vxc[0].T
        fg = vxc[1].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fr = vxc[0].T
        fg = vxc[1].T
        ft = vxc[3].T
    else:  # LDA
        order = 0
        nvar = 1
        fr = vxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = fr.reshape(2, nvar, ngrids)
        else:
            vp = np.empty((2,nvar, ngrids))
            vp[:,0] = fr
            #:vp[:,1:4] = np.einsum('abg,bxg->axg', _stack_fg(fg), rho[:,1:4])
            vp[:,1:4] = _stack_fg(fg, rho=rho)
        if order > 1:
            vp[:,4] = ft
    else:
        if order == 0:
            vp = fr.reshape(nvar, ngrids)
        else:
            vp = np.empty((nvar, ngrids))
            vp[0] = fr
            vp[1:4] = 2 * fg * rho[1:4]
        if order > 1:
            vp[4] = ft
    return vp

def transform_fxc(rho, vxc, fxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,N]
            GGA : [2,4,2,4,N]
            MGGA: [2,5,2,5,N]
        * spin unpolarized
            LDA : [1,1,N]
            GGA : [4,4,N]
            MGGA: [5,5,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fg = vxc[1].T
        frr = fxc[0].T
        frg = fxc[1].T
        fgg = fxc[2].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fg = vxc[1].T
        frr, frg, fgg, ftt, frt, fgt = [fxc[i].T for i in [0, 1, 2, 4, 6, 9]]
    else:  # LDA
        order = 0
        nvar = 1
        frr = fxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = _stack_frr(frr).reshape(2,nvar, 2,nvar, ngrids).transpose(1,3,0,2,4)
        else:
            vp = np.empty((2,nvar, 2,nvar, ngrids)).transpose(1,3,0,2,4)
            vp[0,0] = _stack_frr(frr)
            i3 = np.arange(3)
            #:qgg = _stack_fgg(fgg)
            #:qgg = np.einsum('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            #:qgg = np.einsum('xbcdg,cyg->xybdg', qgg, rho[:,1:4])
            qgg = _stack_fgg(fgg, rho=rho).transpose(1,3,0,2,4)
            qgg[i3,i3] += _stack_fg(fg)
            vp[1:4,1:4] = qgg

            frg = frg.reshape(2,3,ngrids)
            #:qrg = _stack_fg(frg, axis=1)
            #:qrg = np.einsum('rabg,axg->xrbg', qrg, rho[:,1:4])
            qrg = _stack_fg(frg, axis=1, rho=rho).transpose(2,0,1,3)
            vp[0,1:4] = qrg
            vp[1:4,0] = qrg.transpose(0,2,1,3)

        if order > 1:
            fgt = fgt.reshape(3,2,ngrids)
            #:qgt = _stack_fg(fgt, axis=0)
            #:qgt = np.einsum('abrg,axg->xbrg', qgt, rho[:,1:4])
            qgt = _stack_fg(fgt, axis=0, rho=rho).transpose(1,0,2,3)
            vp[1:4,4] = qgt
            vp[4,1:4] = qgt.transpose(0,2,1,3)

            qrt = frt.reshape(2,2,ngrids)
            vp[0,4] = qrt
            vp[4,0] = qrt.transpose(1,0,2)

            vp[4,4] = _stack_frr(ftt)

        vp = vp.transpose(2,0,3,1,4)

    else:
        if order == 0:
            vp = frr.reshape(nvar, nvar, ngrids)
        else:
            vp = np.empty((nvar, nvar, ngrids))
            vp[0,0] = frr
            i3 = np.arange(3)
            qgg = 4 * fgg * rho[1:4] * rho[1:4,None]
            qgg[i3,i3] += fg * 2
            vp[1:4,1:4] = qgg
            vp[0,1:4] = vp[1:4,0] = 2 * frg * rho[1:4]
        if order > 1:
            vp[4,1:4] = vp[1:4,4] = 2 * fgt * rho[1:4]
            vp[0,4] = frt
            vp[4,0] = frt
            vp[4,4] = ftt
    return vp

def transform_kxc(rho, fxc, kxc, xctype, spin=0):
    r'''
    Transform libxc functional derivatives to the derivative tensor of
    parameters in rho:
        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,2,1,N]
            GGA : [2,4,2,4,2,4,N]
            MGGA: [2,5,2,5,2,5,N]
        * spin unpolarized
            LDA : [1,1,1,N]
            GGA : [4,4,4,N]
            MGGA: [5,5,5,N]
    '''
    rho = np.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        frg = fxc[1].T
        fgg = fxc[2].T
        frrr, frrg, frgg, fggg = [x.T for x in kxc[:4]]
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        frg = fxc[1].T
        fgg = fxc[2].T
        fgt = fxc[9].T
        frrr, frrg, frgg, fggg, frrt, frgt, frtt, fggt, fgtt, fttt = \
                [kxc[i].T for i in [0, 1, 2, 3, 5, 7, 10, 12, 15, 19]]
    else:  # LDA
        order = 0
        nvar = 1
        frrr = kxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = _stack_frrr(frrr).reshape(2,nvar, 2,nvar, 2,nvar, ngrids).transpose(1,3,5,0,2,4,6)
        else:
            vp = np.empty((2,nvar, 2,nvar, 2,nvar, ngrids)).transpose(1,3,5,0,2,4,6)
            vp[0,0,0] = _stack_frrr(frrr)
            i3 = np.arange(3)
            #:qggg = _stack_fggg(fggg)
            #:qggg = np.einsum('abcdefg,axg->xbcdefg', qggg, rho[:,1:4])
            #:qggg = np.einsum('xbcdefg,cyg->xybdefg', qggg, rho[:,1:4])
            #:qggg = np.einsum('xybdefg,ezg->xyzbdfg', qggg, rho[:,1:4])
            qggg = _stack_fggg(fggg, rho=rho).transpose(1,3,5,0,2,4,6)
            qgg = _stack_fgg(fgg)
            qgg = np.einsum('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            for i in range(3):
                qggg[:,i,i] += qgg
                qggg[i,:,i] += qgg.transpose(0,2,1,3,4)
                qggg[i,i,:] += qgg.transpose(0,2,3,1,4)
            vp[1:4,1:4,1:4] = qggg

            frgg = frgg.reshape(2,6,ngrids)
            #:qrgg = _stack_fgg(frgg, axis=1)
            #:qrgg = np.einsum('rabcdg,axg->xrbcdg', qrgg, rho[:,1:4])
            #:qrgg = np.einsum('xrbcdg,cyg->xyrbdg', qrgg, rho[:,1:4])
            qrgg = _stack_fgg(frgg, axis=1, rho=rho).transpose(2,4,0,1,3,5)
            qrg = _stack_fg(frg.reshape(2,3,ngrids), axis=1)
            qrgg[i3,i3] += qrg
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg.transpose(0,1,3,2,4,5)
            vp[1:4,1:4,0] = qrgg.transpose(0,1,3,4,2,5)

            frrg = frrg.reshape(3,3,ngrids)
            qrrg = _stack_frr(frrg, axis=0)
            #:qrrg = _stack_fg(qrrg, axis=2)
            #:qrrg = np.einsum('rsabg,axg->rsxbg', qrrg, rho[:,1:4])
            qrrg = _stack_fg(qrrg, axis=2, rho=rho).transpose(3,0,1,2,4)
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg.transpose(0,1,3,2,4)
            vp[1:4,0,0] = qrrg.transpose(0,3,1,2,4)

        if order > 1:
            fggt = fggt.reshape(6,2,ngrids)
            #:qggt = _stack_fgg(fggt, axis=0)
            #:qggt = np.einsum('abcdrg,axg->xbcdrg', qggt, rho[:,1:4])
            #:qggt = np.einsum('xbcdrg,cyg->xybdrg', qggt, rho[:,1:4])
            qggt = _stack_fgg(fggt, axis=0, rho=rho).transpose(1,3,0,2,4,5)
            qgt = _stack_fg(fgt.reshape(3,2,ngrids), axis=0)
            i3 = np.arange(3)
            qggt[i3,i3] += qgt
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt.transpose(0,1,2,4,3,5)
            vp[4,1:4,1:4] = qggt.transpose(0,1,4,2,3,5)

            qgtt = _stack_frr(fgtt.reshape(3,3,ngrids), axis=1)
            #:qgtt = _stack_fg(qgtt, axis=0)
            #:qgtt = np.einsum('abrsg,axg->xbrsg', qgtt, rho[:,1:4])
            qgtt = _stack_fg(qgtt, axis=0, rho=rho).transpose(1,0,2,3,4)
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt.transpose(0,2,1,3,4)
            vp[4,4,1:4] = qgtt.transpose(0,2,3,1,4)

            frgt = frgt.reshape(2,3,2,ngrids)
            #:qrgt = _stack_fg(frgt, axis=1)
            #:qrgt = np.einsum('rabsg,axg->xrbsg', qrgt, rho[:,1:4])
            qrgt = _stack_fg(frgt, axis=1, rho=rho).transpose(2,0,1,3,4)
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt.transpose(0,1,3,2,4)
            vp[1:4,0,4] = qrgt.transpose(0,2,1,3,4)
            vp[4,0,1:4] = qrgt.transpose(0,3,1,2,4)
            vp[1:4,4,0] = qrgt.transpose(0,2,3,1,4)
            vp[4,1:4,0] = qrgt.transpose(0,3,2,1,4)

            qrrt = _stack_frr(frrt.reshape(3,2,ngrids), axis=0)
            vp[0,0,4] = qrrt
            vp[0,4,0] = qrrt.transpose(0,2,1,3)
            vp[4,0,0] = qrrt.transpose(2,0,1,3)

            qrtt = _stack_frr(frtt.reshape(2,3,ngrids), axis=1)
            vp[0,4,4] = qrtt
            vp[4,0,4] = qrtt.transpose(1,0,2,3)
            vp[4,4,0] = qrtt.transpose(1,2,0,3)

            vp[4,4,4] = _stack_frrr(fttt, axis=0)

        vp = vp.transpose(3,0,4,1,5,2,6)

    else:
        if order == 0:
            vp = frrr.reshape(nvar, nvar, nvar, ngrids)
        else:
            vp = np.empty((nvar, nvar, nvar, ngrids))
            vp[0,0,0] = frrr
            i3 = np.arange(3)
            qggg = 8 * fggg * rho[1:4] * rho[1:4,None] * rho[1:4,None,None]
            qgg = 4 * fgg * rho[1:4]
            for i in range(3):
                qggg[i,i,:] += qgg
                qggg[i,:,i] += qgg
                qggg[:,i,i] += qgg
            vp[1:4,1:4,1:4] = qggg

            qrgg = 4 * frgg * rho[1:4] * rho[1:4,None]
            qrgg[i3,i3] += frg * 2
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg
            vp[1:4,1:4,0] = qrgg

            qrrg = 2 * frrg * rho[1:4]
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg
            vp[1:4,0,0] = qrrg

        if order > 1:
            qggt = 4 * fggt * rho[1:4] * rho[1:4,None]
            qggt[i3,i3] += fgt * 2
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt
            vp[4,1:4,1:4] = qggt

            qgtt = 2 * fgtt * rho[1:4]
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt
            vp[4,4,1:4] = qgtt

            qrgt = 2 * frgt * rho[1:4]
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt
            vp[1:4,0,4] = qrgt
            vp[4,0,1:4] = qrgt
            vp[1:4,4,0] = qrgt
            vp[4,1:4,0] = qrgt

            vp[0,0,4] = frrt
            vp[0,4,0] = frrt
            vp[4,0,0] = frrt
            vp[0,4,4] = frtt
            vp[4,0,4] = frtt
            vp[4,4,0] = frtt
            vp[4,4,4] = fttt
    return vp

def transform_lxc(rho, fxc, kxc, lxc, xctype, spin=0):
    r'''
    Transform libxc vxc functional output to the derivative tensor of parameters
    in rho: [density, nabla_x, nabla_y, nabla_z, tau]. The output tensor has the
    shape:
        * spin polarized
            LDA : [2,1,2,1,2,1,2,1,N]
            GGA : [2,4,2,4,2,4,2,4,N]
            MGGA: [2,5,2,5,2,5,2,5,N]
        * spin unpolarized
            LDA : [1,1,1,1,N]
            GGA : [4,4,4,4,N]
            MGGA: [5,5,5,5,N]
    '''
    raise NotImplementedError

def _stack_fg(fg, axis=0, rho=None, out=None):
    '''fg [uu, ud, dd] -> [[uu*2, ud], [du, dd*2]]'''
    if rho is None:
        qg = _stack_frr(fg, axis)
        if axis == 0:
            qg[0,0] *= 2
            qg[1,1] *= 2
        elif axis == 1:
            qg[:,0,0] *= 2
            qg[:,1,1] *= 2
        elif axis == 2:
            qg[:,:,0,0] *= 2
            qg[:,:,1,1] *= 2
        else:
            raise NotImplementedError
        return qg
    else:
        rho = np.asarray(rho, order='C')
        fg = np.asarray(fg, order='C')
        if fg.shape[axis] != 3:
            fg = fg.reshape(fg.shape[:axis] + (3, -1) + fg.shape[axis+1:])
        nvar, ngrids = rho.shape[1:]
        ncounts = int(np.prod(fg.shape[:axis]))
        mcounts = int(np.prod(fg.shape[axis+1:-1]))
        qg = np.empty(fg.shape[:axis] + (2, 3) + fg.shape[axis+1:])
        rho = libdft.VXCfg_to_direct_deriv(
            qg.ctypes.data_as(ctypes.c_void_p),
            fg.ctypes.data_as(ctypes.c_void_p),
            rho.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ncounts), ctypes.c_int(nvar), ctypes.c_int(mcounts),
            ctypes.c_int(ngrids))
        return qg


def _stack_frr(frr, axis=0):
    '''frr [u_u, u_d, d_d] -> [[u_u, u_d], [d_u, d_d]]'''
    if frr.shape[axis] != 3:
        frr = frr.reshape(frr.shape[:axis] + (3, -1) + frr.shape[axis+1:])
    slices = [slice(None)] * frr.ndim
    slices[axis] = [[0,1],[1,2]]
    return frr[tuple(slices)]

def _stack_fgg(fgg, axis=0, rho=None):
    '''
    fgg [uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd] ->
    [[uu_uu, ud_ud, ud_dd],
     [ud_uu, ud_ud, ud_dd],
     [dd_uu, dd_ud, dd_dd]] -> tensor with shape [2,2, 2,2, ...]
    '''
    if fgg.shape[axis] != 6:
        fgg = fgg.reshape(fgg.shape[:axis] + (6, -1) + fgg.shape[axis+1:])
    slices = [slice(None)] * fgg.ndim
    slices[axis] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    fgg = fgg[tuple(slices)]
    return _stack_fg(_stack_fg(fgg, axis=axis+1, rho=rho), axis=axis, rho=rho)

def _stack_frrr(frrr, axis=0):
    '''
    frrr [u_u_u, u_u_d, u_d_d, d_d_d]
    -> tensor with shape [2, 2, 2, ...]
    '''
    if frrr.shape[axis] != 4:
        frrr = frrr.reshape(frrr.shape[:axis] + (4, -1) + frrr.shape[axis+1:])
    slices = [slice(None)] * frrr.ndim
    slices[axis] = [[[0, 1], [1, 2]],
                    [[1, 2], [2, 3]]]
    return frrr[tuple(slices)]

def _stack_fggg(fggg, axis=0, rho=None):
    '''
    fggg [uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd]
    -> tensor with shape [2,2, 2,2, 2,2, ...]
    '''
    if fggg.shape[axis] != 10:
        fggg = fggg.reshape(fggg.shape[:axis] + (10, 2) + fggg.shape[axis+1:])
    slices = [slice(None)] * fggg.ndim
    slices[axis] = [[[0, 1, 2], [1, 3, 4], [2, 4, 5]],
                    [[1, 3, 4], [3, 6, 7], [4, 7, 8]],
                    [[2, 4, 5], [4, 7, 8], [5, 8, 9]]]
    fggg = fggg[tuple(slices)]
    fggg = _stack_fg(fggg, axis=axis+2, rho=rho)
    fggg = _stack_fg(fggg, axis=axis+1, rho=rho)
    return _stack_fg(fggg, axis=axis, rho=rho)

@lru_cache(100)
def _product_uniq_indices(nvars, order):
    '''
    Indexing the symmetry unique elements in cartesian product
    '''
    # n = 0
    # for i range(nvars):
    #    for j in range(i, nvars):
    #        for k in range(k, nvars):
    #            ...
    #            prod[(i,j,k,...)] = n
    #            n += 1
    prod = np.ones([nvars] * order, dtype=int)
    uniq_idx = list(itertools.combinations_with_replacement(range(nvars), order))
    prod[tuple(np.array(uniq_idx).T)] = np.arange(len(uniq_idx))

    idx = np.where(prod)
    sidx = np.sort(idx, axis=0)
    prod[idx] = prod[tuple(sidx)]
    return prod

def _pair_combinations(lst):
    n = len(lst)
    if n <= 2:
        return [[tuple(lst)]]
    a = lst[0]
    results = []
    for i in range(1, n):
        pair = (a, lst[i])
        rests = _pair_combinations(lst[1:i]+lst[i+1:])
        for rest in rests:
            rest.append(pair)
        results.extend(rests)
    return results

def _unfold_gga(rho, xc_val, spin, order, nvar, xlen, reserve=0):
    assert nvar >= 4
    ngrids = rho.shape[-1]
    n_transform = order - reserve

    if spin == 0:
        nvar2 = nvar
        drv = libdft.VXCunfold_sigma_spin0
    else:
        nvar2 = nvar * 2
        drv = libdft.VXCunfold_sigma_spin1

    # xc_val[idx] expands unique xc elements to n-order tensors
    idx = _product_uniq_indices(xlen, order)
    xc_tensor = np.empty((xlen**reserve * nvar2**n_transform, ngrids))
    xc_tensor[:idx.size] = xc_val[idx.ravel()]
    buf = np.empty_like(xc_tensor)
    for i in range(n_transform):
        xc_tensor, buf = buf, xc_tensor
        ncounts = xlen**(order-1-i) * nvar2**i
        drv(xc_tensor.ctypes, buf.ctypes, rho.ctypes,
            ctypes.c_int(ncounts), ctypes.c_int(nvar), ctypes.c_int(ngrids))
    if spin == 0:
        xc_tensor = xc_tensor.reshape([xlen]*reserve + [nvar]*n_transform + [ngrids])
    else:
        xc_tensor = xc_tensor.reshape([xlen]*reserve + [2,nvar]*n_transform + [ngrids])
    return xc_tensor

def _diagonal_indices(idx, order):
    assert order > 0
    n = len(idx)
    diag_idx = [np.asarray(idx)]
    for i in range(1, order):
        last_dim = diag_idx[-1]
        diag_idx = [np.repeat(x, n) for x in diag_idx]
        diag_idx.append(np.tile(last_dim, n))
    # repeat(diag_idx, 2)
    return tuple(x for x in diag_idx for i in range(2))

_XC_NVAR = {
    ('HF', 0): (1, 1),
    ('HF', 1): (1, 2),
    ('LDA', 0): (1, 1),
    ('LDA', 1): (1, 2),
    ('GGA', 0): (4, 2),
    ('GGA', 1): (4, 5),
    ('MGGA', 0): (5, 3),
    ('MGGA', 1): (5, 7),
}

def transform_xc(rho, xc_val, xctype, spin, order):
    '''General transformation to construct XC derivative tensor'''
    xc_val = np.asarray(xc_val, order='C')
    if order == 0:
        return xc_val[0]

    nvar, xlen = _XC_NVAR[xctype, spin]
    ngrids = xc_val.shape[-1]
    offsets = [0] + [count_combinations(xlen, o) for o in range(order+1)]
    p0, p1 = offsets[order:order+2]
    if xctype == 'LDA' or xctype == 'HF':
        xc_out = xc_val[p0:p1]
        if spin == 0:
            return xc_out.reshape([1]*order + [ngrids])
        else:
            idx = _product_uniq_indices(xlen, order)
            return xc_out[idx].reshape([2,1]*order + [ngrids])

    rho = np.asarray(rho, order='C')
    assert rho.size == (spin+1) * nvar * ngrids
    xc_tensor = _unfold_gga(rho, xc_val[p0:p1], spin, order, nvar, xlen)
    nabla_idx = np.arange(1, 4)
    if spin == 0:
        for n_pairs in range(1, order//2+1):
            p0, p1 = offsets[order-n_pairs:order-n_pairs+2]
            xc_sub = _unfold_gga(rho, xc_val[p0:p1], spin, order-n_pairs,
                                 nvar, xlen, n_pairs)
            # Just the sigma components
            xc_sub = xc_sub[(1,)*n_pairs] * 2**n_pairs

            # low_sigmas indicates the index for the extra sigma terms
            low_sigmas = itertools.combinations(range(order), n_pairs*2)
            pair_combs = [list(itertools.chain(*p[::-1]))
                          for p in _pair_combinations(list(range(n_pairs*2)))]
            diag_idx = _diagonal_indices(nabla_idx, n_pairs)

            for dim_lst in low_sigmas:
                rest_dims = [i for i in range(xc_tensor.ndim) if i not in dim_lst]
                for pair_comb in pair_combs:
                    xc_tensor_1 = xc_tensor.transpose(
                        [dim_lst[i] for i in pair_comb] + rest_dims)
                    xc_tensor_1[diag_idx] += xc_sub
    else:
        i3to2x2 = _product_uniq_indices(2, 2)
        for n_pairs in range(1, order//2+1):
            p0, p1 = offsets[order-n_pairs:order-n_pairs+2]
            xc_sub = _unfold_gga(rho, xc_val[p0:p1], spin, order-n_pairs,
                                 nvar, xlen, n_pairs)
            # Just the sigma components
            xc_sub = xc_sub[(slice(2,5),) * n_pairs]
            for i in range(n_pairs):
                xc_sub[(slice(None),)*i+(0,)] *= 2
                xc_sub[(slice(None),)*i+(2,)] *= 2
            sigma_idx = (i3to2x2[(slice(None),)*2 + (np.newaxis,)*(i*2)]
                         for i in reversed(range(n_pairs)))
            xc_sub = xc_sub[tuple(sigma_idx)]

            low_sigmas = itertools.combinations(range(order), n_pairs*2)
            pair_combs = [list(itertools.chain(*p[::-1]))
                          for p in _pair_combinations(list(range(n_pairs*2)))]
            diag_idx = _diagonal_indices(nabla_idx, n_pairs)

            for dim_lst in low_sigmas:
                dim_lst2 = np.array(dim_lst)*2 + np.array([1, 0])[:,None]
                rest_dims = [i for i in range(xc_tensor.ndim) if i not in dim_lst2]
                for pair_comb in pair_combs:
                    leading_dims = list(dim_lst2[:,pair_comb].ravel())
                    xc_tensor_1 = xc_tensor.transpose(leading_dims + rest_dims)
                    xc_tensor_1[diag_idx] += xc_sub
    return xc_tensor

def count_combinations(nvar, order):
    '''sum(len(combinations_with_replacement(range(nvar), o) for o in range(order))'''
    return lib.comb(nvar+order, order)

def ud2ts(v_ud):
    '''XC derivatives spin-up/spin-down representations to
    total-density/spin-density representations'''
    v_ud = np.asarray(v_ud, order='C')
    v_ts = np.empty_like(v_ud)
    nvar, ngrids = v_ts.shape[-2:]
    nvar2 = nvar * 2
    order = v_ud.ndim // 2
    drv = libdft.VXCud2ts
    v_ts, v_ud = v_ud, v_ts
    for i in range(order):
        v_ts, v_ud = v_ud, v_ts
        n = nvar2**i
        nvg = nvar * nvar2**(order-1-i) * ngrids
        #:c = np.array([[.5,  .5],    # vrho = (va + vb) / 2
        #:              [.5, -.5]])   # vs   = (va - vb) / 2
        #:v_ts = np.einsum('ra,...ax...g->...rx...g', c, v_ud)
        drv(v_ts.ctypes, v_ud.ctypes, ctypes.c_int(n), ctypes.c_int(nvg))
    return v_ts

def ts2ud(v_ts):
    '''XC derivatives total-density/spin-density representations to
    spin-up/spin-down representations'''
    v_ts = np.asarray(v_ts, order='C')
    v_ud = np.empty_like(v_ts)
    nvar, ngrids = v_ts.shape[-2:]
    nvar2 = nvar * 2
    order = v_ud.ndim // 2
    drv = libdft.VXCts2ud
    v_ts, v_ud = v_ud, v_ts
    for i in range(order):
        v_ts, v_ud = v_ud, v_ts
        n = nvar2**i
        nvg = nvar * nvar2**(order-1-i) * ngrids
        #:c = np.array([[1,  1],    # va = vrho + vs
        #:              [1, -1]])   # vb = vrho - vs
        #:v_ud = np.einsum('ra,...ax...g->...rx...g', c, v_ts)
        drv(v_ud.ctypes, v_ts.ctypes, ctypes.c_int(n), ctypes.c_int(nvg))
    return v_ud
