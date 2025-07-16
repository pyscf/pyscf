#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
Numerical integration functions for (2-component) GKS with real AO basis
'''

import functools
import numpy as np
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao, _tau_dot, BLKSIZE
from pyscf.dft import xc_deriv
from pyscf import __config__


@lib.with_doc(
    '''Calculate the electron density and magnetization spin density in the
    framework of 2-component real basis.
    ''' + numint.eval_rho.__doc__)
def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    nao = ao.shape[-1]
    assert dm.ndim == 2 and nao * 2 == dm.shape[0]

    ngrids, nao = ao.shape[-2:]
    xctype = xctype.upper()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc

    if xctype == 'LDA':
        c0a = _dot_ao_dm(mol, ao, dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao, dm[nao:], non0tab, shls_slice, ao_loc)
        rho_m = _contract_rho_m((ao, ao), (c0a, c0b), hermi, True)
    elif xctype == 'GGA':
        # first 4 ~ (rho, m), second 4 ~ (0th order, dx, dy, dz)
        if hermi:
            rho_m = np.empty((4, 4, ngrids))
        else:
            rho_m = np.empty((4, 4, ngrids), dtype=np.complex128)
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi, True)
        for i in range(1, 4):
            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi, False)
        if hermi:
            rho_m[:,1:4] *= 2  # *2 for |ao> dm < dx ao| + |dx ao> dm < ao|
        else:
            for i in range(1, 4):
                c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
                c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi, False)
    else: # meta-GGA
        if hermi:
            dtype = np.double
        else:
            dtype = np.complex128
        if with_lapl:
            rho_m = np.empty((4, 6, ngrids), dtype=dtype)
            tau_idx = 5
        else:
            rho_m = np.empty((4, 5, ngrids), dtype=dtype)
            tau_idx = 4
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi, True)
        rho_m[:,tau_idx] = 0
        for i in range(1, 4):
            c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
            c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
            rho_m[:,tau_idx] += _contract_rho_m((ao[i], ao[i]), (c1a, c1b), hermi, True)

            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi, False)
            if hermi:
                rho_m[:,i] *= 2
            else:
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi, False)
        if with_lapl:
            # TODO: rho_m[:,4] = \nabla^2 rho
            raise NotImplementedError
        # tau = 1/2 (\nabla f)^2
        rho_m[:,tau_idx] *= .5
    return rho_m

def _gks_mcol_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                  max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    make_rho, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi, False)
    nao = n2c // 2

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('LDA' , 'm'): (_mcol_lda_vxc_mat , 0),
            ('GGA' , 'm'): (_mcol_gga_vxc_mat , 1),
            ('MGGA', 'm'): (_mcol_mgga_vxc_mat, 1),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

        if ni.collinear[0] == 'm':  # mcol
            eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
        else:
            eval_xc = ni.eval_xc_eff

        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
                if xctype == 'LDA':
                    den = rho[0] * weight
                else:
                    den = rho[0,0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                vmat[i] += fmat(mol, ao, weight, rho, vxc, mask, shls_slice,
                                ao_loc, hermi)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_vxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat

def _gks_mcol_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    assert ni.collinear[0] == 'm'  # mcol
    xctype = ni._xc_type(xc_code)
    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0,
                                  max_memory=max_memory)[2]

    if xctype == 'MGGA':
        fmat, ao_deriv = (_mcol_mgga_fxc_mat , 1)
    elif xctype == 'GGA':
        fmat, ao_deriv = (_mcol_gga_fxc_mat  , 1)
    else:
        fmat, ao_deriv = (_mcol_lda_fxc_mat  , 0)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    make_rho1, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi, False)
    nao = n2c // 2
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        _rho0 = None
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,:,:,p0:p1]
            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                vmat[i] += fmat(mol, ao, weight, _rho0, rho1, _fxc,
                                mask, shls_slice, ao_loc, hermi)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def _ncol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of non-collinear LDA'''
    # NOTE vxc in u/d representation
    r, mx, my, mz = rho
    vxc = xc_deriv.ud2ts(vxc)
    vr, vs = vxc[:,0]
    s = lib.norm(rho[1:4], axis=0)

    wv = weight * vr
    with np.errstate(divide='ignore',invalid='ignore'):
        ws = vs * weight / s
    ws[s < 1e-20] = 0

    # * .5 because of v+v.conj().T in r_vxc
    if hermi:
        wv *= .5
        ws *= .5
    aow = None
    aow = _scale_ao(ao, ws*mx, out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, ws*my, out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if hermi:
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = np.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
    tmpx = tmpy = None
    aow = _scale_ao(ao, wv+ws*mz, out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wv-ws*mz, out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.block([[mataa, matab], [matba, matbb]])
    return mat

def _eval_xc_eff(ni, xc_code, rho, deriv=1, omega=None, xctype=None,
                 verbose=None, spin=None):
    r'''Returns the derivative tensor against the density parameters

    [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
     [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

    It differs from the eval_xc method in the derivatives of non-local part.
    The eval_xc method returns the XC functional derivatives to sigma
    (|\nabla \rho|^2)

    Args:
        rho: 2-dimensional or 3-dimensional array
            Total density and spin density (and their derivatives if GGA or MGGA
            functionals) on grids

    Kwargs:
        deriv: int
            derivative orders
        omega: float
            define the exponent in the attenuated Coulomb for RSH functional
    '''
    if omega is None: omega = ni.omega
    if xctype is None: xctype = ni._xc_type(xc_code)
    if ni.collinear[0] == 'c':  # collinear
        t = rho[0]
        s = rho[3]
    elif ni.collinear[0] == 'n':  # ncol
        t = rho[0]
        m = rho[1:4]
        s = lib.norm(m, axis=0)
    elif ni.collinear[0] == 'm':  # mcol
        # called by mcfun.eval_xc_eff which passes (rho, s) only
        t, s = rho
    else:
        raise RuntimeError(f'Unknown collinear scheme {ni.collinear}')

    rho = np.stack([(t + s) * .5, (t - s) * .5])
    if xctype == 'MGGA' and rho.shape[1] == 6:
        rho = np.asarray(rho[:,[0,1,2,3,5],:], order='C')

    assert spin is None or spin == 1
    spin = 1

    out = ni.libxc.eval_xc1(xc_code, rho, spin, deriv, omega)
    evfk = [out[0]]
    for order in range(1, deriv+1):
        evfk.append(xc_deriv.transform_xc(rho, out, xctype, spin, order))
    if deriv < 3:
        # The return has at least [e, v, f, k] terms
        evfk.extend([None] * (3 - deriv))
    return evfk

# * Mcfun requires functional derivatives to total-density and spin-density.
# * Make it a global function than a closure so as to be callable by multiprocessing
def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    for order in range(1, deriv+1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk

def mcfun_eval_xc_adapter(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun'''
    try:
        import mcfun
    except ImportError:
        raise ImportError('This feature requires mcfun library.\n'
                          'Try install mcfun with `pip install mcfun`')

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()
    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None, spin=None):
        return mcfun.eval_xc_eff(
            fn_eval_xc, rho, deriv, spin_samples=ni.spin_samples,
            collinear_threshold=ni.collinear_thrd,
            collinear_samples=ni.collinear_samples, workers=nproc)
    return eval_xc_eff

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(ao, wmx[0], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wmy[0], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if hermi:
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = np.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
    tmpx = tmpy = None
    aow = _scale_ao(ao, wr[0]+wmz[0], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[0]-wmz[0], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.block([[mataa, matab], [matba, matbb]])
    return mat

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == np.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = np.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    mat = np.block([[mataa, matab], [matba, matbb]])
    return mat

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear MGGA'''
    wv = weight * vxc
    tau_idx = 4
    wv[:,tau_idx] *= .5  # *.5 for 1/2 in tau
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
        wv[:,tau_idx] *= .5
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    mataa += _tau_dot(mol, ao, ao, wr[tau_idx]+wmz[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matbb += _tau_dot(mol, ao, ao, wr[tau_idx]-wmz[tau_idx], mask, shls_slice, ao_loc)

    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpx += _tau_dot(mol, ao, ao, wmx[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpy += _tau_dot(mol, ao, ao, wmy[tau_idx], mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == np.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = np.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    mat = np.block([[mataa, matab], [matba, matbb]])
    return mat

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear LDA'''
    vxc1 = np.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _mcol_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi)

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear GGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi)

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                       mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear MGGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                              ao_loc, hermi)

def _contract_rho_m(bra, ket, hermi=0, bra_eq_ket=False):
    '''
    hermi indicates whether the density matrix is hermitian.
    bra_eq_ket indicates whether bra and ket basis are the same AOs.
    '''
    # rho = einsum('xgi,ij,xgj->g', ket, dm, bra.conj())
    # mx = einsum('xy,ygi,ij,xgj->g', sx, ket, dm, bra.conj())
    # my = einsum('xy,ygi,ij,xgj->g', sy, ket, dm, bra.conj())
    # mz = einsum('xy,ygi,ij,xgj->g', sz, ket, dm, bra.conj())
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    nao = min(ket_a.shape[1], bra_a.shape[1])
    ngrids = ket_a.shape[0]
    if hermi:
        raa = np.einsum('pi,pi->p', bra_a.real, ket_a[:,:nao].real)
        raa+= np.einsum('pi,pi->p', bra_a.imag, ket_a[:,:nao].imag)
        rab = np.einsum('pi,pi->p', bra_a.conj(), ket_b[:,:nao])
        rbb = np.einsum('pi,pi->p', bra_b.real, ket_b[:,nao:].real)
        rbb+= np.einsum('pi,pi->p', bra_b.imag, ket_b[:,nao:].imag)
        rho_m = np.empty((4, ngrids))
        rho_m[0,:] = raa + rbb     # rho
        rho_m[1,:] = rab.real      # mx
        rho_m[2,:] = rab.imag      # my
        rho_m[3,:] = raa - rbb     # mz
        if bra_eq_ket:
            rho_m[1,:] *= 2
            rho_m[2,:] *= 2
        else:
            rba = np.einsum('pi,pi->p', bra_b.conj(), ket_a[:,nao:])
            rho_m[1,:] += rba.real
            rho_m[2,:] -= rba.imag
    else:
        raa = np.einsum('pi,pi->p', bra_a.conj(), ket_a[:,:nao])
        rba = np.einsum('pi,pi->p', bra_b.conj(), ket_a[:,nao:])
        rab = np.einsum('pi,pi->p', bra_a.conj(), ket_b[:,:nao])
        rbb = np.einsum('pi,pi->p', bra_b.conj(), ket_b[:,nao:])
        rho_m = np.empty((4, ngrids), dtype=np.complex128)
        rho_m[0,:] = raa + rbb         # rho
        rho_m[1,:] = rab + rba         # mx
        rho_m[2,:] = (rba - rab) * 1j  # my
        rho_m[3,:] = raa - rbb         # mz
    return rho_m


class NumInt2C(lib.StreamObject, numint.LibXCMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    spin_samples = getattr(__config__, 'dft_numint_RnumInt_spin_samples', 770)
    collinear_thrd = getattr(__config__, 'dft_numint_RnumInt_collinear_thrd', 0.99)
    collinear_samples = getattr(__config__, 'dft_numint_RnumInt_collinear_samples', 200)

    make_mask = staticmethod(numint.make_mask)
    eval_ao = staticmethod(numint.eval_ao)
    eval_rho = staticmethod(eval_rho)

    def eval_rho1(self, mol, ao, dm, screen_index=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=None, ao_cutoff=None, pair_mask=None,
                  verbose=None):
        return self.eval_rho(mol, ao, dm, screen_index, xctype, hermi,
                             with_lapl, verbose=verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  with_lapl=True, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        if self.collinear[0] in ('n', 'm'):
            # TODO:
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            rho = self.eval_rho(mol, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
            return rho

        #if mo_coeff.dtype == np.double:
        #    nao = ao.shape[-1]
        #    assert nao * 2 == mo_coeff.shape[0]
        #    mo_aR = mo_coeff[:nao]
        #    mo_bR = mo_coeff[nao:]
        #    rho  = numint.eval_rho2(mol, ao, mo_aR, mo_occ, non0tab, xctype, with_lapl, verbose)
        #    rho += numint.eval_rho2(mol, ao, mo_bR, mo_occ, non0tab, xctype, with_lapl, verbose)
        #else:
        #    dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
        #    hermi = 1
        #    rho = self.eval_rho(mol, dm, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
        #mx = my = mz = None
        #return rho, mx, my, mz
        raise NotImplementedError(self.collinear)

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = mo_coeff.shape[0]
        nao = n2c // 2
        with_lapl = numint.MGGA_DENSITY_LAPL

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            rho = []
            for ao, mask, weight, coords \
                    in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho.append(self.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype,
                                          with_lapl))
            rho = np.concatenate(rho,axis=-1)
            assert rho.dtype == np.double
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            # rhoa and rhob must be real
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            dm_a = dm[:nao,:nao].real.copy('C')
            dm_b = dm[nao:,nao:].real.copy('C')
            ni = self._to_numint1c()
            hermi = 1
            rhoa = []
            rhob = []
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                # rhoa and rhob must be real
                rhoa.append(ni.eval_rho(mol, ao, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(mol, ao, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def cache_xc_kernel1(self, mol, grids, xc_code, dm, spin=0, max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        else:
            ao_deriv = 0
        n2c = dm.shape[0]
        nao = n2c // 2

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            hermi = 1 # rho must be real. We need to assume dm hermitian
            with_lapl = False
            rho = []
            for ao, mask, weight, coords \
                    in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho.append(self.eval_rho1(mol, ao, dm, mask, xctype, hermi, with_lapl))
            rho = np.concatenate(rho,axis=-1)
            assert rho.dtype == np.double
            if self.collinear[0] == 'm':  # mcol
                eval_xc = self.mcfun_eval_xc_adapter(xc_code)
            else:
                eval_xc = self.eval_xc_eff
            vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        else:
            # rhoa and rhob must be real. We need to assume dm hermitian
            hermi = 1
            # TODO: test if dm[:nao,:nao].imag == 0
            dm_a = dm[:nao,:nao].real.copy('C')
            dm_b = dm[nao:,nao:].real.copy('C')
            ni = self._to_numint1c()
            with_lapl = True
            rhoa = []
            rhob = []
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                # rhoa and rhob must be real
                rhoa.append(ni.eval_rho(mol, ao, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(mol, ao, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.concatenate(rhoa,axis=-1), np.concatenate(rhob,axis=-1)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc

    def get_rho(self, mol, dm, grids, max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self._to_numint1c()
        return ni.get_rho(mol, dm_a+dm_b, grids, max_memory)

    _gks_mcol_vxc = _gks_mcol_vxc
    _gks_mcol_fxc = _gks_mcol_fxc

    @lib.with_doc(numint.nr_rks.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            n, exc, vmat = self._gks_mcol_vxc(mol, grids, xc_code, dms, relativity,
                                              hermi, max_memory, verbose)
        else:
            nao = dms.shape[-1] // 2
            # ground state density is always real
            dm_a = dms[...,:nao,:nao].real.copy('C')
            dm_b = dms[...,nao:,nao:].real.copy('C')
            dm1 = (dm_a, dm_b)
            ni = self._to_numint1c()
            n, exc, v = ni.nr_uks(mol, grids, xc_code, dm1, relativity,
                                  hermi, max_memory, verbose)
            vmat = np.zeros_like(dms)
            vmat[...,:nao,:nao] = v[0]
            vmat[...,nao:,nao:] = v[1]
        return n.sum(), exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    @lib.with_doc(numint.nr_nlc_vxc.__doc__)
    def nr_nlc_vxc(self, mol, grids, xc_code, dm, spin=0, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
        assert dm.ndim == 2
        nao = dm.shape[-1] // 2
        # ground state density is always real
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self._to_numint1c()
        n, exc, v = ni.nr_nlc_vxc(mol, grids, xc_code, dm_a+dm_b, relativity,
                                  hermi, max_memory, verbose)
        vmat = np.zeros_like(dm)
        vmat[:nao,:nao] = v
        vmat[nao:,nao:] = v
        return n, exc, vmat

    @lib.with_doc(numint.nr_rks_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if self.collinear[0] not in ('c', 'm'):  # col or mcol
            raise NotImplementedError('non-collinear fxc')

        if self.collinear[0] == 'm':  # mcol
            fxcmat = self._gks_mcol_fxc(mol, grids, xc_code, dm0, dms,
                                        relativity, hermi, rho0, vxc, fxc,
                                        max_memory, verbose)
        else:
            dms = np.asarray(dms)
            nao = dms.shape[-1] // 2
            if dm0 is not None:
                dm0a = dm0[:nao,:nao].real.copy('C')
                dm0b = dm0[nao:,nao:].real.copy('C')
                dm0 = (dm0a, dm0b)
            # dms_a and dms_b may be complex if they are TDDFT amplitudes
            dms_a = dms[...,:nao,:nao].copy()
            dms_b = dms[...,nao:,nao:].copy()
            dm1 = (dms_a, dms_b)
            ni = self._to_numint1c()
            vmat = ni.nr_uks_fxc(mol, grids, xc_code, dm0, dm1, relativity,
                                 hermi, rho0, vxc, fxc, max_memory, verbose)
            fxcmat = np.zeros_like(dms)
            fxcmat[...,:nao,:nao] = vmat[0]
            fxcmat[...,nao:,nao:] = vmat[1]
        return fxcmat
    get_fxc = nr_gks_fxc = nr_fxc

    eval_xc_eff = _eval_xc_eff
    mcfun_eval_xc_adapter = mcfun_eval_xc_adapter

    block_loop = numint.NumInt.block_loop
    _gen_rho_evaluator = numint.NumInt._gen_rho_evaluator

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(numint.NumInt)
