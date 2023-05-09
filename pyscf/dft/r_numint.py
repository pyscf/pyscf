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
Numerical integration functions for 4-component or 2-component relativistic
methods with j-adapted AO basis
'''

import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao, _tau_dot, BLKSIZE
from pyscf.dft.numint2c import _eval_xc_eff, mcfun_eval_xc_adapter
from pyscf.dft import xc_deriv
from pyscf import __config__


def eval_ao(mol, coords, deriv=0, with_s=True, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None):
    '''Evaluates the value of 2-component or 4-component j-adapted basis on grids.
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    if not with_s:
        # aoLa, aoLb = aoL
        ao = aoL = mol.eval_gto(feval, coords, comp, shls_slice, non0tab,
                                cutoff=cutoff, out=out)
    else:
        assert (deriv <= 1)  # only GTOval_ipsp_spinor
        ngrids = coords.shape[0]
        nao = mol.nao_2c()
        ao = numpy.ndarray((4,comp,nao,ngrids), dtype=numpy.complex128, buffer=out)
        aoL = mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=ao[:2])  # noqa
        ao = ao.transpose(0,1,3,2)
        aoS = ao[2:]
        aoSa, aoSb = aoS
        feval_gto = ['GTOval_sp_spinor', 'GTOval_ipsp_spinor']
        p1 = 0
        for n in range(deriv+1):
            comp = (n+1)*(n+2)//2
            p0, p1 = p1, p1 + comp
            aoSa[p0:p1], aoSb[p0:p1] = mol.eval_gto(
                feval_gto[n], coords, comp, shls_slice, non0tab, cutoff=cutoff)

        if deriv == 0:
            ao = ao[:,0]
    return ao

def _dot_spinor_dm(mol, ket, dm, non0tab, shls_slice, ao_loc):
    ket_a, ket_b = ket
    outa = _dot_ao_dm(mol, ket_a, dm, non0tab, shls_slice, ao_loc)
    outb = _dot_ao_dm(mol, ket_b, dm, non0tab, shls_slice, ao_loc)
    return outa, outb

def _contract_rho_m(bra, ket, hermi=0, bra_eq_ket=False):
    '''
    hermi indicates whether the density matrix is hermitian.
    bra_eq_ket indicates whether bra and ket basis are the same AOs.
    '''
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    ngrids = ket_a.shape[0]
    if hermi:
        raa = numpy.einsum('pi,pi->p', bra_a.real, ket_a.real)
        raa+= numpy.einsum('pi,pi->p', bra_a.imag, ket_a.imag)
        rab = numpy.einsum('pi,pi->p', bra_a.conj(), ket_b)
        rbb = numpy.einsum('pi,pi->p', bra_b.real, ket_b.real)
        rbb+= numpy.einsum('pi,pi->p', bra_b.imag, ket_b.imag)
        rho_m = numpy.empty((4, ngrids))
        rho_m[0,:] = raa + rbb     # rho
        rho_m[1,:] = rab.real      # mx
        rho_m[2,:] = rab.imag      # my
        rho_m[3,:] = raa - rbb     # mz
        if bra_eq_ket:
            rho_m[1,:] *= 2
            rho_m[2,:] *= 2
        else:
            rba = numpy.einsum('pi,pi->p', bra_b.conj(), ket_a)
            rho_m[1,:] += rba.real
            rho_m[2,:] -= rba.imag
    else:
        raa = numpy.einsum('pi,pi->p', bra_a.conj(), ket_a)
        rab = numpy.einsum('pi,pi->p', bra_a.conj(), ket_b)
        rba = numpy.einsum('pi,pi->p', bra_b.conj(), ket_a)
        rbb = numpy.einsum('pi,pi->p', bra_b.conj(), ket_b)
        rho_m = numpy.empty((4, ngrids), dtype=numpy.complex128)
        rho_m[0,:] = raa + rbb         # rho
        rho_m[1,:] = rba + rab         # mx
        rho_m[2,:] = (rba - rab) * 1j  # my
        rho_m[3,:] = raa - rbb         # mz
    return rho_m

def _eval_rho_2c(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, with_lapl=False):
    aoa, aob = ao
    ngrids, nao = aoa.shape[-2:]
    xctype = xctype.upper()

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_spinor_dm(mol, ao, dm, non0tab, shls_slice, ao_loc)
        rho_m = _contract_rho_m(ao, c0, hermi, True)
    elif xctype == 'GGA':
        # first 4 ~ (rho, m), second 4 ~ (0th order, dx, dy, dz)
        if hermi:
            rho_m = numpy.empty((4, 4, ngrids))
        else:
            rho_m = numpy.empty((4, 4, ngrids), dtype=numpy.complex128)
        c0 = _dot_spinor_dm(mol, ao[:,0], dm, non0tab, shls_slice, ao_loc)
        rho_m[:,0] = _contract_rho_m(ao[:,0], c0, hermi, True)
        for i in range(1, 4):
            rho_m[:,i] = _contract_rho_m(ao[:,i], c0, hermi, False)
        if hermi:
            rho_m[:,1:4] *= 2  # *2 for |ao> dm < dx ao| + |dx ao> dm < ao|
        else:
            for i in range(1, 4):
                c1 = _dot_spinor_dm(mol, ao[:,i], dm, non0tab, shls_slice, ao_loc)
                rho_m[:,i] += _contract_rho_m(ao[:,0], c1, hermi, False)
    else: # meta-GGA
        if hermi:
            dtype = numpy.double
        else:
            dtype = numpy.complex128
        if with_lapl:
            rho_m = numpy.empty((4, 6, ngrids), dtype=dtype)
            tau_idx = 5
        else:
            rho_m = numpy.empty((4, 5, ngrids), dtype=dtype)
            tau_idx = 4
        c0 = _dot_spinor_dm(mol, ao[:,0], dm, non0tab, shls_slice, ao_loc)
        rho_m[:,0] = _contract_rho_m(ao[:,0], c0, hermi, True)
        rho_m[:,tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_spinor_dm(mol, ao[:,i], dm, non0tab, shls_slice, ao_loc)
            rho_m[:,tau_idx] += _contract_rho_m(ao[:,i], c1, hermi, True)
            rho_m[:,i] = _contract_rho_m(ao[:,i], c0, hermi, False)
            if hermi:
                rho_m[:,i] *= 2
            else:
                rho_m[:,i] += _contract_rho_m(ao[:,0], c1, hermi, False)
        if with_lapl:
            raise NotImplementedError
        # tau = 1/2 (\nabla f)^2
        rho_m[:,tau_idx] *= .5
    return rho_m

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, with_lapl=False,
             verbose=None):
    r'''Calculate the electron density and magnetization spin density with
    j-adapted spinor basis.
    ''' + numint.eval_rho.__doc__
    with_s = len(ao) == 4  # aoLa, aoLb, aoSa, aoSb
    if with_s:
        n2c = dm.shape[1] // 2
        dmLL = dm[:n2c,:n2c].copy()
        dmSS = dm[n2c:,n2c:].copy()
        c1 = .5 / lib.param.LIGHT_SPEED
        rho  = _eval_rho_2c(mol, ao[:2], dmLL, non0tab, xctype, hermi, with_lapl)
        rhoS = _eval_rho_2c(mol, ao[2:], dmSS, non0tab, xctype, hermi, with_lapl)
        rhoS *= c1**2
        rho[0] += rhoS[0]
        # M = |\beta\Sigma|
        rho[1:4] -= rhoS[1:4]
        return rho
    else:
        return _eval_rho_2c(mol, ao, dm, non0tab, xctype, hermi, with_lapl)

def _ncol_lda_vxc_mat(mol, ao, weight, rho_tm, vxc, mask, shls_slice, ao_loc,
                      hermi, on_LL=True):
    '''Vxc matrix of non-collinear LDA'''
    # NOTE vxc in u/d representation
    aoa, aob = ao
    r, mx, my, mz = rho_tm
    vxc = xc_deriv.ud2ts(vxc)
    vr, vs = vxc[:,0]
    s = lib.norm(rho_tm[1:4], axis=0)

    wv = weight * vr
    with numpy.errstate(divide='ignore',invalid='ignore'):
        ws = vs * weight / s
    ws[s < 1e-20] = 0
    if not on_LL:
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        ws *= -1

    # * .5 because of v+v.conj().T in r_vxc
    if hermi:
        wv *= .5
        ws *= .5

    # einsum('g,g,xgi,xgj->ij', vr, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vs*m[0]/s, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vs*m[1]/s, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vs*m[2]/s, weight, ao, ao)
    aow = None
    aow = _scale_ao(aoa, ws*(mx+my*1j), out=aow)
    mat = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    if hermi:
        mat = mat + mat.conj().T
    else:
        aow = _scale_ao(aob, ws*(mx-my*1j), out=aow)  # Mx
        mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aoa, wv+ws*mz, out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wv-ws*mz, out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    return mat

def _ncol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      hermi, on_LL=True):
    '''Vxc matrix of non-collinear GGA'''
    raise NotImplementedError

def _ncol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                       hermi, on_LL=True):
    '''Vxc matrix of non-collinear MGGA'''
    raise NotImplementedError

def _col_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                     hermi, on_LL=True):
    '''Vxc matrix of collinear LDA'''
    # NOTE vxc in u/d representation
    aoa, aob = ao
    wv = weight * vxc
    if hermi:
        wv *= .5
    if on_LL:
        wva, wvb = wv
    else:  # for SS block
        # v_rho = (vxc_a + vxc_b) * .5
        # v_mz  = (vxc_a - vxc_b) * .5
        # For small components, M = \beta\Sigma leads to
        # (v_rho - sigma_z*v_mz) = [vxc_b, 0    ]
        #                          [0    , vxc_a]
        wvb, wva = wv
    mat  = _dot_ao_ao(mol, aoa, _scale_ao(aoa, wva[0]), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob, _scale_ao(aob, wvb[0]), mask, shls_slice, ao_loc)
    return mat

def _col_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                     hermi, on_LL=True):
    '''Vxc matrix of collinear GGA'''
    # NOTE vxc in u/d representation
    aoa, aob = ao
    wv = weight * vxc
    if hermi:
        wv[:,0] *= .5
    if on_LL:
        wva, wvb = wv
    else:  # for SS block
        wvb, wva = wv
    aow = None
    aow = _scale_ao(aoa[:4], wva[:4], out=aow)
    mat = _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob[:4], wvb[:4], out=aow)
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    if hermi != 1:
        aow = _scale_ao(aoa[1:4], wva[1:4].conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aob[1:4], wvb[1:4].conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)
    return mat

def _col_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      hermi, on_LL=True):
    '''Vxc matrix of collinear MGGA'''
    # NOTE vxc in u/d representation
    aoa, aob = ao
    wv = weight * vxc
    tau_idx = 4
    wv[:,tau_idx] *= .5  # *.5 for 1/2 in tau
    if hermi:
        wv[:,0] *= .5
        wv[:,tau_idx] *= .5
    if on_LL:
        wva, wvb = wv
    else:
        wvb, wva = wv
    aow = None
    aow = _scale_ao(aoa[:4], wva[:4], out=aow)
    mat = _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob[:4], wvb[:4], out=aow)
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    if hermi != 1:
        aow = _scale_ao(aoa[1:4], wva[1:4].conj(), out=aow)
        mat += _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aob[1:4], wvb[1:4].conj(), out=aow)
        mat += _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)

    mat += _tau_dot(mol, aoa, aoa, wva[tau_idx], mask, shls_slice, ao_loc)
    mat += _tau_dot(mol, aob, aob, wvb[tau_idx], mask, shls_slice, ao_loc)
    return mat

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      hermi, on_LL=True):
    '''Vxc matrix of multi-collinear LDA'''
    # NOTE vxc in t/m representation
    aoa, aob = ao
    wv = weight * vxc
    if hermi:
        wv *= .5  # * .5 because of v+v.conj().T in r_vxc
    if not on_LL:
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv[1:4] *= -1
    wr, wmx, wmy, wmz = wv

    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(aoa, wmx[0]+wmy[0]*1j, out=aow)
    mat = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == numpy.double
        mat = mat + mat.conj().T
    else:
        aow = _scale_ao(aob, wmx[0]-wmy[0]*1j, out=aow)
        mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aoa, wr[0]+wmz[0], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[0]-wmz[0], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    return mat

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      hermi, on_LL=True):
    '''Vxc matrix of multi-collinear LDA'''
    # NOTE vxc in t/m representation
    aoa, aob = ao
    wv = weight * vxc
    if hermi:
        wv[:,0] *= .5
    if not on_LL:
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv[1:4] *= -1
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(aoa[:4], wmx[:4]+wmy[:4]*1j, out=aow)
    mat = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == numpy.double
        mat = mat + mat.conj().T
    else:
        aow = _scale_ao(aob[:4], wmx[:4]-wmy[:4]*1j, out=aow)
        mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
        aow = _scale_ao(aob[1:4], (wmx[1:4]+wmy[1:4]*1j).conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aoa[1:4], (wmx[1:4]-wmy[1:4]*1j).conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aoa[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mat+= _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aob[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        mat+= _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)
    aow = _scale_ao(aoa, wr[:4]+wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[:4]-wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    return mat

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                       hermi, on_LL=True):
    '''Vxc matrix of multi-collinear MGGA'''
    # NOTE vxc in t/m representation
    aoa, aob = ao
    wv = weight * vxc
    tau_idx = 4
    wv[:,tau_idx] *= .5  # *.5 for 1/2 in tau
    if hermi:
        wv[:,0] *= .5
        wv[:,tau_idx] *= .5
    if not on_LL:
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv[1:4] *= -1
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(aoa[:4], wmx[:4]+wmy[:4]*1j, out=aow)
    mat = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    mat += _tau_dot(mol, aob, aoa, wmx[tau_idx]+wmy[tau_idx]*1j, mask,
                    shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == numpy.double
        mat = mat + mat.conj().T
    else:
        aow = _scale_ao(aob[:4], wmx[:4]-wmy[:4]*1j, out=aow)
        mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
        mat+= _tau_dot(mol, aoa, aob, wmx[tau_idx]-wmy[tau_idx]*1j, mask,
                       shls_slice, ao_loc)

        aow = _scale_ao(aob[1:4], (wmx[1:4]+wmy[1:4]*1j).conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aoa[1:4], (wmx[1:4]-wmy[1:4]*1j).conj(), out=aow)
        mat+= _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aoa[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mat+= _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(aob[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        mat+= _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)

    aow = _scale_ao(aoa, wr[:4]+wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    mat+= _tau_dot(mol, aoa, aoa, wr[tau_idx]+wmz[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[:4]-wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    mat+= _tau_dot(mol, aob, aob, wr[tau_idx]-wmz[tau_idx], mask, shls_slice, ao_loc)
    return mat

def r_vxc(ni, mol, grids, xc_code, dms, spin=0, relativity=1, hermi=1,
          max_memory=2000, verbose=None):
    '''Calculate 2-component or 4-component Vxc matrix in j-adapted basis.
    ''' + numint.nr_rks.__doc__
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    matLL = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros_like(matLL)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'c'): (_col_lda_vxc_mat  , 0),
            ('GGA' , 'c'): (_col_gga_vxc_mat  , 1),
            ('MGGA', 'c'): (_col_mgga_vxc_mat , 1),
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('GGA' , 'n'): (_ncol_gga_vxc_mat , 1),
            ('MGGA', 'n'): (_ncol_mgga_vxc_mat, 1),
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
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 with_s=with_s):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
                if xctype == 'LDA':
                    den = rho[0] * weight
                else:
                    den = rho[0,0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)

                matLL[i] += fmat(mol, ao[:2], weight, rho, vxc,
                                 mask, shls_slice, ao_loc, hermi, True)
                if with_s:
                    matSS[i] += fmat(mol, ao[2:], weight, rho, vxc,
                                     mask, shls_slice, ao_loc, hermi, False)

        if hermi:
            matLL = matLL + matLL.conj().transpose(0,2,1)
            if with_s:
                matSS = matSS + matSS.conj().transpose(0,2,1)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'r_vxc for functional {xc_code}')

    if with_s:
        matSS *= (.5 / lib.param.LIGHT_SPEED)**2
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        vmat[:,:n2c,:n2c] = matLL
        vmat[:,n2c:,n2c:] = matSS
    else:
        vmat = matLL

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat.reshape(dms.shape)

def _col_rho_tm2ud(rho_tm):
    return numpy.stack([(rho_tm[0] + rho_tm[3]) * .5,
                        (rho_tm[0] - rho_tm[3]) * .5,])

def _col_lda_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                     mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of collinear LDA'''
    rho1 = _col_rho_tm2ud(rho1)
    vxc1 = numpy.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _col_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                            ao_loc, hermi, on_LL)

def _col_gga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                     mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of collinear GGA'''
    rho1 = _col_rho_tm2ud(rho1)
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _col_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                            ao_loc, hermi, on_LL)

def _col_mgga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of collinear MGGA'''
    rho1 = _col_rho_tm2ud(rho1)
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _col_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi, on_LL)

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of multi-collinear LDA'''
    vxc1 = numpy.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _mcol_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi, on_LL)

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of multi-collinear GGA'''
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi, on_LL)

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                       mask, shls_slice, ao_loc, hermi, on_LL=True):
    '''Kernel matrix of multi-collinear MGGA'''
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                              ao_loc, hermi, on_LL)

def r_fxc(ni, mol, grids, xc_code, dm0, dms, spin=0, relativity=1, hermi=0,
          rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Calculate 2-component or 4-component Vxc matrix in j-adapted basis.
    ''' + numint.nr_rks_fxc.__doc__
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]
    if ni.collinear[0] not in ('c', 'm'):  # col or mcol
        raise NotImplementedError('non-collinear fxc')

    make_rho1, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM

    if rho0 is None and (xctype != 'LDA' or fxc is None):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, 1)[0]
    else:
        make_rho0 = None

    matLL = numpy.zeros((nset,n2c,n2c), dtype=dms.dtype)
    matSS = numpy.zeros_like(matLL)
    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'c'): (_col_lda_fxc_mat   , 0),
            ('GGA' , 'c'): (_col_gga_fxc_mat   , 1),
            ('MGGA', 'c'): (_col_mgga_fxc_mat  , 1),
            ('LDA' , 'm'): (_mcol_lda_fxc_mat  , 0),
            ('GGA' , 'm'): (_mcol_gga_fxc_mat  , 1),
            ('MGGA', 'm'): (_mcol_mgga_fxc_mat , 1),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]
        if ni.collinear[0] == 'm':  # mcol
            eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
        else:
            eval_xc = ni.eval_xc_eff

        _rho0 = None
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 with_s=with_s):
            p0, p1 = p1, p1 + weight.size
            if fxc is None:
                if rho0 is not None:
                    if xctype == 'LDA':
                        _rho0 = numpy.asarray(rho0[:,p0:p1], order='C')
                    else:
                        _rho0 = numpy.asarray(rho0[:,:,p0:p1], order='C')
                elif make_rho0 is not None:
                    _rho0 = make_rho0(0, ao, mask, xctype)

                _fxc = eval_xc(xc_code, _rho0, deriv=2, xctype=xctype)[2]
            else:
                _fxc = fxc[:,:,:,:,p0:p1]

            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                matLL[i] += fmat(mol, ao[:2], weight, _rho0, rho1, _fxc,
                                 mask, shls_slice, ao_loc, hermi)
                if with_s:
                    matSS[i] += fmat(mol, ao[2:], weight, _rho0, rho1, _fxc,
                                     mask, shls_slice, ao_loc, hermi, False)

        if hermi:
            # for (\nabla\mu) \nu + \mu (\nabla\nu)
            matLL = matLL + matLL.conj().transpose(0,2,1)
            if with_s:
                matSS = matSS + matSS.conj().transpose(0,2,1)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'r_fxc for functional {xc_code}')

    if with_s:
        matSS *= (.5 / lib.param.LIGHT_SPEED)**2
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        vmat[:,:n2c,:n2c] = matLL
        vmat[:,n2c:,n2c:] = matSS
    else:
        vmat = matLL

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=1,
                    max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype in ('GGA', 'MGGA'):
        ao_deriv = 1
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        ao_deriv = 0

    # Ignore density laplacian for mcfun
    dm = numpy.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi)
    n2c = mol.nao_2c()
    with_s = (nao == n2c*2)  # 4C DM
    rho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                             with_s=with_s):
        rho.append(make_rho(0, ao, mask, xctype))
    rho = numpy.concatenate(rho,axis=-1)

    if ni.collinear[0] == 'm':  # mcol
        eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
    else:
        eval_xc = ni.eval_xc_eff
    vxc, fxc = eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    return rho, vxc, fxc

def get_rho(ni, mol, dm, grids, max_memory=2000):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi=1)
    n2c = mol.nao_2c()
    with_s = (nao == n2c*2)  # 4C DM
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory, with_s=with_s):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')[0]
    return rho


class RNumInt(numint._NumIntMixin):
    '''NumInt for j-adapted (spinor) basis'''

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    spin_samples = getattr(__config__, 'dft_numint_RnumInt_spin_samples', 770)
    collinear_thrd = getattr(__config__, 'dft_numint_RnumInt_collinear_thrd', 0.99)
    collinear_samples = getattr(__config__, 'dft_numint_RnumInt_collinear_samples', 200)

    get_rho = get_rho
    cache_xc_kernel = cache_xc_kernel
    get_vxc = r_vxc = r_vxc
    get_fxc = r_fxc = r_fxc

    eval_xc_eff = _eval_xc_eff
    mcfun_eval_xc_adapter = mcfun_eval_xc_adapter
    eval_ao = staticmethod(eval_ao)
    eval_rho = staticmethod(eval_rho)

    def eval_rho1(self, mol, ao, dm, screen_index=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=None, ao_cutoff=None, pair_mask=None,
                  verbose=None):
        return self.eval_rho(mol, ao, dm, screen_index, xctype, hermi,
                             with_lapl, verbose=verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  with_lapl=True, verbose=None):
        raise NotImplementedError

    def block_loop(self, mol, grids, nao, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None, with_s=False):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.weights.size
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = min(int(max_memory*1e6/((comp*4+4)*nao*16*BLKSIZE))*BLKSIZE, ngrids)
            blksize = max(blksize, BLKSIZE)
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=numpy.uint8)

        if buf is None:
            buf = numpy.empty((4,comp,blksize,nao), dtype=numpy.complex128)
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = self.eval_ao(mol, coords, deriv=deriv, with_s=with_s,
                              non0tab=non0, cutoff=grids.cutoff, out=buf)
            yield ao, non0, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=1, with_lapl=False, grids=None):
        dms = numpy.asarray(dms, order='C')
        if dms.ndim == 2:
            dms = dms[numpy.newaxis]
        ndms, nao = dms.shape[:2]
        def make_rho(idm, ao, non0tab, xctype):
            return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi, with_lapl)
        return make_rho, ndms, nao

_RNumInt = RNumInt
