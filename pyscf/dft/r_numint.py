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

import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, BLKSIZE


def eval_ao(mol, coords, deriv=0, with_s=True, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=out)
    if with_s:
        assert(deriv <= 1)  # only GTOval_ipsp_spinor
        ngrid, nao = aoLa.shape[-2:]
        if out is not None:
            aoSa = numpy.empty((comp,nao,ngrid), dtype=numpy.complex128)
            aoSb = numpy.empty((comp,nao,ngrid), dtype=numpy.complex128)
        else:
            out = numpy.ndarray((4,comp,nao,ngrid), dtype=numpy.complex128, buffer=out)
            aoSa, aoSb = out[2:]
        comp = 1
        ao = mol.eval_gto('GTOval_sp_spinor', coords, comp, shls_slice, non0tab)
        aoSa[0] = ao[0].T
        aoSb[0] = ao[1].T
        fevals = ['GTOval_sp_spinor', 'GTOval_ipsp_spinor']
        p1 = 1
        for n in range(1, deriv+1):
            comp = (n+1)*(n+2)//2
            ao = mol.eval_gto(fevals[n], coords, comp, shls_slice, non0tab)
            p0, p1 = p1, p1 + comp
            for k in range(comp):
                aoSa[p0:p1] = ao[0].transpose(0,2,1)
                aoSb[p0:p1] = ao[1].transpose(0,2,1)
        aoSa = aoSa.transpose(0,2,1)
        aoSb = aoSb.transpose(0,2,1)
        if deriv == 0:
            aoSa = aoSa[0]
            aoSb = aoSb[0]
        return aoLa, aoLb, aoSa, aoSb
    else:
        return aoLa, aoLb

def _dm2c_to_rho2x2(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    aoa, aob = ao
    out = _dot_ao_dm(mol, aoa, dm, non0tab, shls_slice, ao_loc, out=out)
    rhoaa = numpy.einsum('pi,pi->p', aoa.real, out.real)
    rhoaa+= numpy.einsum('pi,pi->p', aoa.imag, out.imag)
    rhoba = numpy.einsum('pi,pi->p', aob, out.conj())
    out = _dot_ao_dm(mol, aob, dm, non0tab, shls_slice, ao_loc, out=out)
    rhoab = numpy.einsum('pi,pi->p', aoa, out.conj())
    rhobb = numpy.einsum('pi,pi->p', aob.real, out.real)
    rhobb+= numpy.einsum('pi,pi->p', aob.imag, out.imag)
    return rhoaa, rhoab, rhoba, rhobb

def _rho2x2_to_rho_m(rho2x2):
    raa, rab, rba, rbb = rho2x2
    rho = (raa + rbb).real
    mx = rab.real + rba.real
    my = rba.imag - rab.imag
    mz = raa - rbb
    m = numpy.vstack((mx, my, mz))
    return rho, m

#TODO: \nabla^2 rho and tau = 1/2 (\nabla f)^2
def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    aoa, aob = ao
    ngrids, nao = aoa.shape[-2:]
    xctype = xctype.upper()

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    if xctype == 'LDA':
        tmp = _dm2c_to_rho2x2(mol, ao, dm, non0tab, shls_slice, ao_loc)
        rho, m = _rho2x2_to_rho_m(tmp)
    elif xctype == 'GGA':
        raise NotImplementedError
    else: # meta-GGA
        raise NotImplementedError
    return rho, m

def _vxc2x2_to_mat(mol, ao, weight, rho, vrho, non0tab, shls_slice, ao_loc):
    aoa, aob = ao
    r, m = rho
    vr, vm = vrho.T
    aow = numpy.empty_like(aoa)
#    aow = numpy.einsum('pi,p->pi', aoa, weight*vr, out=aow)
#    mat = _dot_ao_ao(mol, aoa, aow, non0tab, shls_slice, ao_loc)
#    aow = numpy.einsum('pi,p->pi', aob, weight*vr, out=aow)
#    mat+= _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
#
#    s = lib.norm(m, axis=0)
#    ws = vm * weight / (s+1e-300)
#    aow = numpy.einsum('pi,p->pi', aoa, ws*m[0], out=aow)  # Mx
#    tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
#    mat+= tmp + tmp.T.conj()
#    aow = numpy.einsum('pi,p->pi', aoa, ws*m[1], out=aow)  # My
#    tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
#    mat+= (tmp - tmp.T.conj()) * 1j
#    aow = numpy.einsum('pi,p->pi', aoa, ws*m[2], out=aow)  # Mz
#    mat+= _dot_ao_ao(mol, aoa, aow, non0tab, shls_slice, ao_loc)
#    aow = numpy.einsum('pi,p->pi', aob, ws*m[2], out=aow)
#    mat-= _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)

    s = lib.norm(m, axis=0)
    idx = s < 1e-20
    with numpy.errstate(divide='ignore',invalid='ignore'):
        ws = vm * weight / s
    ws[idx] = 0

    aow = numpy.einsum('pi,p->pi', aoa, ws*m[0], out=aow)  # Mx
    tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
    mat = tmp + tmp.T.conj()
    aow = numpy.einsum('pi,p->pi', aoa, ws*m[1], out=aow)  # My
    tmp = _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
    mat+= (tmp - tmp.T.conj()) * 1j
    aow = numpy.einsum('pi,p->pi', aoa, weight*vr, out=aow)
    aow+= numpy.einsum('pi,p->pi', aoa, ws*m[2])  # Mz
    mat+= _dot_ao_ao(mol, aoa, aow, non0tab, shls_slice, ao_loc)
    aow = numpy.einsum('pi,p->pi', aob, weight*vr, out=aow)
    aow-= numpy.einsum('pi,p->pi', aob, ws*m[2])  # Mz
    mat+= _dot_ao_ao(mol, aob, aow, non0tab, shls_slice, ao_loc)
    return mat

def eval_mat(mol, ao, weight, rho, vxc,
             non0tab=None, xctype='LDA', verbose=None):
    aoa, aob = ao
    xctype = xctype.upper()
    ngrids, nao = aoa.shape[-2:]

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    if xctype == 'LDA':
        vrho = vxc[0]
        mat = _vxc2x2_to_mat(mol, ao, weight, rho, vrho, non0tab, shls_slice, ao_loc)
    else:
        raise NotImplementedError
    return mat

def r_vxc(ni, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
          max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    matLL = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    if xctype == 'LDA':
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, 0, with_s, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 1, relativity, 1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den*exc).sum()

                matLL[idm] += _vxc2x2_to_mat(mol, ao[:2], weight, rho, vrho,
                                             mask, shls_slice, ao_loc)
                if with_s:
                    matSS[idm] += _vxc2x2_to_mat(mol, ao[2:], weight, rho, vrho,
                                                 mask, shls_slice, ao_loc)
                rho = m = exc = vxc = vrho = None
    elif xctype == 'GGA':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if with_s:
        c1 = .5 / lib.param.LIGHT_SPEED
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        for idm in range(nset):
            vmat[idm,:n2c,:n2c] = matLL[idm]
            vmat[idm,n2c:,n2c:] = matSS[idm] * c1**2
    else:
        vmat = matLL

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat.reshape(dms.shape)


def get_rho(ni, mol, dm, grids, max_memory=2000):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi=1)
    n2c = mol.nao_2c()
    with_s = (nao == n2c*2)  # 4C DM
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, with_s, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')[0]
    return rho


class RNumInt(numint.NumInt):

    r_vxc = nr_vxc = r_vxc
    get_rho = get_rho

    def eval_ao(self, mol, coords, deriv=0, with_s=True, shls_slice=None,
                non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, with_s, shls_slice, non0tab, out, verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        raise NotImplementedError

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, verbose)

    def block_loop(self, mol, grids, nao, deriv=0, with_s=False, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
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
                              non0tab=non0, out=buf)
            yield ao, non0, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        dms = numpy.asarray(dms)
        nao = dms.shape[-1]
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms.reshape(1,nao,nao)
        ndms = len(dms)
        n2c = mol.nao_2c()
        with_s = (nao == n2c*2)  # 4C DM
        if with_s:
            c1 = .5 / lib.param.LIGHT_SPEED
            dmLL = dms[:,:n2c,:n2c].copy('C')
            dmSS = dms[:,n2c:,n2c:] * c1**2
            def make_rho(idm, ao, non0tab, xctype):
                rho , m  = self.eval_rho(mol, ao[:2], dmLL[idm], non0tab, xctype)
                rhoS, mS = self.eval_rho(mol, ao[2:], dmSS[idm], non0tab, xctype)
                rho += rhoS
                # M = |\beta\Sigma|
                m[0] -= mS[0]
                m[1] -= mS[1]
                m[2] -= mS[2]
                return rho, m
        else:
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao

    def eval_xc(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None: omega = self.omega
        # JTCC, 2, 257
        r, m = rho[:2]
        s = lib.norm(m, axis=0)
        rhou = (r + s) * .5
        rhod = (r - s) * .5
        rho = (rhou, rhod)
        xc = self.libxc.eval_xc(xc_code, rho, 1, relativity, deriv,
                                omega, verbose)
        exc, vxc = xc[:2]
        # update vxc[0] inplace
        vrho = vxc[0]
        vr, vm = (vrho[:,0]+vrho[:,1])*.5, (vrho[:,0]-vrho[:,1])*.5
        vrho[:,0] = vr
        vrho[:,1] = vm
        return xc
_RNumInt = RNumInt


if __name__ == '__main__':
    import time
    from pyscf import gto
    from pyscf.dft import dks

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6311g*',)
    mf = dks.UKS(mol)
    mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),}
    mf.grids.prune = None
    mf.grids.build()
    dm = mf.get_init_guess(key='minao')

    print(time.clock())
    res = mf._numint.r_vxc(mol, mf.grids, mf.xc, dm, spin=0)
    print(res[1] - 0)
    print(time.clock())
