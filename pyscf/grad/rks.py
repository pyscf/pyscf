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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Non-relativistic RKS analytical nuclear gradients'''


import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.dft import numint, radi, gen_grid, xc_deriv
from pyscf import __config__
import ctypes

libdft = lib.load_library('libdft')


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = _initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vxc += vj - vk * .5

    return lib.tag_array(vxc, exc1_grid=exc)

def _initialize_grids(ks_grad):
    mf = ks_grad.base
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nlcgrids = None
    if mf.do_nlc():
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(with_non0tab=True)
    return grids, nlcgrids


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc[0]
                aow = numint._scale_ao(ao[0], wv)
                _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:10], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                _tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def get_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((3,nao,nao))
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        vvrho.append(make_rho(0, ao[:4], mask, 'GGA'))
    rho = numpy.hstack(vvrho)

    vxc = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                          grids.coords, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5  # *.5 because vmat + vmat.T at the end
        _gga_grad_sum_(vmat, mol, ao, wv, mask, ao_loc)

    exc = None
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _make_dR_dao_w(ao, wv):
    #:aow = numpy.einsum('npi,p->npi', ao[1:4], wv[0])
    aow = [
        numint._scale_ao(ao[1], wv[0]),  # dX nabla_x
        numint._scale_ao(ao[2], wv[0]),  # dX nabla_y
        numint._scale_ao(ao[3], wv[0]),  # dX nabla_z
    ]
    # XX, XY, XZ = 4, 5, 6
    # YX, YY, YZ = 5, 7, 8
    # ZX, ZY, ZZ = 6, 8, 9
    aow[0] += numint._scale_ao(ao[4], wv[1])  # dX nabla_x
    aow[0] += numint._scale_ao(ao[5], wv[2])  # dX nabla_y
    aow[0] += numint._scale_ao(ao[6], wv[3])  # dX nabla_z
    aow[1] += numint._scale_ao(ao[5], wv[1])  # dY nabla_x
    aow[1] += numint._scale_ao(ao[7], wv[2])  # dY nabla_y
    aow[1] += numint._scale_ao(ao[8], wv[3])  # dY nabla_z
    aow[2] += numint._scale_ao(ao[6], wv[1])  # dZ nabla_x
    aow[2] += numint._scale_ao(ao[8], wv[2])  # dZ nabla_y
    aow[2] += numint._scale_ao(ao[9], wv[3])  # dZ nabla_z
    return aow

def _d1_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:
        vmat[0] += numint._dot_ao_ao(mol, ao1[0], ao2, mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1[1], ao2, mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1[2], ao2, mask, shls_slice, ao_loc)
    else:
        vmat[0] += numint._dot_ao_ao(mol, ao1, ao2[0], mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1, ao2[1], mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1, ao2[2], mask, shls_slice, ao_loc)

def _gga_grad_sum_(vmat, mol, ao, wv, mask, ao_loc):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
    aow = numint._scale_ao(ao[:4], wv[:4])
    _d1_dot_(vmat, mol, ao[1:4], aow, mask, ao_loc, True)
    aow = _make_dR_dao_w(ao, wv[:4])
    _d1_dot_(vmat, mol, aow, ao[0], mask, ao_loc, True)
    return vmat

# XX, XY, XZ = 4, 5, 6
# YX, YY, YZ = 5, 7, 8
# ZX, ZY, ZZ = 6, 8, 9
def _tau_grad_dot_(vmat, mol, ao, wv, mask, ao_loc, dR1_on_bra=True):
    '''The tau part of MGGA functional'''
    aow = numint._scale_ao(ao[1], wv)
    _d1_dot_(vmat, mol, [ao[4], ao[5], ao[6]], aow, mask, ao_loc, True)
    aow = numint._scale_ao(ao[2], wv, aow)
    _d1_dot_(vmat, mol, [ao[5], ao[7], ao[8]], aow, mask, ao_loc, True)
    aow = numint._scale_ao(ao[3], wv, aow)
    _d1_dot_(vmat, mol, [ao[6], ao[8], ao[9]], aow, mask, ao_loc, True)

def _vv10nlc_grad(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    # VV10 gradient term from Vydrov and Van Voorhis 2010 eq. 25-26
    # https://doi.org/10.1063/1.3521275
    thresh=1e-8

    #output
    exc=numpy.zeros((rho[0,:].size,3))

    #outer grid needs threshing
    threshind=rho[0,:]>=thresh
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=thresh
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    #constants and parameters
    Pi=numpy.pi
    Pi43=4.*Pi/3.
    Bvv, Cvv = nlc_pars
    Kvv=Bvv*1.5*Pi*((9.*Pi)**(-1./6.))
    Beta=((3./(Bvv*Bvv))**(0.75))/32.

    #inner grid
    W0p=Gp/(Rp*Rp)
    W0p=Cvv*W0p*W0p
    W0p=(W0p+Pi43*Rp)**0.5
    Kp=Kvv*(Rp**(1./6.))

    #outer grid
    W0tmp=G/(R**2)
    W0tmp=Cvv*W0tmp*W0tmp
    W0=(W0tmp+Pi43*R)**0.5
    K=Kvv*(R**(1./6.))

    vvcoords = numpy.asarray(vvcoords, order='C')
    coords = numpy.asarray(coords, order='C')
    F = numpy.empty(R.shape +(3,), order='C')
    libdft.VXC_vv10nlc_grad(F.ctypes.data_as(ctypes.c_void_p),
                            vvcoords.ctypes.data_as(ctypes.c_void_p),
                            coords.ctypes.data_as(ctypes.c_void_p),
                            W0p.ctypes.data_as(ctypes.c_void_p),
                            W0.ctypes.data_as(ctypes.c_void_p),
                            K.ctypes.data_as(ctypes.c_void_p),
                            Kp.ctypes.data_as(ctypes.c_void_p),
                            RpW.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(vvcoords.shape[0]),
                            ctypes.c_int(coords.shape[0]))
    #exc is multiplied by Rho later
    exc[threshind] = F
    return exc, Beta

def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = numpy.zeros((mol.natm,3))
    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = numpy.empty((3,nao,nao))
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[0], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc[0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv)
            _d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho, weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[:4], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[0] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = wv = None

    elif xctype == 'MGGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho = make_rho(0, ao[:10], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau

            vtmp = numpy.zeros((3,nao,nao))
            _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            _tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = wv = None

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat

def get_nlc_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                              max_memory=2000, verbose=None):
    '''Full NLC functional response including the response of the grids'''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = numpy.zeros((mol.natm,3))
    vmat = numpy.zeros((3,nao,nao))
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2
    vvrho = []
    vvcoords = []
    vvweights = []
    for atm_id, (coords, weight) in enumerate(grids_noresponse_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        vvrho.append(make_rho(0, ao[:4], mask, 'GGA'))
        vvcoords.append(coords)
        vvweights.append(weight)
    vvcoords_flat = numpy.vstack(vvcoords)
    vvweights_flat = numpy.concatenate(vvweights)
    vvrho_flat = numpy.hstack(vvrho)

    vv_vxc = []
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        rho = vvrho[atm_id]
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)

        exc, vxc = numint._vv10nlc(rho, coords, vvrho_flat, vvweights_flat,
                                   vvcoords_flat, nlc_pars)
        vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)
        wv = vv_vxc * weight
        wv[0] *= .5
        vtmp = numpy.zeros((3,nao,nao))
        _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
        vmat += vtmp

        vvrho_sub = numpy.hstack(
            [r for i, r in enumerate(vvrho) if i != atm_id])
        vvcoords_sub = numpy.vstack(
            [r for i, r in enumerate(vvcoords) if i != atm_id])
        vvweights_sub = numpy.concatenate(
            [r for i, r in enumerate(vvweights) if i != atm_id])
        egrad, Beta = _vv10nlc_grad(rho, coords, vvrho_sub,
                                    vvweights_sub, vvcoords_sub, nlc_pars)
        # account for factor of 2 in double integration
        exc -= 0.5 * Beta
        # response of weights
        excsum += 2 * numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
        # response of grids coordinates
        excsum[atm_id] += 2 * numpy.einsum('xij,ji->x', vtmp, dms)
        excsum[atm_id] += numpy.einsum('r,rx->x', rho[0]*weight, egrad)
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


# JCP 98, 5612 (1993); DOI:10.1063/1.464906
def grids_response_cc(grids):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = numpy.linalg.norm(v,axis=0) + 1e-200
            v /= normv
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = numpy.ones((mol.natm,ngrids))
        dpbecke = numpy.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                p0 = fadjust(ia, ib, g)
                p1 = (3 - p0**2) * p0 * .5
                p2 = (3 - p1**2) * p1 * .5
                p3 = (3 - p2**2) * p2 * .5
                t_uab = 27./16 * (1-p2**2) * (1-p1**2) * (1-p0**2) * gadjust(ia, ib, g)

                s_uab = .5 * (1 - p3 + 1e-200)
                s_uba = .5 * (1 + p3 + 1e-200)
                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab / s_uab
                pt_uba = t_uab / s_uba

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]

        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = 1./pbecke.sum(axis=0)
        w1 = dpbecke[:,ia] * z
        w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
        w1 *= vol
        w0 = vol * pbecke[ia] * z
        yield coords, w0, w1


def grids_noresponse_cc(grids):
    # same as above but without the response, for nlc grids response routine
    assert grids.becke_scheme == gen_grid.original_becke
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    coords_all, weights_all = gen_grid.get_partition(mol, atom_grids_tab,
                                                     grids.radii_adjust,
                                                     grids.atomic_radii,
                                                     grids.becke_scheme,
                                                     concat=False)
    natm = mol.natm
    for ia in range(natm):
        yield coords_all[ia], weights_all[ia]


class Gradients(rhf_grad.Gradients):

    # This parameter has no effects for HF gradients. Add this attribute so that
    # the kernel function can be reused in the DFT gradients code.
    grid_response = getattr(__config__, 'grad_rks_Gradients_grid_response', False)

    _keys = {'grid_response', 'grids', 'nlcgrids'}

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    def dump_flags(self, verbose=None):
        rhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        #if callable(self.base.grids.prune):
        #    logger.info(self, 'Grid pruning %s may affect DFT gradients accuracy.'
        #                'Call mf.grids.run(prune=False) to mute grid pruning',
        #                self.base.grids.prune)
        return self

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        if self.grid_response:
            vhf = envs['vhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            return vhf.exc1_grid[atom_id]
        else:
            return 0

Grad = Gradients

from pyscf import dft
dft.rks.RKS.Gradients = dft.rks_symm.RKS.Gradients = lib.class_as_method(Gradients)
dft.roks.ROKS.Gradients = lib.invalid_method('Gradients')
