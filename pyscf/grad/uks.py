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

'''Non-relativistic UKS analytical nuclear gradients'''

import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.dft import numint, gen_grid
from pyscf import __config__


def get_veff(ks_grad, mol=None, dm=None):
    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (time.clock(), time.time())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    nao = vxc.shape[-1]
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with mol.with_range_coulomb(omega):
                vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
        vxc += vj[0] + vj[1] - vk

    return lib.tag_array(vxc, exc1_grid=exc)


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[1]
            wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)

            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wva, mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wvb, mask, ao_loc)
            rho_a = rho_b = vxc = wva = wvb = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    exc = numpy.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()

    excsum = 0
    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[:2]
            vrho = vxc[0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a+rho_b, weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[:2]
            wva, wvb = numint._uks_gga_wv0((rho_a,rho_b), vxc, weight)

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wva, mask, ao_loc)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wvb, mask, ao_loc)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2
            rho_a = rho_b = vxc = wva = wvb = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(uhf_grad.Gradients):

    grid_response = getattr(__config__, 'grad_uks_Gradients_grid_response', False)

    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    def dump_flags(self):
        uhf_grad.Gradients.dump_flags(self)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. ,  0.757 , 0.587)] ]
    mol.basis = '631g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-12
    #mf.grids.atom_grid = (20,86)
    e0 = mf.scf()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.12090786243525126)
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
    g.grid_response = True
    print(lib.finger(g.kernel()) - -0.12091122429043633)

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.11509739136150157)
#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
    g.grid_response = True
    print(lib.finger(g.kernel()) - -0.11507986316077731)

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.10202554999695367)
#[[ -3.59411142e-16  -2.68753987e-16   1.21557501e-02]
# [  4.04977877e-17   2.11112794e-02  -6.08181640e-03]
# [  1.52600378e-16  -2.11112794e-02  -6.08181640e-03]]


    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.charge = -1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(lib.finger(Gradients(mf).kernel()) - 0.10365160440876001)
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
    mf = dft.UKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(lib.finger(Gradients(mf).kernel()) - 0.10365040148752827)
#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]

