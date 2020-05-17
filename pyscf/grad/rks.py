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

import time
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.dft import numint, radi, gen_grid
from pyscf import __config__


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
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
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with mol.with_range_coulomb(omega):
                vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
        vxc += vj - vk * .5

    return lib.tag_array(vxc, exc1_grid=exc)


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, 'LDA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                 verbose=verbose)[1]
                vrho = vxc[0]
                aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
                _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)
                rho = vxc = vrho = aow = None
    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, 'GGA')
                vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                 verbose=verbose)[1]
                wv = numint._rks_gga_wv0(rho, vxc, weight)
                _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                rho = vxc = vrho = wv = None
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    exc = None
    if nset == 1:
        vmat = vmat.reshape(3,nao,nao)
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def _make_dR_dao_w(ao, wv):
    aow = numpy.einsum('npi,p->npi', ao[1:4], wv[0])
    # XX, XY, XZ = 4, 5, 6
    # YX, YY, YZ = 5, 7, 8
    # ZX, ZY, ZZ = 6, 8, 9
    aow[0] += numpy.einsum('pi,p->pi', ao[4], wv[1])  # dX nabla_x
    aow[0] += numpy.einsum('pi,p->pi', ao[5], wv[2])  # dX nabla_y
    aow[0] += numpy.einsum('pi,p->pi', ao[6], wv[3])  # dX nabla_z
    aow[1] += numpy.einsum('pi,p->pi', ao[5], wv[1])  # dY nabla_x
    aow[1] += numpy.einsum('pi,p->pi', ao[7], wv[2])  # dY nabla_y
    aow[1] += numpy.einsum('pi,p->pi', ao[8], wv[3])  # dY nabla_z
    aow[2] += numpy.einsum('pi,p->pi', ao[6], wv[1])  # dZ nabla_x
    aow[2] += numpy.einsum('pi,p->pi', ao[8], wv[2])  # dZ nabla_y
    aow[2] += numpy.einsum('pi,p->pi', ao[9], wv[3])  # dZ nabla_z
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
    aow = numpy.einsum('npi,np->pi', ao[:4], wv)
    _d1_dot_(vmat, mol, ao[1:4], aow, mask, ao_loc, True)
    aow = _make_dR_dao_w(ao, wv)
    _d1_dot_(vmat, mol, aow, ao[0], mask, ao_loc, True)
    return vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    excsum = 0
    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = numpy.empty((3,nao,nao))
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            ngrids = weight.size
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                  verbose=verbose)[:2]
            vrho = vxc[0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
            _d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho, weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho = make_rho(0, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                  verbose=verbose)[:2]

            vtmp = numpy.zeros((3,nao,nao))
            wv = numint._rks_gga_wv0(rho, vxc, weight)
            _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            vmat += vtmp

            # response of weights
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            # response of grids coordinates
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms) * 2
            rho = vxc = vrho = wv = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

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


class Gradients(rhf_grad.Gradients):

# This parameter has no effects for HF gradients. Add this attribute so that
# the kernel function can be reused in the DFT gradients code.
    grid_response = getattr(__config__, 'grad_rks_Gradients_grid_response', False)

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
# This parameter has no effects for HF gradients. Add this attribute so that
# the kernel function can be reused in the DFT gradients code.
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

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


if __name__ == '__main__':
    from pyscf import dft

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. ,  0.757 , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    #mf.grids.atom_grid = (20,86)
    e0 = mf.scf()
    g = mf.Gradients()
    print(lib.finger(g.kernel()) - -0.049887865971659243)
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
    g.grid_response = True
    print(lib.finger(g.kernel()) - -0.049891265876709084)

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.050382923259300716)
#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
    g.grid_response = True
    print(lib.finger(g.kernel()) - -0.05036316927480719)

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.035613964330885352)
#[[ -3.59411142e-16  -2.68753987e-16   1.21557501e-02]
# [  4.04977877e-17   2.11112794e-02  -6.08181640e-03]
# [  1.52600378e-16  -2.11112794e-02  -6.08181640e-03]]


    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(lib.finger(Gradients(mf).kernel()) - 0.0018831588319051444)
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
    mf = dft.RKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(lib.finger(Gradients(mf).kernel()) - 0.0018819497229394144)
#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]
