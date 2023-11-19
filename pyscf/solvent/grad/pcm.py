#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
Gradient of PCM family solvent models
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.solvent import pcm
from pyscf.solvent.pcm import PI, switch_h
from pyscf.solvent._attach_solvent import _Solvation

libdft = lib.load_library('libdft')

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def get_dF_dA(surface):
    '''
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    '''

    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    R_in_J      = surface['R_in_J']
    R_sw_J      = surface['R_sw_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    dF = numpy.zeros([ngrids, natom, 3])
    dA = numpy.zeros([ngrids, natom, 3])

    for ia in range(atom_coords.shape[0]):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        p1 = p0 + coords.shape[0]
        ri_rJ = numpy.expand_dims(coords, axis=1) - atom_coords
        riJ = numpy.linalg.norm(ri_rJ, axis=-1)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        ri_rJ[:,ia,:] = 0.0
        ri_rJ[diJ < 1e-8] = 0.0

        fiJ = switch_h(diJ)
        dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
        dfiJ = numpy.expand_dims(dfiJ, axis=-1) * ri_rJ

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        # grids response
        Fi = numpy.expand_dims(Fi, axis=-1)
        Ai = numpy.expand_dims(Ai, axis=-1)
        dFi_grid = numpy.sum(dfiJ, axis=1)

        dF[p0:p1,ia,:] += Fi * dFi_grid
        dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = numpy.expand_dims(Fi, axis=-2)
        Ai = numpy.expand_dims(Ai, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        dA[p0:p1,:,:] -= Ai * dfiJ

    return dF, dA

def get_dD_dS(surface, dF, with_S=True, with_D=False):
    '''
    derivative of D and S w.r.t grids, partial_i D_ij = -partial_j D_ij
    S is symmetric, D is not
    '''
    grid_coords = surface['grid_coords']
    exponents   = surface['charge_exp']
    norm_vec    = surface['norm_vec']
    switch_fun  = surface['switch_fun']

    xi_i, xi_j = numpy.meshgrid(exponents, exponents, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    ri_rj = numpy.expand_dims(grid_coords, axis=1) - grid_coords
    rij = numpy.linalg.norm(ri_rj, axis=-1)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)

    dS_dr = -(scipy.special.erf(xi_r_ij) - 2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2))/rij**2
    numpy.fill_diagonal(dS_dr, 0)

    dS_dr= numpy.expand_dims(dS_dr, axis=-1)
    drij = ri_rj/numpy.expand_dims(rij, axis=-1)
    dS = dS_dr * drij

    dD = None
    if with_D:
        nj_rij = numpy.sum(ri_rj * norm_vec, axis=-1)
        dD_dri = 4.0*xi_r_ij**2 * xi_ij / PI**0.5 * numpy.exp(-xi_r_ij**2) * nj_rij / rij**3
        numpy.fill_diagonal(dD_dri, 0.0)

        rij = numpy.expand_dims(rij, axis=-1)
        nj_rij = numpy.expand_dims(nj_rij, axis=-1)
        nj = numpy.expand_dims(norm_vec, axis=0)
        dD_dri = numpy.expand_dims(dD_dri, axis=-1)

        dD = dD_dri * drij + dS_dr * (-nj/rij + 3.0*nj_rij/rij**2 * drij)

    dSii_dF = -exponents * (2.0/PI)**0.5 / switch_fun**2
    dSii = numpy.expand_dims(dSii_dF, axis=(1,2)) * dF

    return dD, dS, dSii

def grad_kernel(pcmobj, dm):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR - dK q)
    '''
    if pcmobj.surface is None:
        pcmobj.build()
    if 'v_grids' not in pcmobj._intermediates:
        pcmobj._get_vind(dm)

    mol = pcmobj.mol
    nao = mol.nao
    aoslice = mol.aoslice_by_atom()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    K            = pcmobj._intermediates['K']
    q            = pcmobj._intermediates['q']
    q_sym        = pcmobj._intermediates['q_sym']

    vK_1 = numpy.linalg.solve(K.T, v_grids)

    # ----------------- potential response -----------------------
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))
    ngrids = grid_coords.shape[0]
    atom_coords = mol.atom_coords(unit='B')

    dvj = numpy.zeros([nao,3])
    dq = numpy.zeros([ngrids,3])
    for p0, p1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
        # charge response
        v_nj_ip1 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1', aosym='s1', comp=3)
        vj = numpy.einsum('xijn,n->xij', v_nj_ip1, q_sym)
        dvj += numpy.einsum('xij,ij->ix', vj, dm)
        dvj += numpy.einsum('xij,ji->ix', vj, dm)

        # electronic potential response
        v_nj_ip2 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip2', aosym='s1', comp=3)
        dq_slice = numpy.einsum('xijn,ij->nx', v_nj_ip2, dm)
        dq[p0:p1] = numpy.einsum('nx,n->nx', dq_slice, q_sym[p0:p1])

    de = numpy.zeros_like(atom_coords)
    de += numpy.asarray([numpy.sum(dq[p0:p1], axis=0) for p0,p1 in gridslice])
    de += numpy.asarray([numpy.sum(dvj[p0:p1], axis=0) for p0,p1 in aoslice[:,2:]])

    atom_charges = mol.atom_charges()
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)

    # nuclei response
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    dv_g = numpy.einsum('g,xng->nx', q_sym, v_ng_ip1)
    de -= numpy.einsum('nx,n->nx', dv_g, atom_charges)

    # nuclei potential response
    int2c2e_ip2 = mol._add_suffix('int2c2e_ip2')
    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip2, fakemol_nuc, fakemol)
    dv_g = numpy.einsum('n,xng->gx', atom_charges, v_ng_ip2)
    dv_g = numpy.einsum('gx,g->gx', dv_g, q_sym)
    de -= numpy.asarray([numpy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])

    ## --------------- response from stiffness matrices ----------------
    gridslice = pcmobj.surface['gslice_by_atom']
    dF, dA = get_dF_dA(pcmobj.surface)

    with_D = pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE'
    dD, dS, dSii = get_dD_dS(pcmobj.surface, dF, with_D=with_D, with_S=True)

    if pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE':
        DA = D*A

    epsilon = pcmobj.eps

    #de_dF = v0 * -dSii_dF * q
    #de += 0.5*numpy.einsum('i,inx->nx', de_dF, dF)
    # dQ = v^T K^-1 (dR - dK K^-1 R) v
    if pcmobj.method.upper() == 'C-PCM' or pcmobj.method.upper() == 'COSMO':
        # dR = 0, dK = dS
        de_dS = numpy.einsum('i,ijx,j->ix', vK_1, dS, q)
        de -= numpy.asarray([numpy.sum(de_dS[p0:p1], axis=0) for p0,p1, in gridslice])
        de -= 0.5*numpy.einsum('i,ijx,i->jx', vK_1, dSii, q)

    elif pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE':
        # IEF-PCM and SS(V)PE formally are the same in gradient calculation
        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * numpy.einsum('i,ijx,j->ix', vK_1, dD, Av)
        de_dR -= 0.5*fac * numpy.einsum('i,ijx,j->jx', vK_1, dD, Av)
        de_dR  = numpy.asarray([numpy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dR += 0.5*fac * numpy.einsum('i,ij,jnx,j->nx', vK_1, D, dA, v_grids)

        de_dS0  = 0.5*numpy.einsum('i,ijx,j->ix', vK_1, dS, q)
        de_dS0 -= 0.5*numpy.einsum('i,ijx,j->jx', vK_1, dS, q)
        de_dS0  = numpy.asarray([numpy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dS0 += 0.5*numpy.einsum('i,inx,i->nx', vK_1, dSii, q)

        vK_1_DA = numpy.dot(vK_1, DA)
        de_dS1  = 0.5*numpy.einsum('j,jkx,k->jx', vK_1_DA, dS, q)
        de_dS1 -= 0.5*numpy.einsum('j,jkx,k->kx', vK_1_DA, dS, q)
        de_dS1  = numpy.asarray([numpy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dS1 += 0.5*numpy.einsum('j,jnx,j->nx', vK_1_DA, dSii, q)

        Sq = numpy.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*numpy.einsum('i,ijx,j->ix', vK_1, dD, ASq)
        de_dD -= 0.5*numpy.einsum('i,ijx,j->jx', vK_1, dD, ASq)
        de_dD  = numpy.asarray([numpy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = numpy.dot(vK_1, D)
        de_dA = 0.5*numpy.einsum('j,jnx,j->nx', vK_1_D, dA, Sq)

        de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
        de += de_dR - de_dK
    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    return de

def make_grad_object(mf, grad_method):
    '''
    return solvent gradient object
    '''
    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')


    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        def kernel(self, *args, dm=None, atmlst=None, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_solvent = grad_kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)

pcm.PCM.nuc_grad_method = make_grad_object
