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
PCM family solvent models
'''

import numpy
import scipy
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent

libdft = lib.load_library('libdft')

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def pcm_for_casscf(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def pcm_for_casci(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def pcm_for_post_scf(method, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(method._scf, 'with_solvent', None), PCM):
            solvent_obj = method._scf.with_solvent
        else:
            solvent_obj = PCM(method.mol)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_tdscf.__doc__)
def pcm_for_tdscf(method, solvent_obj=None, dm=None):
    scf_solvent = getattr(method._scf, 'with_solvent', None)
    assert scf_solvent is None or isinstance(scf_solvent, PCM)

    if solvent_obj is None:
        solvent_obj = PCM(method.mol)
    return _attach_solvent._for_tdscf(method, solvent_obj, dm)


# Inject PCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf
mp.mp2.MP2.PCM    = mp.mp2.MP2.PCM    = pcm_for_post_scf
ci.cisd.CISD.PCM  = ci.cisd.CISD.PCM  = pcm_for_post_scf
cc.ccsd.CCSD.PCM  = cc.ccsd.CCSD.PCM  = pcm_for_post_scf
tdscf.rhf.TDBase.PCM = tdscf.rhf.TDBase.PCM = pcm_for_tdscf
mcscf.casci.CASCI.PCM = mcscf.casci.CASCI.PCM = pcm_for_casci
mcscf.mc1step.CASSCF.PCM = mcscf.mc1step.CASSCF.PCM = pcm_for_casscf

# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

Bondi = radii.VDW
Bondi[1] = 1.1/radii.BOHR      # modified version
PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, vdw_scale=1.2):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = numpy.empty((ng,4))
    libdft.MakeAngularGrid(unit_sphere.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ng))

    atom_coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()
    N_J = ng * numpy.ones(mol.natm)
    R_J = numpy.asarray([vdw_scale*Bondi[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg = gto.charge(symb)
        r_vdw = vdw_scale*Bondi[chg]

        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        fiJ = switch_h(diJ)

        w = unit_sphere[:,3] * 4.0 * PI
        swf = numpy.prod(fiJ, axis=1)
        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        xi = XI[ng] / (r_vdw * w[idx]**0.5)
        charge_exp.append(xi)
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords
    }
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A


def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    S = scipy.special.erf(xi_r_ij) / rij
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)

    D = None
    if with_D:
        drij = numpy.expand_dims(grid_coords, axis=1) - grid_coords
        nrij = numpy.sum(drij * norm_vec, axis=-1)

        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S


class PCM(ddcosmo.DDCOSMO):
    def __init__(self, mol):
        ddcosmo.DDCOSMO.__init__(self, mol)
        self.method = 'C-PCM'
        self.vdw_scale = 1.2 # default value in qchem
        self.surface = {}
        self.lebedev_order = 29
        self._intermediates = {}

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'PCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'lmax = %s'         , self.lmax)
        logger.info(self, 'eta = %s'          , self.eta)
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.debug2(self, 'radii_table %s', self.radii_table)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        self.grids.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.surface = None
        super().reset(mol)
        return self

    def build(self, ng=None):
        vdw_scale = self.vdw_scale
        self.radii_table = vdw_scale * Bondi
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(mol, ng=ng, vdw_scale=vdw_scale)
        self._intermediates = {}
        F, A = get_F_A(self.surface)
        D, S = get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        if self.method.upper() == 'C-PCM':
            f_epsilon = (epsilon-1.)/epsilon
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0)
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'IEF-PCM':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

    def _get_vind(self, dms):
        if not self._intermediates or self.grids.coords is None:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = self._get_v(self.surface, dms)
        b = numpy.dot(R, v_grids)
        q = numpy.linalg.solve(K, b)

        vK_1 = numpy.linalg.solve(K.T, v_grids)
        q_sym = (q + numpy.dot(R.T, vK_1))/2.0

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym, v_grids)

        self._intermediates['K'] = K
        self._intermediates['R'] = R
        self._intermediates['q'] = q
        self._intermediates['q_sym'] = q_sym
        self._intermediates['v_grids'] = v_grids

        return epcm, vmat

    def _get_v(self, surface, dms):
        '''
        return electrostatic potential on surface
        '''
        mol = self.mol
        nao = dms.shape[-1]
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()
        grid_coords = surface['grid_coords']
        exponents   = surface['charge_exp']

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        ngrids = grid_coords.shape[0]
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        v_grids_e = numpy.empty(ngrids)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            v_grids_e[p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[0])

        int2c2e = mol._add_suffix('int2c2e')

        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        v_grids_n = numpy.dot(atom_charges, v_ng)

        v_grids = v_grids_n - v_grids_e
        return v_grids

    def _get_vmat(self, q):
        mol = self.mol
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        ngrids = grid_coords.shape[0]
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        vmat = numpy.zeros([nao,nao])
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            vmat += -numpy.einsum('ijL,L->ij', v_nj, q[p0:p1])
        return vmat

