#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic generalized Kohn-Sham analytical nuclear gradients
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import ghf as ghf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uks as uks_grad


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of the GKS effective potential matrix (wrt electron
    coordinates)

    Args:
        ks_grad : grad.gks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ni.collinear[0] != 'c':
        raise NotImplementedError(
            'GKS gradients for collinear=%r. Only the collinear ("col") '
            'scheme is implemented; the non-collinear and multi-collinear '
            'schemes need the spin-density XC derivatives on the grid.'
            % (ni.collinear,))

    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    # A collinear functional only sees the diagonal spin blocks, exactly as in
    # UKS. Those blocks are Hermitian, and their antisymmetric imaginary parts
    # integrate to zero against the density, so only the real parts are used.
    nao = dm.shape[-1] // 2
    dm_sb = numpy.asarray([numpy.asarray(dm[:nao,:nao].real, order='C'),
                           numpy.asarray(dm[nao:,nao:].real, order='C')])

    ni1c = ni._to_numint1c()
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9 - mem_now)
    if ks_grad.grid_response:
        exc, vxc = uks_grad.get_vxc_full_response(
            ni1c, mol, grids, mf.xc, dm_sb,
            max_memory=max_memory, verbose=ks_grad.verbose)
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = uks_grad.get_vxc(
            ni1c, mol, grids, mf.xc, dm_sb,
            max_memory=max_memory, verbose=ks_grad.verbose)
    if getattr(mf, 'do_nlc', lambda: False)():
        raise NotImplementedError('GKS gradients with NLC functionals')
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    dm_j, dm_k = ghf_grad._spin_block_dms(dm)
    vj = ks_grad.get_j(mol, dm_j)
    if not ni1c.libxc.is_hybrid_xc(mf.xc):
        vk = None
    else:
        omega, alpha, hyb = ni1c.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vk = ks_grad.get_k(mol, dm_k) * hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm_k, omega=omega) * (alpha - hyb)

    return vj, vk, lib.tag_array(vxc, exc1_grid=exc)


class Gradients(ghf_grad.Gradients):
    '''Non-relativistic generalized Kohn-Sham gradients
    '''

    _keys = {'grid_response', 'grids', 'nlcgrids'}

    def __init__(self, mf):
        ghf_grad.Gradients.__init__(self, mf)
        self.grid_response = False
        self.grids = None
        self.nlcgrids = None

    def dump_flags(self, verbose=None):
        ghf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        The grid response of the numerical XC integration is added here.
        '''
        if self.grid_response:
            vxc = envs['vxc']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vxc.exc1_grid[atom_id])
            return vxc.exc1_grid[atom_id]
        else:
            return 0

Grad = Gradients

from pyscf import dft
dft.gks.GKS.Gradients = dft.gks_symm.GKS.Gradients = lib.class_as_method(Gradients)
