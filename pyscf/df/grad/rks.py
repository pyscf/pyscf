#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#

#
# Copyright 2019 Tencent America LLC. All Rights Reserved.
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


from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad
from pyscf.df.grad import rhf as df_rhf_grad


def get_veff(ks_grad, mol=None, dm=None):
    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(
            ni, mol, grids, mf.xc, dm,
            max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(
            ni, mol, grids, mf.xc, dm,
            max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux.sum ((0,1))
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux *= hyb
        vk[:] *= hyb # Don't erase the .aux tags!
        if omega != 0:  # For range separated Coulomb operator
            # TODO: replaced with vk_sr which is numerically more stable for
            # inv(int2c2e)
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vxc += vj - vk * .5
        if ks_grad.auxbasis_response:
            e1_aux = (vj.aux - vk.aux * .5).sum ((0,1))

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
        vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
    else:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc


class Gradients(rks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)

    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True

    get_jk = df_rhf_grad.Gradients.get_jk
    get_j = df_rhf_grad.Gradients.get_j
    get_k = df_rhf_grad.Gradients.get_k
    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        e1 = rks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['vhf'].aux[atom_id]
        return e1

Grad = Gradients
