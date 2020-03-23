#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#
# 
# #
# # Copyright 2019 Tencent America LLC. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # Author: Qiming Sun <osirpt.sun@gmail.com>
# #


import time
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import uks as uks_grad
from pyscf.df.grad import rhf as df_rhf_grad


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
        exc, vxc = uks_grad.get_vxc_full_response(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = uks_grad.get_vxc(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk_aux = vk.aux * hyb
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            raise NotImplementedError
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk_aux += vk_lr.aux * (alpha - hyb)
        vxc += vj[0] + vj[1] - vk
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux - vk_aux

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
        vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
    else:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc


class Gradients(uks_grad.Gradients):
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = True
        uks_grad.Gradients.__init__(self, mf)

    get_jk = df_rhf_grad.get_jk

    def get_j(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_k=False)[0]

    def get_k(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_j=False)[1]

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            e1 = uks_grad.Gradients.extra_force(self, atom_id, envs)
            return e1 + envs['vhf'].aux[atom_id]
        else:
            return 0

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
    mf = dft.UKS(mol).density_fit()
    mf.conv_tol = 1e-12
    e0 = mf.scf()
    g = Gradients(mf).set(auxbasis_response=False)
    print(lib.finger(g.kernel()) - -0.12092643506961044)
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.12092884149543644)
# O    -0.0000000000     0.0000000000     0.0533109212
# H    -0.0000000000     0.0675360271    -0.0266615265
# H     0.0000000000    -0.0675360271    -0.0266615265
    g.grid_response = True
# O    -0.0000000000     0.0000000000     0.0533189584
# H    -0.0000000000     0.0675362403    -0.0266594792
# H     0.0000000000    -0.0675362403    -0.0266594792
    print(lib.finger(g.kernel()) - -0.12093220332146028)

    mf.xc = 'b3lypg'
    e0 = mf.kernel()
    g = Gradients(mf)
    print(lib.finger(g.kernel()) - -0.1020433598546214)
# O    -0.0000000000    -0.0000000000     0.0397385108
# H    -0.0000000000     0.0587977564    -0.0198734952
# H     0.0000000000    -0.0587977564    -0.0198734952

