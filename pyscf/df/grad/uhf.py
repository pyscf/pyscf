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


from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import uhf as uhf_grad
from pyscf.df.grad import rhf as df_rhf_grad


class Gradients(uhf_grad.Gradients):
    '''Unrestricted density-fitting Hartree-Fock gradients'''
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = True
        uhf_grad.Gradients.__init__(self, mf)

    get_jk = df_rhf_grad.get_jk

    def get_j(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_k=False)[0]

    def get_k(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_j=False)[1]

    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj[0]+vj[1] - vk
        if self.auxbasis_response:
            e1_aux = vj.aux - vk.aux
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vhf = lib.tag_array(vhf, aux=e1_aux)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).density_fit()
    mf.conv_tol = 1e-14
    e0 = mf.scf()
    g = Gradients(mf).set(auxbasis_response=False).kernel()
    print(lib.finger(g) - -0.19670644982746546)
    g = Gradients(mf).kernel()
    print(lib.finger(g) - -0.19660674423263175)
# O     0.0000000000    -0.0000000000     0.1236878122
# H    -0.0000000000     0.0970412174    -0.0618439061
# H     0.0000000000    -0.0970412174    -0.0618439061

    mfs = mf.as_scanner()
    e1 = mfs([['O' , (0. , 0.     , 0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    e2 = mfs([['O' , (0. , 0.     ,-0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    print((e1-e2)/0.002*lib.param.BOHR)
