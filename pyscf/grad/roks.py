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

'''
Non-relativistic ROKS analytical nuclear gradients
'''

from pyscf import lib
from pyscf.scf import addons
from pyscf.grad import rks as rks_grad
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import uks as uks_grad


class Gradients(rks_grad.Gradients):
    '''Non-relativistic ROHF gradients
    '''

    get_veff = uks_grad.get_veff

    make_rdm1e = rohf_grad.make_rdm1e

    grad_elec = uhf_grad.grad_elec

Grad = Gradients

from pyscf import dft
dft.roks.ROKS.Gradients = dft.rks_symm.ROKS.Gradients = lib.class_as_method(Gradients)

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
    mf = dft.ROKS(mol)
    mf.conv_tol = 1e-12
    #mf.grids.atom_grid = (20,86)
    e0 = mf.scf()
    g = mf.Gradients()
    print(g.kernel())
#[[  0.    0.               0.0529837158]
# [  0.    0.0673568416    -0.0264979200]
# [  0.   -0.0673568416    -0.0264979200]]
    g.grid_response = True
    print(g.kernel())
#[[  0.    0.               0.0529917556]
# [  0.    0.0673570505    -0.0264958778]
# [  0.   -0.0673570505    -0.0264958778]]

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[  0.    0.               0.0516999634]
# [  0.    0.0638666270    -0.0258541362]
# [  0.   -0.0638666270    -0.0258541362]]
    g.grid_response = True
    print(g.kernel())
#[[  0.    0.               0.0516940546]
# [  0.    0.0638566430    -0.0258470273]
# [  0.   -0.0638566430    -0.0258470273]]

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[  0.    0.               0.0395990911]
# [  0.    0.0586841789    -0.0198038250]
# [  0.   -0.0586841789    -0.0198038250]]

    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.charge = -1
    mol.spin = 1
    mol.build()

    mf = dft.ROKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0   -0.1479101538]
# [ 0  0    0.1479140846]]
    mf = dft.ROKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
#[[ 0  0   -0.1479101105]
# [ 0  0    0.1479099093]]
