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
from pyscf.grad import uks as uks_grad


class Gradients(uks_grad.Gradients):
    '''Non-relativistic ROHF gradients
    '''
    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, addons.convert_to_uhf(mf))

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
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
    g.grid_response = True
    print(g.kernel())

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
    g.grid_response = True
    print(g.kernel())

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
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

    mf = dft.ROKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
    mf = dft.ROKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]

