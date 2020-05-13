#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Non-relativistic unrestricted Hartree-Fock analytical nuclear gradients
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0_sf)
# s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        de[k] += numpy.einsum('sxij,sij->x', vhf[:,:,p0:p1], dm0[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2

        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de

def get_veff(mf_grad, mol, dm):
    '''
    First order derivative of HF potential matrix (wrt electron coordinates)

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    vj, vk = mf_grad.get_jk(mol, dm)
    return vj[0]+vj[1] - vk

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    return numpy.asarray((rhf_grad.make_rdm1e(mo_energy[0], mo_coeff[0], mo_occ[0]),
                          rhf_grad.make_rdm1e(mo_energy[1], mo_coeff[1], mo_occ[1])))


class Gradients(rhf_grad.Gradients):
    '''Non-relativistic unrestricted Hartree-Fock gradients
    '''
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

Grad = Gradients

from pyscf import scf
scf.uhf.UHF.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    mf = scf.UHF(mol)
    mf.scf()
    g = mf.Gradients()
    print(g.grad())

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.grad())
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.grad())
#[[ 0   0                3.27774948e-03]
# [ 0   4.31591309e-02  -1.63887474e-03]
# [ 0  -4.31591309e-02  -1.63887474e-03]]
