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
# Author: Yang Gao <younggao1994@gmail.com>

#
'''
Non-relativistic analytical nuclear gradients for unrestricted Hartree Fock with kpoints sampling
'''

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.grad import krhf as rhf_grad
from pyscf.pbc import gto


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    cell = mf_grad.cell
    kpts = mf.kpts
    nkpts = len(kpts)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(cell.natm)

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(cell, kpts)
    s1 = mf_grad.get_ovlp(cell, kpts)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = mf_grad.get_veff(dm0, kpts)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(cell.natm)
    aoslices = cell.aoslice_by_atom()
    de = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xkij,kji->x', h1ao, dm0_sf).real
        de[k] += np.einsum('xskij,skji->x', vhf[:,:,:,p0:p1], dm0[:,:,:,p0:p1]).real * 2
        de[k] -= np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0_sf[:,:,p0:p1]).real * 2
        de[k] /= nkpts
        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose > logger.DEBUG:
        log.debug('gradients of electronic part')
        mf_grad._write(log, cell, de, atmlst)
    return de


def get_veff(mf_grad, dm, kpts):
    '''NR Hartree-Fock Coulomb repulsion'''
    vj, vk = mf_grad.get_jk(dm, kpts)
    vj = vj[:,0] + vj[:,1]
    return vj[:,None] - vk

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    dm1ea = rhf_grad.make_rdm1e(mo_energy[0], mo_coeff[0], mo_occ[0])
    dm1eb = rhf_grad.make_rdm1e(mo_energy[1], mo_coeff[1], mo_occ[1])
    return np.stack((dm1ea,dm1eb), axis=0)

class Gradients(rhf_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''

    def get_veff(self, dm=None, kpts=None):
        if kpts is None: kpts = self.kpts
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, dm, kpts)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec


if __name__=='__main__':
    from pyscf.pbc import scf
    cell = gto.Cell()
    cell.atom = [['He', [0.0, 0.0, 0.0]], ['He', [1, 1.1, 1.2]]]
    cell.basis = 'gth-dzv'
    cell.a = np.eye(3) * 3
    cell.unit='bohr'
    cell.pseudo='gth-pade'
    cell.verbose=4
    cell.build()

    nmp = [1,1,3]
    kpts = cell.make_kpts(nmp)
    kmf = scf.KUHF(cell, kpts, exxdiv=None)
    kmf.kernel()
    mygrad = Gradients(kmf)
    mygrad.kernel()
