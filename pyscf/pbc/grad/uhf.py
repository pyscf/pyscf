#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import uhf as mol_uhf
from pyscf.grad.rhf import _write
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.grad import rhf as rhf_grad

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, kpt=np.zeros(3)):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    s1 = mf_grad.get_ovlp(mol, kpt)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0, kpt)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = lib.add(dm0[0], dm0[1])
    dme0_sf = lib.add(dme0[0], dme0[1])

    if atmlst is None:
        atmlst = range(mol.natm)

    de = 0
    if np.sum(kpt) < 1e-9:
        de = mf.with_df.vpploc_part1_nuc_grad(dm0_sf, kpts=kpt.reshape(-1,3))
        de = lib.add(de, pp_int.vpploc_part2_nuc_grad(mol, dm0_sf), out=de)
        de = lib.add(de, pp_int.vppnl_nuc_grad(mol, dm0_sf), out=de)
        h1ao = -mol.pbc_intor('int1e_ipkin', kpt=kpt)
        if mf.with_df.vpplocG_part1 is None or mf.with_df.pp_with_erf:
            h1ao += -mf.with_df.get_vpploc_part1_ip1(kpts=kpt.reshape(-1,3))
        de = lib.add(de, rhf_grad._contract_vhf_dm(mf_grad, h1ao, dm0_sf, atmlst=atmlst) * 2)
        for s in range(2):
            de = lib.add(de, rhf_grad._contract_vhf_dm(mf_grad, vhf[s], dm0[s], atmlst=atmlst) * 2)
        de = lib.add(de, rhf_grad._contract_vhf_dm(mf_grad, s1, dme0_sf, atmlst=atmlst) * -2)
        #TODO extra_force need rewrite
    else:
        raise NotImplementedError

    h1ao = s1 = vhf = dm0 = dme0 = dm0_sf = dme0_sf = None
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
    return de

def get_veff(mf_grad, mol, dm, kpt=np.zeros(3)):
    mf = mf_grad.base
    mydf = mf.with_df
    xc_code = getattr(mf, 'xc', None)
    kpts = kpt.reshape(-1,3)
    return -mydf.get_veff_ip1(dm, xc_code=xc_code, kpts=kpts, spin=1)

class Gradients(rhf_grad.GradientsMixin):
    '''Non-relativistic Gamma-point restricted Hartree-Fock gradients'''
    def get_veff(self, mol=None, dm=None, kpt=np.zeros(3)):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm, kpt)

    make_rdm1e = mol_uhf.Gradients.make_rdm1e
    grad_elec = grad_elec
