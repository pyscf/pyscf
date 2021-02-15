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
'''
Non-relativistic analytical nuclear gradients for unrestricted Kohn Sham with kpoints sampling
'''
#

import numpy as np
from pyscf.lib import logger
from pyscf import lib
from pyscf.grad import rks as rks_grad
from pyscf.pbc.grad import kuhf as uhf_grad
from pyscf.pbc.dft import numint
from pyscf.pbc import gto
import time

def get_veff(ks_grad, dm=None, kpts=None):
    mf = ks_grad.base
    cell = ks_grad.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    t0 = (time.clock(), time.time())

    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        raise NotImplementedError
    else:
        vxc =  get_vxc(ni, cell, grids, mf.xc, dm, kpts,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vj = ks_grad.get_j(dm, kpts)
        vxc += vj[:,0][:,None] + vj[:,1][:,None]
    else:
        vj, vk = ks_grad.get_jk(dm, kpts)
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with cell.with_range_coulomb(omega):
                vk += ks_grad.get_k(dm, kpts) * (alpha - hyb)
        vxc += vj[:,0][:,None] + vj[:,1][:,None] - vk

    return vxc

def get_vxc(ni, cell, grids, xc_code, dms, kpts, kpts_band=None, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)
    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()
    nkpts = len(kpts)
    vmat = np.zeros((3,nset,nkpts,nao,nao), dtype=dms.dtype)
    excsum = np.zeros(nset)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory):
            ao_k1 = np.asarray(ao_k1)
            ao_k2 = np.asarray(ao_k2)
            rho_a = make_rho(0, ao_k2[:,0], mask, xctype)
            rho_b = make_rho(1, ao_k2[:,0], mask, xctype)
            vxc = ni.eval_xc(xc_code, (rho_a, rho_b), 1, relativity, 1)[1]
            vrho = vxc[0]
            aowa = np.einsum('xpi,p->xpi', ao_k1[:,0], weight*vrho[:,0])
            aowb = np.einsum('xpi,p->xpi', ao_k1[:,0], weight*vrho[:,1])
            ao_k2 = rho_a = rho_b = vxc = None
            for kn in range(nkpts):
                rks_grad._d1_dot_(vmat[:,0,kn], cell, ao_k1[kn,1:4], aowa[kn], mask, ao_loc, True)
                rks_grad._d1_dot_(vmat[:,1,kn], cell, ao_k1[kn,1:4], aowb[kn], mask, ao_loc, True)
            ao_k1 = aowa = aowb = None

    elif xctype=='GGA':
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory):
            ao_k1 = np.asarray(ao_k1)
            ao_k2 = np.asarray(ao_k2)
            rho_a = make_rho(0, ao_k2[:,:4], mask, xctype)
            rho_b = make_rho(1, ao_k2[:,:4], mask, xctype)
            vxc = ni.eval_xc(xc_code, (rho_a, rho_b), 1, relativity, 1)[1]
            wva, wvb = numint._uks_gga_wv0((rho_a, rho_b), vxc, weight)
            ao_k2 = rho_a = rho_b = vxc = None
            for kn in range(nkpts):
                rks_grad._gga_grad_sum_(vmat[:,0,kn], cell, ao_k1[kn], wva, mask, ao_loc)
                rks_grad._gga_grad_sum_(vmat[:,1,kn], cell, ao_k1[kn], wvb, mask, ao_loc)
            ao_k1 = wva = wvb = None

    elif xctype=='NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError("metaGGA")

    return -vmat

class Gradients(uhf_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    get_veff = get_veff

if __name__=='__main__':
    from pyscf.pbc import dft, gto, scf
    cell = gto.Cell()
    cell.atom = [['He', [0.0, 0.0, 0.0]], ['He', [1, 1.1, 1.2]]]
    cell.basis = 'gth-dzv'
    cell.a = np.eye(3) * 3
    cell.mesh = [19,19,19]
    cell.unit='bohr'
    cell.pseudo='gth-pade'
    cell.verbose=5
    cell.build()

    nmp = [1,1,5]
    kpts = cell.make_kpts(nmp)
    kmf = dft.KUKS(cell, kpts)
    kmf.exxdiv = None
    kmf.xc = 'b3lyp'
    kmf.kernel()
    mygrad = Gradients(kmf)
    mygrad.kernel()
