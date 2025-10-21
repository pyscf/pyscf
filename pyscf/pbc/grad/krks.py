#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
Non-relativistic analytical nuclear gradients for restricted Kohn Sham with kpoints sampling
'''

from pyscf.pbc.grad import krhf as rhf_grad
from pyscf.grad import rks as rks_grad
import numpy as np
from pyscf.dft import numint
from pyscf import lib
from pyscf.lib import logger


def get_veff(ks_grad, dm=None, kpts=None):
    mf = ks_grad.base
    cell = ks_grad.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        raise NotImplementedError
    else:
        vxc = get_vxc(ni, cell, grids, mf.xc, dm, kpts,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc', *t0)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(dm, kpts)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        vj, vk = ks_grad.get_jk(dm, kpts)
        vk *= hyb
        if omega != 0:
            with cell.with_range_coulomb(omega):
                vk += ks_grad.get_k(dm, kpts) * (alpha - hyb)
        vxc += vj - vk * .5

    return vxc

def get_vxc(ni, cell, grids, xc_code, dms, kpts, kpts_band=None, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)
    ao_loc = cell.ao_loc_nr()
    nkpts = len(kpts)
    vmat = np.zeros((3,nset,nkpts,nao,nao), dtype=dms.dtype)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory):
            ao_k1 = np.asarray(ao_k1)
            ao_k2 = np.asarray(ao_k2)
            for i in range(nset):
                rho = make_rho(i, ao_k2[:,0], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
                wv = vxc * weight
                aow = np.einsum('xpi,p->xpi', ao_k1[:,0], wv[0])
                for kn in range(nkpts):
                    rks_grad._d1_dot_(vmat[:,i,kn], cell, ao_k1[kn,1:4], aow[kn], mask, ao_loc, True)
                rho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory):
            ao_k1 = np.asarray(ao_k1)
            ao_k2 = np.asarray(ao_k2)
            for i in range(nset):
                rho = make_rho(i, ao_k2[:,:4], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype,
                                     spin=0)[1]
                wv = vxc * weight
                wv[0] *= .5
                for kn in range(nkpts):
                    rks_grad._gga_grad_sum_(vmat[:,i,kn], cell, ao_k1[kn], wv, mask, ao_loc)
                rho = vxc = wv = None

    elif xctype == 'HF':
        pass

    elif xctype=='NLC':
        raise NotImplementedError("NLC")
    else:
        raise NotImplementedError("metaGGA")

    if nset ==1 :
        return -vmat[:,0]
    else:
        return -vmat

class Gradients(rhf_grad.Gradients):
    _keys = {'grid_response', 'grids'}

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    def reset(self, cell=None):
        if self.grids is not None:
            self.grids.reset(cell)
        return rhf_grad.Gradients.reset(cell)

    def dump_flags(self, verbose=None):
        rhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

    def get_stress(self):
        from pyscf.pbc.grad import krks_stress
        return krks_stress.kernel(self)
