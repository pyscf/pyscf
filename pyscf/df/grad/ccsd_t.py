#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

'''
Density-fitting CCSD(T) analytical nuclear gradients

Copied from pyscf.grad.ccsd_t.py.  Only the assembly of the gradient from the
relaxed density matrices differs; it is taken from pyscf.df.grad.ccsd.
'''

from pyscf import lib
from pyscf.cc import ccsd_t_rdm
from pyscf.df.grad import ccsd as dfccsd_grad

# Only works with canonical orbitals
def grad_elec(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              verbose=lib.logger.INFO):
    mycc = cc_grad.base
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = mycc.ao2mo()
    d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris,
                                          for_grad=True)
    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris,
                                    fd2intermediate, True)
    de = dfccsd_grad.grad_elec(cc_grad, t1, t2, l1, l2, eris, atmlst,
                               d1, d2, verbose)
    return de

class Gradients(dfccsd_grad.Gradients):
    grad_elec = grad_elec

Grad = Gradients
