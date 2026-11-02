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

from pyscf import lib
from pyscf.cc import ccsd_t_rdm
from pyscf.cc import ccsd_t_lambda
from pyscf.grad import ccsd as ccsd_grad

# Only works with canonical orbitals
def grad_elec(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              verbose=lib.logger.INFO):
    mycc = cc_grad.base
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if eris is None: eris = mycc.ao2mo()
    if l1 is None or l2 is None:
        # mycc.l1, mycc.l2 hold the CCSD lambda (from mycc.solve_lambda); the
        # CCSD(T) gradient needs the multipliers of the CCSD(T) Lagrangian.
        l1, l2 = _solve_t_lambda(mycc, eris, t1, t2, verbose)
    d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris,
                                          for_grad=True)
    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris,
                                    fd2intermediate, True)
    # Call the CCSD module function on this object (not on a fresh
    # ccsd_grad.Gradients), so that attributes it sets, such as the relaxed
    # density rdm1_relaxed, land on the gradient object the caller holds.
    de = ccsd_grad.grad_elec(cc_grad, t1, t2, l1, l2, eris, atmlst,
                             d1, d2, verbose)
    return de

def _solve_t_lambda(mycc, eris, t1, t2, verbose=None):
    log = lib.logger.new_logger(mycc, verbose)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2, verbose=log)
    if not conv:
        log.warn('CCSD(T) lambda equations not converged')
    return l1, l2

class Gradients(ccsd_grad.Gradients):
    grad_elec = grad_elec

    def kernel(self, t1=None, t2=None, l1=None, l2=None, eris=None,
               atmlst=None, verbose=None):
        '''CCSD(T) nuclear gradients.

        l1 and l2 are the multipliers of the CCSD(T) Lagrangian. When either is
        not given, they are solved here with ccsd_t_lambda. The CCSD lambda
        (mycc.l1, mycc.l2, mycc.solve_lambda) does not give the derivative of
        the CCSD(T) energy.
        '''
        mycc = self.base
        if l1 is None or l2 is None:
            if t1 is None:
                if mycc.t1 is None:
                    mycc.run()
                t1 = mycc.t1
            if t2 is None: t2 = mycc.t2
            if eris is None:
                eris = mycc.ao2mo()
            l1, l2 = _solve_t_lambda(mycc, eris, t1, t2, verbose)
        return ccsd_grad.Gradients.kernel(self, t1, t2, l1, l2, eris,
                                           atmlst, verbose)
