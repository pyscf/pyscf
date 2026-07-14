#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
CASCI analytical nuclear gradients with restricted KS orbitals

This module implements the RKS-orbital CASCI gradient formulation.  The
orbital response is the CPKS response of the underlying RKS reference, with
the associated XC first-derivative terms in the KS orbital-source constraints.
'''

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf
from pyscf.grad import casci as casci_grad


def _check_supported_orbital_source(mc):
    if isinstance(mc._scf, rohf.ROHF):
        raise NotImplementedError(
            'CASCI gradients with ROKS orbitals are not implemented')
    if not isinstance(mc._scf, hf.KohnShamDFT):
        raise RuntimeError('KS-CASCI gradients require a KohnShamDFT reference')


def grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    _check_supported_orbital_source(mc_grad.base)
    return casci_grad._grad_elec(mc_grad, mo_coeff, ci, atmlst, verbose,
                                 with_ks_response=True)


class Gradients(casci_grad.Gradients):
    '''Non-relativistic RKS-CASCI gradients'''

    grad_elec = grad_elec

    def dump_flags(self, verbose=None):
        casci_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'KS orbital source = %s', self.base._scf.xc)
        return self


Grad = Gradients
