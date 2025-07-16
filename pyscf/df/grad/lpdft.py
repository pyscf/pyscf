#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.grad import lpdft as lpdft_grad
from pyscf.df.grad import sacasscf as dfsacasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from functools import partial

# I need to resolve the __init__ and get_ham_response members. Otherwise everything should be fine!
class Gradients (dfsacasscf_grad.Gradients, lpdft_grad.Gradients):

    def __init__(self, pdft, state=None):
        self.auxbasis_response = True
        lpdft_grad.Gradients.__init__(self, pdft, state=state)

    # TODO: rewrite the partialized fn to take the actual caller, use getattr,
    # and delete this
    def get_ham_response (self, **kwargs):
        pfn = partial (lpdft_grad.lpdft_HellmanFeynman_grad,
         auxbasis_response=self.auxbasis_response)
        with lib.temporary_env (lpdft_grad, lpdft_HellmanFeynman_grad=pfn):
            return lpdft_grad.Gradients.get_ham_response (self, **kwargs)

    kernel = lpdft_grad.Gradients.kernel
    get_wfn_response = lpdft_grad.Gradients.get_wfn_response
    get_init_guess = lpdft_grad.Gradients.get_init_guess
    get_otp_gradient_response = lpdft_grad.Gradients.get_otp_gradient_response
    get_Aop_Adiag = lpdft_grad.Gradients.get_Aop_Adiag

