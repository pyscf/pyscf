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
from pyscf import lib
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.grad import mspdft as mspdft_grad
from pyscf.grad import mcpdft as mcpdft_grad
from pyscf.df.grad import casscf as dfcasscf_grad
from pyscf.df.grad import sacasscf as dfsacasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from functools import partial

# I need to resolve the __init__ and get_ham_response members. Otherwise everything should be fine!
class Gradients (mspdft_grad.Gradients):

    def __init__(self, pdft):
        self.auxbasis_response = True
        mspdft_grad.Gradients.__init__(self, pdft)

    def make_fcasscf (self, state=None, casscf_attr={}, fcisolver_attr={}):
        fcasscf = sacasscf_grad.Gradients.make_fcasscf (self, state=state,
            casscf_attr=casscf_attr, fcisolver_attr=fcisolver_attr)
        def fcasscf_grad ():
            return dfcasscf_grad.Gradients (fcasscf)
        fcasscf.nuc_grad_method = fcasscf_grad
        return fcasscf

    # TODO: rewrite the partialized fn to take the actual caller, use getattr,
    # and delete this
    def get_ham_response (self, **kwargs):
        pfn = partial (mcpdft_grad.mcpdft_HellmanFeynman_grad,
         auxbasis_response=self.auxbasis_response)
        with lib.temporary_env (mcpdft_grad, mcpdft_HellmanFeynman_grad=pfn):
            return mspdft_grad.Gradients.get_ham_response (self, **kwargs)

    def get_LdotJnuc (self, Lvec, **kwargs):
        with lib.temporary_env (sacasscf_grad,
         Lci_dot_dgci_dx=dfsacasscf_grad.Lci_dot_dgci_dx,
         Lorb_dot_dgorb_dx=dfsacasscf_grad.Lorb_dot_dgorb_dx):
            return mspdft_grad.Gradients.get_LdotJnuc (self, Lvec, **kwargs)


