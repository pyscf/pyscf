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

'''
Non-relativistic ROKS analytical nuclear gradients
'''

from pyscf import lib
from pyscf.scf import addons
from pyscf.grad import rks as rks_grad
from pyscf.grad import rohf as rohf_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import uks as uks_grad


class Gradients(rks_grad.Gradients):
    '''Non-relativistic ROHF gradients
    '''

    get_veff = uks_grad.get_veff

    make_rdm1e = rohf_grad.make_rdm1e

    grad_elec = uhf_grad.grad_elec

    _tag_rdm1 = rohf_grad._tag_rdm1

Grad = Gradients

from pyscf import dft
dft.roks.ROKS.Gradients = dft.rks_symm.ROKS.Gradients = lib.class_as_method(Gradients)
