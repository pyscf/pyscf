#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
Analytical nuclear gradients
============================

Simple usage::

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol).run()
    >>> grad.RHF(mf).kernel()
'''

from pyscf.grad import rhf
from pyscf.grad import dhf
from pyscf.grad import uhf
from pyscf.grad import rohf
from pyscf.grad import ccsd
from pyscf.grad import cisd
#from pyscf.grad import rks
from pyscf.grad.rhf import Gradients as RHF
from pyscf.grad.dhf import Gradients as DHF
from pyscf.grad.uhf import Gradients as UHF
from pyscf.grad.rohf import Gradients as ROHF
#from pyscf.grad.rks import Gradients as RKS
#CCSD = ccsd.Gradients

grad_nuc = rhf.grad_nuc
