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
Analytical nuclear gradients
============================

Simple usage::

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol).run()
    >>> grad.RHF(mf).kernel()
'''

from . import rhf
from . import dhf
from . import uhf
from . import rohf
from .rhf import Gradients as RHF
from .dhf import Gradients as DHF
from .uhf import Gradients as UHF
from .rohf import Gradients as ROHF

grad_nuc = rhf.grad_nuc

try:
    from . import casci
    from . import casscf
    from . import sacasscf
    from . import ccsd
    #from . import ccsd_t
    from . import cisd
    from . import mp2
    from . import rks
    from . import roks
    from . import tdrhf
    from . import tdrks
    from . import tduhf
    from . import tduks
    from . import uccsd
    #from . import uccsd_t
    from . import ucisd
    from . import uks
    from . import ump2

    from .rks import Gradients as RKS
    from .uks import Gradients as UKS
    from .roks import Gradients as ROKS
    from . import dispersion

except (ImportError, OSError):
    pass
