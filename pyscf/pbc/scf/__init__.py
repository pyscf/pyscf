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

'''Hartree-Fock for periodic systems
'''

from pyscf.pbc.scf import hf
rhf = hf
from pyscf.pbc.scf import uhf
from pyscf.pbc.scf import rohf
from pyscf.pbc.scf import ghf
from pyscf.pbc.scf import khf
krhf = khf
from pyscf.pbc.scf import kuhf
from pyscf.pbc.scf import krohf
from pyscf.pbc.scf import kghf
from pyscf.pbc.scf import newton_ah
from pyscf.pbc.scf import addons

RHF = rhf.RHF
UHF = uhf.UHF
ROHF = rohf.ROHF
GHF = ghf.GHF

KRHF = krhf.KRHF
KUHF = kuhf.KUHF
KROHF = krohf.KROHF
KGHF = kghf.KGHF

newton = newton_ah.newton

def HF(cell, *args, **kwargs):
    if cell.spin == 0:
        return RHF(cell, *args, **kwargs)
    else:
        return UHF(cell, *args, **kwargs)

def KHF(cell, *args, **kwargs):
    if cell.spin == 0:
        return KRHF(cell, *args, **kwargs)
    else:
        return KUHF(cell, *args, **kwargs)

