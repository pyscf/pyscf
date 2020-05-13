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

UHF = uhf.UHF
ROHF = rohf.ROHF
GHF = ghf.GHF

def RHF(cell, *args, **kwargs):
    if cell.spin == 0:
        return rhf.RHF(cell, *args, **kwargs)
    else:
        return rohf.ROHF(cell, *args, **kwargs)
RHF.__doc__ = rhf.RHF.__doc__

KRHF = krhf.KRHF  # KRHF supports cell.spin != 0 if number of k-points is even
KUHF = kuhf.KUHF
KROHF = krohf.KROHF
KGHF = kghf.KGHF

newton = newton_ah.newton

def HF(cell, *args, **kwargs):
    if cell.spin == 0:
        return rhf.RHF(cell, *args, **kwargs)
    else:
        return uhf.UHF(cell, *args, **kwargs)

def KHF(cell, *args, **kwargs):
    if cell.spin == 0:
        return krhf.KRHF(cell, *args, **kwargs)
    else:
        return kuhf.KUHF(cell, *args, **kwargs)


def KS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.KS(cell, *args, **kwargs)

def KKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.KKS(cell, *args, **kwargs)

def RKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.RKS(cell, *args, **kwargs)

def ROKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.ROKS(cell, *args, **kwargs)

def UKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.UKS(cell, *args, **kwargs)

def KRKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.KRKS(cell, *args, **kwargs)

def KROKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.KROKS(cell, *args, **kwargs)

def KUKS(cell, *args, **kwargs):
    from pyscf.pbc import dft
    return dft.KUKS(cell, *args, **kwargs)

