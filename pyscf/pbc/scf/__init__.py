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
from pyscf.pbc.scf import khf_ksymm
from pyscf.pbc.scf import kuhf_ksymm
from pyscf.pbc.scf import kghf_ksymm
from pyscf.pbc.scf import newton_ah
from pyscf.pbc.scf import addons
from pyscf.pbc.lib import kpts as libkpts

UHF = uhf.UHF
ROHF = rohf.ROHF
GHF = ghf.GHF

def RHF(cell, *args, **kwargs):
    if cell.spin == 0:
        return rhf.RHF(cell, *args, **kwargs)
    else:
        return rohf.ROHF(cell, *args, **kwargs)
RHF.__doc__ = rhf.RHF.__doc__

#KRHF = krhf.KRHF  # KRHF supports cell.spin != 0 if number of k-points is even
def KRHF(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return khf_ksymm.KRHF(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return khf_ksymm.KRHF(cell, *args, **kwargs)
    return krhf.KRHF(cell, *args, **kwargs)

#KUHF = kuhf.KUHF
def KUHF(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return kuhf_ksymm.KUHF(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return kuhf_ksymm.KUHF(cell, *args, **kwargs)
    return kuhf.KUHF(cell, *args, **kwargs)

KROHF = krohf.KROHF

#KGHF = kghf.KGHF
def KGHF(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return kghf_ksymm.KGHF(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return kghf_ksymm.KGHF(cell, *args, **kwargs)
    return kghf.KGHF(cell, *args, **kwargs)

newton = newton_ah.newton

def HF(cell, *args, **kwargs):
    if cell.spin == 0:
        return rhf.RHF(cell, *args, **kwargs)
    else:
        return uhf.UHF(cell, *args, **kwargs)

def KHF(cell, *args, **kwargs):
    if cell.spin == 0:
        return KRHF(cell, *args, **kwargs)
    else:
        return KUHF(cell, *args, **kwargs)


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

