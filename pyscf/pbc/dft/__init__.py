# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''Kohn-Sham DFT for periodic systems
'''

from pyscf.pbc import gto
from pyscf.pbc.dft.gen_grid import UniformGrids, BeckeGrids
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks
from pyscf.pbc.dft import roks
from pyscf.pbc.dft import gks
from pyscf.pbc.dft import krks
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft import kroks
from pyscf.pbc.dft import kgks
from pyscf.pbc.dft import krks_ksymm
from pyscf.pbc.dft import kuks_ksymm
from pyscf.pbc.dft import krkspu
from pyscf.pbc.dft import kukspu
from pyscf.pbc.dft import krkspu_ksymm
from pyscf.pbc.dft import kukspu_ksymm
from pyscf.pbc.dft.rks import KohnShamDFT
from pyscf.pbc.lib import kpts as libkpts

GKS = gks.GKS
UKS = uks.UKS
ROKS = roks.ROKS

def KRKS(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return krks_ksymm.KRKS(cell, *args, **kwargs)
    return krks.KRKS(cell, *args, **kwargs)

def KUKS(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return kuks_ksymm.KUKS(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return kuks_ksymm.KUKS(cell, *args, **kwargs)
    return kuks.KUKS(cell, *args, **kwargs)

KROKS = kroks.KROKS
KGKS = kgks.KGKS

def KRKSpU(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return krkspu_ksymm.KRKSpU(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return krkspu_ksymm.KRKSpU(cell, *args, **kwargs)
    return krkspu.KRKSpU(cell, *args, **kwargs)

def KUKSpU(cell, *args, **kwargs):
    for arg in args:
        if isinstance(arg, libkpts.KPoints):
            return kukspu_ksymm.KUKSpU(cell, *args, **kwargs)
    if 'kpts' in kwargs:
        if isinstance(kwargs['kpts'], libkpts.KPoints):
            return kukspu_ksymm.KUKSpU(cell, *args, **kwargs)
    return kukspu.KUKSpU(cell, *args, **kwargs)

def RKS(cell, *args, **kwargs):
    if cell.spin == 0:
        return rks.RKS(cell, *args, **kwargs)
    else:
        return roks.ROKS(cell, *args, **kwargs)
RKS.__doc__ = rks.RKS.__doc__

def KS(cell, *args, **kwargs):
    if cell.spin == 0:
        return RKS(cell, *args, **kwargs)
    else:
        return UKS(cell, *args, **kwargs)
KS.__doc__ = '''
A wrap function to create DFT object (RKS or UKS) for PBC systems.\n
''' + rks.RKS.__doc__

def KKS(cell, *args, **kwargs):
    if cell.spin == 0:
        return KRKS(cell, *args, **kwargs)
    else:
        return KUKS(cell, *args, **kwargs)
KKS.__doc__ = '''
A wrap function to create DFT object with k-point sampling (KRKS or KUKS).\n
''' + krks.KRKS.__doc__
