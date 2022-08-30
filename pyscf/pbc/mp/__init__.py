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

from pyscf.pbc import scf
from pyscf.pbc.mp import mp2
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp import kmp2_ksymm
from pyscf.pbc.lib import kpts as libkpts

def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)

MP2 = RMP2

def UMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_uhf(mf)
    return mp2.UMP2(mf, frozen, mo_coeff, mo_occ)

def GMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_ghf(mf)
    return mp2.GMP2(mf, frozen, mo_coeff, mo_occ)

def KRMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf.kpts, libkpts.KPoints):
        return kmp2_ksymm.KRMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return kmp2.KRMP2(mf, frozen, mo_coeff, mo_occ)

KMP2 = KRMP2
