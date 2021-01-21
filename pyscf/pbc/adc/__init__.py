#!/usr/bin/env python
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

from pyscf.pbc import scf
from pyscf.pbc.adc import adc,kadc_rhf 

def RADC(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return adc.RADC(mf, frozen, mo_coeff, mo_occ)

ADC = RADC


def KRADC(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.adc import kadc_rhf
    if not isinstance(mf, scf.khf.KRHF):
        mf = scf.addons.convert_to_rhf(mf)
    return kadc_rhf.RADC(mf, frozen, mo_coeff, mo_occ)

#KCCSD = KRCCSD
