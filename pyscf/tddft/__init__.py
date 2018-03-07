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

from pyscf.tddft import rks
from pyscf.tddft import rhf
from pyscf.tddft.rhf import TDHF, CIS, RPA
from pyscf.tddft.rks import dRPA

def TD(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    return TDDFT(mf)

def TDA(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'xc'):
        return rks.TDA(mf)
    else:
        return rhf.TDA(mf)

def TDDFT(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'xc'):
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            return rks.TDDFT(mf)
        else:
            return rks.TDDFTNoHybrid(mf)
    else:
        return rhf.TDHF(mf)
