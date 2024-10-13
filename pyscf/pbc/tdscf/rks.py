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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import lib
from pyscf.pbc import dft
from pyscf.tdscf import rks
from pyscf.pbc.tdscf import rhf
from pyscf.pbc.lib.kpts_helper import gamma_point

RPA = TDRKS = TDDFT = rhf.TDHF

class CasidaTDDFT(TDDFT):
    init_guess = rhf.TDA.init_guess
    _gen_vind = rks.TDDFTNoHybrid.gen_vind
    gen_vind = rhf.TDA.gen_vind
    kernel = rks.TDDFTNoHybrid.kernel

TDDFTNoHybrid = CasidaTDDFT

def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    kpt = getattr(mf, 'kpt', None)
    if not mf._numint.libxc.is_hybrid_xc(mf.xc) and gamma_point(kpt):
        return CasidaTDDFT(mf)
    else:
        return TDDFT(mf)

dft.rks.RKS.TDA           = lib.class_as_method(rhf.TDA)
dft.rks.RKS.TDHF          = None
dft.rks.RKS.TDDFTNoHybrid = tddft
dft.rks.RKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.rks.RKS.TDDFT         = tddft
#dft.rks.RKS.dTDA          = lib.class_as_method(dTDA)
#dft.rks.RKS.dRPA          = lib.class_as_method(dRPA)
