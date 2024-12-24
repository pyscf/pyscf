#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.pbc.dft import krkspu, krks_ksymm
from pyscf.pbc.lib import kpts as libkpts

@lib.with_doc(krkspu.KRKSpU.__doc__)
class KsymAdaptedKRKSpU(krks_ksymm.KRKS):

    get_veff = krkspu.get_veff
    energy_elec = krkspu.energy_elec
    to_hf = lib.invalid_method('to_hf')

    @lib.with_doc(krkspu.KRKSpU.__init__.__doc__)
    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO', **kwargs):
        krkspu.KRKSpU.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv,
                               U_idx=U_idx, U_val=U_val, C_ao_lo=C_ao_lo,
                               minao_ref=minao_ref, **kwargs)

KRKSpU = KsymAdaptedKRKSpU
