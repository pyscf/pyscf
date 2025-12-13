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
from pyscf.lib import logger
from pyscf.pbc.dft import kukspu, kuks_ksymm
from pyscf.pbc.lib import kpts as libkpts

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + (Hubbard - double counting) for KUKSpU.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    if isinstance(kpts, np.ndarray):
        return kukspu.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)

    # J + V_xc
    vxc = kuks_ksymm.get_veff(ks, cell, dm, dm_last=dm_last, vhf_last=vhf_last,
                              hermi=hermi, kpts=kpts, kpts_band=kpts_band)
    return kukspu._add_Vhubbard(vxc, ks, dm, kpts)

@lib.with_doc(kukspu.KUKSpU.__doc__)
class KsymAdaptedKUKSpU(kuks_ksymm.KUKS):

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 'minao_ref', 'alpha'}

    get_veff = get_veff
    energy_elec = kukspu.energy_elec
    to_hf = lib.invalid_method('to_hf')

    @lib.with_doc(kukspu.KUKSpU.__init__.__doc__)
    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=[], U_val=[], C_ao_lo=None, minao_ref='MINAO', **kwargs):
        kukspu.KUKSpU.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv,
                               U_idx=U_idx, U_val=U_val, C_ao_lo=C_ao_lo,
                               minao_ref=minao_ref, **kwargs)

KUKSpU = KsymAdaptedKUKSpU
