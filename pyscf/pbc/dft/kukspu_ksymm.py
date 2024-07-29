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
from pyscf.pbc.dft import kukspu, kuks_ksymm
from pyscf.pbc.lib import kpts as libkpts

@lib.with_doc(kukspu.KUKSpU.__doc__)
class KsymAdaptedKUKSpU(kuks_ksymm.KUKS):

    get_veff = kukspu.get_veff
    energy_elec = kukspu.energy_elec
    to_hf = lib.invalid_method('to_hf')

    @lib.with_doc(kukspu.KUKSpU.__init__.__doc__)
    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO'):
        kukspu.KUKSpU.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv,
                               U_idx=U_idx, U_val=U_val, C_ao_lo=C_ao_lo,
                               minao_ref=minao_ref)

KUKSpU = KsymAdaptedKUKSpU

if __name__ == '__main__':
    from pyscf.pbc import gto
    np.set_printoptions(3, linewidth=1000, suppress=True)
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.build()
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, wrap_around=True,
                          space_group_symmetry=True, time_reversal_symmetry=True)
    #U_idx = ["2p", "2s"]
    #U_val = [5.0, 2.0]
    U_idx = ["1 C 2p"]
    U_val = [5.0]

    mf = KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, C_ao_lo='minao',
                minao_ref='gth-szv')
    mf.conv_tol = 1e-10
    print (mf.U_idx)
    print (mf.U_val)
    print (mf.C_ao_lo.shape)
    print (mf.kernel())
