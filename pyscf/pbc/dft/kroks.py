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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restricted open-shell Kohn-Sham for periodic systems with k-point sampling
'''

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.scf import krohf
from pyscf.pbc.dft import rks
from pyscf.pbc.dft.kuks import get_veff, energy_elec


class KROKS(krohf.KROHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        krohf.KROHF.__init__(self, cell, kpts)
        rks._dft_common_init_(self)

    def dump_flags(self):
        krohf.KROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff

    energy_elec = energy_elec

    define_xc_ = rks.define_xc_

    density_fit = rks._patch_df_beckegrids(krohf.KROHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(krohf.KROHF.mix_density_fit)


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    mf = KROKS(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
