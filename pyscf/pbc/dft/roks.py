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

'''
Restricted open-shell Kohn-Sham for periodic systems at a single k-point
'''

import time
import numpy
import pyscf.dft
from pyscf.pbc.scf import rohf as pbcrohf
from pyscf.lib import logger
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks


get_veff = uks.get_veff

class ROKS(pbcrohf.ROHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''
    def __init__(self, cell, kpt=numpy.zeros(3)):
        pbcrohf.ROHF.__init__(self, cell, kpt)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pbcrohf.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = pyscf.dft.uks.energy_elec
    define_xc_ = rks.define_xc_

    density_fit = rks._patch_df_beckegrids(pbcrohf.ROHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(pbcrohf.ROHF.mix_density_fit)


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
    mf = ROKS(cell)
    print(mf.kernel())
