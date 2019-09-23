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
from pyscf import lib
from pyscf.pbc.scf import rohf
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks


@lib.with_doc(uks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    if hasattr(dm, 'mo_coeff'):
        mo_coeff = dm.mo_coeff
        mo_occ_a = (dm.mo_occ > 0).astype(np.double)
        mo_occ_b = (dm.mo_occ ==2).astype(np.double)
        dm = lib.tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                           mo_occ=(mo_occ_a,mo_occ_b))
    return uks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpt, kpts_band)


class ROKS(rohf.ROHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''
    def __init__(self, cell, kpt=numpy.zeros(3)):
        rohf.ROHF.__init__(self, cell, kpt)
        rks._dft_common_init_(self)

    def dump_flags(self):
        rohf.ROHF.dump_flags(self)
        lib.logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = pyscf.dft.uks.energy_elec
    define_xc_ = rks.define_xc_

    density_fit = rks._patch_df_beckegrids(rohf.ROHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(rohf.ROHF.mix_density_fit)


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
