#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.pbc.scf import krohf
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft.kuks import energy_elec
from pyscf import __config__


@lib.with_doc(kuks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = dm.mo_coeff
        mo_occ_a = [(x > 0).astype(np.double) for x in dm.mo_occ]
        mo_occ_b = [(x ==2).astype(np.double) for x in dm.mo_occ]
        dm = lib.tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                           mo_occ=(mo_occ_a,mo_occ_b))
    return kuks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)


class KROKS(rks.KohnShamDFT, krohf.KROHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        krohf.KROHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        krohf.KROHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = kuks.get_rho

    density_fit = rks._patch_df_beckegrids(krohf.KROHF.density_fit)
    rs_density_fit = rks._patch_df_beckegrids(krohf.KROHF.rs_density_fit)
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
