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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restricted open-shell Kohn-Sham for periodic systems at a single k-point
'''

import numpy
import pyscf.dft
from pyscf import lib
from pyscf.pbc.scf import rohf
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks
from pyscf import __config__


@lib.with_doc(uks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    if getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = dm.mo_coeff
        mo_occ_a = (dm.mo_occ > 0).astype(numpy.double)
        mo_occ_b = (dm.mo_occ ==2).astype(numpy.double)
        dm = lib.tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                           mo_occ=(mo_occ_a,mo_occ_b))
    return uks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpt, kpts_band)


class ROKS(rks.KohnShamDFT, rohf.ROHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''

    get_vsap = rks.RKS.get_vsap
    init_guess_by_vsap = rks.RKS.init_guess_by_vsap
    get_veff = get_veff
    energy_elec = pyscf.dft.uks.energy_elec

    def __init__(self, cell, kpt=numpy.zeros(3), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        rohf.ROHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        rohf.ROHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert to RHF object.'''
        from pyscf.pbc import scf
        return self._transfer_attrs_(scf.ROHF(self.cell, self.kpt))

    to_gpu = lib.to_gpu


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
