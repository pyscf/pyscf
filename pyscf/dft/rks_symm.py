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
Non-relativistic Restricted Kohn-Sham
'''

from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import rks
from pyscf.dft import uks


class RKS(pyscf.scf.hf_symm.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.RHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.hf_symm.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = rks.get_veff
    energy_elec = rks.energy_elec


class ROKS(pyscf.scf.hf_symm.ROHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.ROHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.hf_symm.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags()

    get_veff = uks.get_veff
    energy_elec = uks.energy_elec


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 2
    mol.output = '/dev/null'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.symmetry = 1
    mol.build()

    m = RKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

