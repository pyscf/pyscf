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
Non-relativistic Unrestricted Kohn-Sham
'''

from pyscf.lib import logger
from pyscf.scf import uhf_symm
from pyscf.dft import uks
from pyscf.dft import rks


class SymAdaptedUKS(rks.KohnShamDFT, uhf_symm.UHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol, xc='LDA,VWN'):
        uhf_symm.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        uhf_symm.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = uks.get_veff
    get_vsap = uks.get_vsap
    energy_elec = uks.energy_elec

    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import uks
        return uks.Gradients(self)

UKS = SymAdaptedUKS


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.symmetry = 1
    mol.build()

    m = UKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

