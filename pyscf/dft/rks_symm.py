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
Non-relativistic Restricted Kohn-Sham
'''

from pyscf.scf import hf_symm
from pyscf.dft import rks
from pyscf.dft import uks


class SymAdaptedRKS(rks.KohnShamDFT, hf_symm.SymAdaptedRHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol, xc='LDA,VWN'):
        hf_symm.RHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf_symm.RHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = rks.get_veff
    get_vsap = rks.get_vsap
    energy_elec = rks.energy_elec

    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import rks
        return rks.Gradients(self)

RKS = SymAdaptedRKS


class SymAdaptedROKS(rks.KohnShamDFT, hf_symm.SymAdaptedROHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol, xc='LDA,VWN'):
        hf_symm.ROHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf_symm.ROHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = uks.get_veff
    get_vsap = rks.get_vsap
    energy_elec = uks.energy_elec

    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import roks
        return roks.Gradients(self)

ROKS = SymAdaptedROKS


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

