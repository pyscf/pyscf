#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

'''
X2C 2-component DFT methods
'''

from pyscf.x2c import x2c
from pyscf.dft import dks

class UKS(dks.KohnShamDFT, x2c.UHF):
    def __init__(self, mol, xc='LDA,VWN'):
        x2c.UHF.__init__(self, mol)
        dks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        x2c.UHF.dump_flags(self, verbose)
        dks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert the input mean-field object to an X2C-HF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(x2c.UHF)
        mf.converged = False
        return mf

X2C_UKS = UKS

class RKS(dks.KohnShamDFT, x2c.RHF):
    def __init__(self, mol, xc='LDA,VWN'):
        x2c.RHF.__init__(self, mol)
        dks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        x2c.RHF.dump_flags(self, verbose)
        dks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert the input mean-field object to an X2C-HF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(x2c.RHF)
        mf.converged = False
        return mf

X2C_RKS = RKS
