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
Non-relativistic restricted open-shell Kohn-Sham
'''

from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf.dft.uks import get_veff, energy_elec
from pyscf.dft import rks


class ROKS(rohf.ROHF):
    '''Restricted open-shell Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol):
        rohf.ROHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        rohf.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = energy_elec
    define_xc_ = rks.define_xc_

    def nuc_grad_method(self):
        from pyscf.grad import roks
        return roks.Gradients(self)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf.dft import xcfun
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = ROKS(mol)
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

    m = ROKS(mol)
    m._numint.libxc = xcfun
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

