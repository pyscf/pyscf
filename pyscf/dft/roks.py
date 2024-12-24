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
Non-relativistic restricted open-shell Kohn-Sham
'''

import numpy
from pyscf import lib
from pyscf.scf import rohf
from pyscf.dft.uks import energy_elec
from pyscf.dft import rks
from pyscf.dft import uks


@lib.with_doc(uks.get_veff.__doc__)
def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if dm is None:
        dm = ks.make_rdm1()
    elif getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = dm.mo_coeff
        mo_occ_a = (dm.mo_occ > 0).astype(numpy.double)
        mo_occ_b = (dm.mo_occ ==2).astype(numpy.double)
        if dm.ndim == 2:  # RHF DM
            dm = numpy.repeat(dm[None]*.5, 2, axis=0)
        dm = lib.tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                           mo_occ=(mo_occ_a,mo_occ_b))
    elif dm.ndim == 2:  # RHF DM
        dm = numpy.repeat(dm[None]*.5, 2, axis=0)
    return uks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)


class ROKS(rks.KohnShamDFT, rohf.ROHF):
    '''Restricted open-shell Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol, xc='LDA,VWN'):
        rohf.ROHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        rohf.ROHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    get_vsap = rks.get_vsap
    energy_elec = energy_elec

    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import roks
        return roks.Gradients(self)

    def to_hf(self):
        '''Convert to ROHF object.'''
        return self._transfer_attrs_(self.mol.ROHF())

    to_gpu = lib.to_gpu


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

    m = ROKS(mol).run()
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

    m = ROKS(mol)
    m._numint.libxc = xcfun
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405
