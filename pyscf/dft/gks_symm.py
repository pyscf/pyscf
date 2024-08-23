#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
Generalized Kohn-Sham
'''

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ghf_symm
from pyscf.dft import gks
from pyscf.dft import rks
from pyscf.dft.numint2c import NumInt2C


class GKS(rks.KohnShamDFT, ghf_symm.GHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol, xc='LDA,VWN'):
        ghf_symm.GHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = NumInt2C()

    def dump_flags(self, verbose=None):
        ghf_symm.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        logger.info(self, 'collinear = %s', self._numint.collinear)
        if self._numint.collinear[0] == 'm':
            logger.info(self, 'mcfun spin_samples = %s', self._numint.spin_samples)
            logger.info(self, 'mcfun collinear_thrd = %s', self._numint.collinear_thrd)
            logger.info(self, 'mcfun collinear_samples = %s', self._numint.collinear_samples)
        return self

    get_veff = gks.get_veff
    energy_elec = rks.energy_elec

    @property
    def collinear(self):
        return self._numint.collinear
    @collinear.setter
    def collinear(self, val):
        self._numint.collinear = val

    def nuc_grad_method(self):
        raise NotImplementedError

    to_gpu = lib.to_gpu


if __name__ == '__main__':
    import numpy
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = 'H 0 0 0; H 0 0 1; O .5 .6 .2'
    mol.symmetry = True
    mol.basis = 'ccpvdz'
    mol.build()

    mf = GKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    dm = mf.init_guess_by_1e(mol)
    dm = dm + 0j
    nao = mol.nao_nr()
    numpy.random.seed(12)
    dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
    dm[nao:,:nao] = dm[:nao,nao:].T.conj()
    mf.kernel(dm)
    mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    mf.analyze()
    print(mf.spin_square())
    print(mf.e_tot - -76.2760114849027)
