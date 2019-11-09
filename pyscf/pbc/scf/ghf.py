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
Generalized Hartree-Fock for periodic systems at a single k-point
'''

import numpy as np
import scipy.linalg
import pyscf.scf.ghf as mol_ghf
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile


def get_jk(mf, cell=None, dm=None, hermi=0, kpt=None, kpts_band=None,
           with_j=True, with_k=True, **kwargs):
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1,nso,nso)
    n_dm = dms.shape[0]

    dmaa = dms[:,:nao,:nao]
    dmab = dms[:,nao:,:nao]
    dmbb = dms[:,nao:,nao:]
    dms = np.vstack((dmaa, dmbb, dmab))

    j1, k1 = mf.with_df.get_jk(dms, hermi, kpt, kpts_band, with_j, with_k,
                               exxdiv=mf.exxdiv)
    j1 = j1.reshape(3,n_dm,nao,nao)
    k1 = k1.reshape(3,n_dm,nao,nao)

    vj = vk = None
    if with_j:
        vj = np.zeros((n_dm,nso,nso), j1.dtype)
        vj[:,:nao,:nao] = vj[:,nao:,nao:] = j1[0] + j1[1]
        vj = vj.reshape(dm.shape)

    if with_k:
        vk = np.zeros((n_dm,nso,nso), k1.dtype)
        vk[:,:nao,:nao] = k1[0]
        vk[:,nao:,nao:] = k1[1]
        vk[:,:nao,nao:] = k1[2]
        vk[:,nao:,:nao] = k1[2].transpose(0,2,1).conj()
        vk = vk.reshape(dm.shape)

    return vj, vk


class GHF(pbchf.SCF, mol_ghf.GHF):
    '''GHF class for PBCs.
    '''

    def get_hcore(self, cell=None, kpt=None):
        hcore = pbchf.SCF.get_hcore(self, cell, kpt)
        return scipy.linalg.block_diag(hcore, hcore)

    def get_ovlp(self, cell=None, kpt=None):
        s = pbchf.SCF.get_ovlp(self, cell, kpt)
        return scipy.linalg.block_diag(s, s)

    get_jk = get_jk
    get_occ = mol_ghf.get_occ
    get_grad = mol_ghf.GHF.get_grad

    def get_j(mf, cell=None, dm=None, hermi=0, kpt=None, kpts_band=None,
              **kwargs):
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, True, False)[0]

    def get_k(self, cell=None, dm=None, hermi=0, kpt=None, kpts_band=None,
              **kwargs):
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, False, True)[1]

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band, True, True)
        vhf = vj - vk
        return vhf

    def get_bands(self, kpts_band, cell=None, dm=None, kpt=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        raise NotImplementedError

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_ghf.GHF.get_init_guess(self, cell, key)
        dm = pbchf.normalize_dm_(self, dm)
        return dm

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        addons.convert_to_ghf(mf, self)
        return self

    stability = None
    nuc_grad_method = None


if __name__ == '__main__':
    from pyscf.scf import addons
    from pyscf.pbc import gto
    from pyscf.pbc import scf

    cell = gto.Cell()
    cell.atom = '''
    H 0 0 0
    H 1 0 0
    H 0 1 0
    H 0 1 1
    '''
    cell.a = np.eye(3)*2
    cell.basis = [[0, [1.2, 1]]]
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([2,2,2])
    mf = scf.RHF(cell, kpt=kpts[7]).run()
    mf = GHF(cell, kpt=kpts[7])
    mf.kernel()
