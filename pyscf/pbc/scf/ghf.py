#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.lib import logger
import pyscf.scf.ghf as mol_ghf
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import addons
from pyscf import __config__


def get_jk(mf, cell=None, dm=None, hermi=0, kpt=None, kpts_band=None,
           with_j=True, with_k=True, **kwargs):
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1,nso,nso)
    if kpts_band is None:  # kpts_band is set to kpt
        nband = 1
        jk_shape = dm.shape
    elif getattr(kpts_band, 'ndim', None) == 1:  # single kpt_band
        nband = 1
        jk_shape = dm.shape
    elif dm.ndim == 2:
        nband = len(kpts_band)
        jk_shape = (nband, nso, nso)
    else:
        nband = len(kpts_band)
        jk_shape = (dm.shape[0], nband, nso, nso)

    dmaa = dms[:,:nao,:nao]
    dmab = dms[:,nao:,:nao]
    dmbb = dms[:,nao:,nao:]
    if with_k:
        if hermi:
            dms = np.stack((dmaa, dmbb, dmab))
        else:
            dmba = dms[:,nao:,:nao]
            dms = np.stack((dmaa, dmbb, dmab, dmba))
        # Note the off-diagonal block breaks the hermitian
        _hermi = 0
    else:
        dms = np.stack((dmaa, dmbb))
        _hermi = 1
    nblocks, n_dm = dms.shape[:2]
    dms = dms.reshape(nblocks*n_dm, nao, nao)

    if mf.rsjk:
        logger.warn(mf, 'RSJK does not support GHF')
        raise NotImplementedError
    j1, k1 = mf.with_df.get_jk(dms, _hermi, kpt, kpts_band, with_j, with_k,
                               exxdiv=mf.exxdiv)

    vj = vk = None
    if with_j:
        j1 = j1.reshape(nblocks,n_dm*nband,nao,nao)
        vj = np.zeros((n_dm*nband,nso,nso), j1.dtype)
        vj[:,:nao,:nao] = vj[:,nao:,nao:] = j1[0] + j1[1]
        vj = vj.reshape(jk_shape)

    if with_k:
        k1 = k1.reshape(nblocks,n_dm*nband,nao,nao)
        vk = np.zeros((n_dm*nband,nso,nso), k1.dtype)
        vk[:,:nao,:nao] = k1[0]
        vk[:,nao:,nao:] = k1[1]
        vk[:,:nao,nao:] = k1[2]
        if hermi:
            vk[:,nao:,:nao] = k1[2].conj().transpose(0,2,1)
        else:
            vk[:,nao:,:nao] = k1[3]
        vk = vk.reshape(jk_shape)

    return vj, vk

class GHF(pbchf.SCF):
    '''GHF class for PBCs at a single point (default: gamma point).
    '''
    _keys = {'with_soc'}

    def __init__(self, cell, kpt=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.with_soc = None

    init_guess_by_chkfile = mol_ghf.GHF.init_guess_by_chkfile
    init_guess_by_minao = mol_ghf.GHF.init_guess_by_minao
    init_guess_by_atom = mol_ghf.GHF.init_guess_by_atom
    init_guess_by_huckel = mol_ghf.GHF.init_guess_by_huckel
    get_jk = get_jk
    get_occ = mol_ghf.get_occ
    get_grad = mol_ghf.GHF.get_grad
    _finalize = mol_ghf.GHF._finalize
    analyze = lib.invalid_method('analyze')
    mulliken_pop = lib.invalid_method('mulliken_pop')
    mulliken_meta = mol_ghf.GHF.mulliken_meta
    spin_square = mol_ghf.GHF.spin_square
    stability = mol_ghf.GHF.stability
    gen_response = NotImplemented

    def get_hcore(self, cell=None, kpt=None):
        hcore = pbchf.SCF.get_hcore(self, cell, kpt)
        hcore = scipy.linalg.block_diag(hcore, hcore)
        if self.with_soc:
            raise NotImplementedError
        return hcore

    def get_ovlp(self, cell=None, kpt=None):
        s = pbchf.SCF.get_ovlp(self, cell, kpt)
        return scipy.linalg.block_diag(s, s)

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
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

    def x2c1e(self):
        '''X2C with spin-orbit coupling effects in spin-orbital basis'''
        from pyscf.pbc.x2c.x2c1e import x2c1e_gscf
        return x2c1e_gscf(self)
    x2c = sfx2c1e = x2c1e

    def to_ks(self, xc='HF'):
        '''Convert to RKS object.
        '''
        from pyscf.pbc import dft
        return self._transfer_attrs_(dft.GKS(self.cell, self.kpt, xc=xc))

    def convert_from_(self, mf):
        '''Convert given mean-field object to GHF'''
        addons.convert_to_ghf(mf, self)
        return self

    to_gpu = lib.to_gpu


if __name__ == '__main__':
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
