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
Restricted open-shell Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf/pbc/scf/khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rohf as mol_rohf
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import uhf as pbcuhf
from pyscf import __config__


get_fock = mol_rohf.get_fock
get_occ = mol_rohf.get_occ
get_grad = mol_rohf.get_grad
make_rdm1 = mol_rohf.make_rdm1
energy_elec = mol_rohf.energy_elec

class ROHF(mol_rohf.ROHF, pbchf.RHF):
    '''ROHF class for PBCs.
    '''

    direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', False)

    def __init__(self, cell, kpt=np.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.nelec = None
        self._keys = self._keys.union(['nelec'])

    def dump_flags(self):
        pbchf.SCF.dump_flags(self)
        if self.nelec is None:
            nelec = self.cell.nelec
        else:
            nelec = self.nelec
        logger.info(self, 'number of electrons per unit cell  '
                    'alpha = %d beta = %d', *nelec)
        return self

    build = pbchf.SCF.build
    check_sanity = pbchf.SCF.check_sanity
    get_hcore = pbchf.SCF.get_hcore
    get_ovlp = pbchf.SCF.get_ovlp
    get_jk = pbchf.SCF.get_jk
    get_j = pbchf.SCF.get_j
    get_k = pbchf.SCF.get_k
    get_jk_incore = pbchf.SCF.get_jk_incore
    energy_tot = pbchf.SCF.energy_tot

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.asarray((dm*.5,dm*.5))
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = dm.mo_coeff
            mo_occ_a = (dm.mo_occ > 0).astype(np.double)
            mo_occ_b = (dm.mo_occ ==2).astype(np.double)
            dm = lib.tag_array(dm, mo_coeff=(mo_coeff,mo_coeff),
                               mo_occ=(mo_occ_a,mo_occ_b))
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        vhf = vj[0] + vj[1] - vk
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

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        # skip dipole memont for crystal
        return

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_rohf.ROHF.get_init_guess(self, cell, key)
        if cell.dimension < 3:
            if isinstance(dm, np.ndarray) and dm.ndim == 2:
                ne = np.einsum('ij,ji->', dm, self.get_ovlp(cell))
            else:
                ne = np.einsum('xij,ji->', dm, self.get_ovlp(cell))
            if abs(ne - cell.nelectron).max() > 1e-7:
                logger.warn(self, 'Big error detected in the electron number '
                            'of initial guess density matrix (Ne/cell = %g)!\n'
                            '  This can cause huge error in Fock matrix and '
                            'lead to instability in SCF for low-dimensional '
                            'systems.\n  DM is normalized to correct number '
                            'of electrons', ne)
                dm *= cell.nelectron / ne
        return dm

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return mol_uhf.UHF.init_guess_by_1e(cell)

    def init_guess_by_chkfile(self, chk=None, project=True, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return pbcuhf.init_guess_by_chkfile(self.cell, chk, project, kpt)

    dump_chk = pbchf.SCF.dump_chk
    _is_mem_enough = pbchf.SCF._is_mem_enough

    density_fit = pbchf.SCF.density_fit
    # mix_density_fit inherits from hf.SCF.mix_density_fit

    x2c = x2c1e = sfx2c1e = pbchf.SCF.sfx2c1e

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        from pyscf.pbc.scf import addons
        addons.convert_to_rhf(mf, self)
        return self

    stability = None
    nuc_grad_method = None

