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
Unrestricted Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf/pbc/scf/khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf as mol_uhf
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile
from pyscf import __config__


def init_guess_by_chkfile(cell, chkfile_name, project=None, kpt=None):
    '''Read the HF results from checkpoint file and make the density matrix
    for UHF initial guess.

    Returns:
        Density matrix, (nao,nao) ndarray
    '''
    from pyscf import gto
    chk_cell, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_cell, cell)

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if kpt is None:
        kpt = np.zeros(3)
    if 'kpt' in scf_rec:
        chk_kpt = scf_rec['kpt']
    elif 'kpts' in scf_rec:
        kpts = scf_rec['kpts'] # the closest kpt from KRHF results
        where = np.argmin(lib.norm(kpts-kpt, axis=1))
        chk_kpt = kpts[where]
        if mo[0].ndim == 2:  # KRHF
            mo = mo[where]
            mo_occ = mo_occ[where]
        else:
            mo = mo[:,where]
            mo_occ = mo_occ[:,where]
    else:  # from molecular code
        chk_kpt = np.zeros(3)

    if project:
        s = cell.pbc_intor('int1e_ovlp', kpt=kpt)
    def fproj(mo):
        if project:
            mo = addons.project_mo_nr2nr(chk_cell, mo, cell, chk_kpt-kpt)
            norm = np.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= np.sqrt(norm)
        return mo

    if mo.ndim == 2:
        mo = fproj(mo)
        mo_occa = (mo_occ>1e-8).astype(np.double)
        mo_occb = mo_occ - mo_occa
        dm = mol_uhf.make_rdm1([mo,mo], [mo_occa,mo_occb])
    else:  # UHF
        dm = mol_uhf.make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)

    # Real DM for gamma point
    if kpt is None or np.allclose(kpt, 0):
        dm = dm.real
    return dm


class UHF(mol_uhf.UHF, pbchf.SCF):
    '''UHF class for PBCs.
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
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (hasattr(kpts_band, 'ndim') and kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_hcore(cell, kpts_band)
        focka, fockb = fock + self.get_veff(cell, dm, kpt=kpt, kpts_band=kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        nkpts = len(kpts_band)
        e_a = []
        e_b = []
        c_a = []
        c_b = []
        for k in range(nkpts):
            e, c = self.eig((focka[k], fockb[k]), s1e[k])
            e_a.append(e[0])
            e_b.append(e[1])
            c_a.append(c[0])
            c_b.append(c[1])
        mo_energy = (e_a, e_b)
        mo_coeff = (c_a, c_b)

        if single_kpt_band:
            mo_energy = (mo_energy[0][0], mo_energy[1][0])
            mo_coeff = (mo_coeff[0][0], mo_coeff[1][0])
        return mo_energy, mo_coeff

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        # skip dipole memont for crystal
        return

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_uhf.UHF.get_init_guess(self, cell, key)
        if cell.dimension < 3:
            if isinstance(dm, np.ndarray) and dm.ndim == 2:
                ne = np.einsum('ij,ji->', dm, self.get_ovlp(cell))
            else:
                ne = np.einsum('xij,ji->', dm, self.get_ovlp(cell))
            if abs(ne - cell.nelectron).sum() > 1e-7:
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
        return mol_uhf.UHF.init_guess_by_1e(self, cell)

    def init_guess_by_chkfile(self, chk=None, project=True, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return init_guess_by_chkfile(self.cell, chk, project, kpt)

    dump_chk = pbchf.SCF.dump_chk
    _is_mem_enough = pbchf.SCF._is_mem_enough

    density_fit = pbchf.SCF.density_fit
    # mix_density_fit inherits from hf.SCF.mix_density_fit

    x2c = x2c1e = sfx2c1e = pbchf.SCF.sfx2c1e

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        addons.convert_to_uhf(mf, self)
        return self

    stability = None
    nuc_grad_method = None

