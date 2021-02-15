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
        if getattr(mo[0], 'ndim', None) == 2:  # KRHF
            mo = mo[where]
            mo_occ = mo_occ[where]
        else:  # KUHF
            mo = [mo[0][where], mo[1][where]]
            mo_occ = [mo_occ[0][where], mo_occ[1][where]]
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

    if getattr(mo, 'ndim', None) == 2:
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


def dip_moment(cell, dm, unit='Debye', verbose=logger.NOTE,
               grids=None, rho=None, kpt=np.zeros(3)):
    ''' Dipole moment in the unit cell.

    Args:
         cell : an instance of :class:`Cell`

         dm_kpts (a list of ndarrays) : density matrices of k-points

    Return:
        A list: the dipole moment on x, y and z components
    '''
    dm = dm[0] + dm[1]
    return pbchf.dip_moment(cell, dm, unit, verbose, grids, rho, kpt)

get_rho = pbchf.get_rho


class UHF(pbchf.SCF, mol_uhf.UHF):
    '''UHF class for PBCs.
    '''

    direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', False)

    def __init__(self, cell, kpt=np.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.nelec = None
        self.init_guess_breaksym = None
        self._keys = self._keys.union(["init_guess_breaksym"])

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            cell = self.cell
            ne = cell.nelectron
            nalpha = (ne + cell.spin) // 2
            nbeta = nalpha - cell.spin
            if nalpha + nbeta != ne:
                raise RuntimeError('Electron number %d and spin %d are not consistent\n'
                                   'Note cell.spin = 2S = Nalpha - Nbeta, not 2S+1' %
                                   (ne, cell.spin))
            return nalpha, nbeta
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    def dump_flags(self, verbose=None):
        pbchf.SCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per unit cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    get_rho = get_rho

    eig = mol_uhf.UHF.eig

    get_fock = mol_uhf.UHF.get_fock
    get_grad = mol_uhf.UHF.get_grad
    get_occ = mol_uhf.UHF.get_occ
    make_rdm1 = mol_uhf.UHF.make_rdm1
    energy_elec = mol_uhf.UHF.energy_elec

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
        single_kpt_band = (getattr(kpts_band, 'ndim', None) == 1)
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

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, cell=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        rho = kwargs.pop('rho', None)
        if rho is None:
            rho = self.get_rho(dm)
        return dip_moment(cell, dm, unit, verbose, rho=rho, kpt=self.kpt, **kwargs)

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_uhf.UHF.get_init_guess(self, cell, key)
        dm = pbchf.normalize_dm_(self, dm)
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

    init_guess_by_minao  = mol_uhf.UHF.init_guess_by_minao
    init_guess_by_atom   = mol_uhf.UHF.init_guess_by_atom
    init_guess_by_huckel = mol_uhf.UHF.init_guess_by_huckel

    analyze = mol_uhf.UHF.analyze
    mulliken_pop = mol_uhf.UHF.mulliken_pop
    mulliken_meta = mol_uhf.UHF.mulliken_meta
    spin_square = mol_uhf.UHF.spin_square
    canonicalize = mol_uhf.UHF.canonicalize
    stability = mol_uhf.UHF.stability

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        addons.convert_to_uhf(mf, self)
        return self


