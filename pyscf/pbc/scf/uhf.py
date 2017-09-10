#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Unrestricted Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import numpy as np
import pyscf.scf.uhf as mol_uhf
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile


def init_guess_by_chkfile(cell, chkfile_name, project=True, kpt=None):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, (nao,nao) ndarray
    '''
    chk_cell, scf_rec = chkfile.load_scf(chkfile_name)
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

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell, chk_kpt-kpt)
        else:
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
    def __init__(self, cell, kpt=np.zeros(3), exxdiv='ewald'):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        n_b = (cell.nelectron - cell.spin) // 2
        self.nelec = (cell.nelectron-n_b, n_b)
        self._keys = self._keys.union(['nelec'])

    def dump_flags(self):
        pbchf.SCF.dump_flags(self)
        logger.info(self, 'number electrons alpha = %d  beta = %d', *self.nelec)
        return self

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
        raise NotImplementedError

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_uhf.UHF.get_init_guess(self, cell, key)
        if cell.dimension < 3:
            if isinstance(dm, np.ndarray) and dm.ndim == 2:
                ne = np.einsum('ij,ji', dm, self.get_ovlp(cell))
            else:
                ne = np.einsum('xij,ji', dm, self.get_ovlp(cell))
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
        return mol_uhf.UHF.init_guess_by_1e(cell)

    def init_guess_by_chkfile(self, chk=None, project=True, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return init_guess_by_chkfile(self.cell, chk, project, kpt)

    dump_chk = pbchf.SCF.dump_chk
    _is_mem_enough = pbchf.SCF._is_mem_enough

    density_fit = pbchf.SCF.density_fit
    # mix_density_fit inherits from hf.RHF.mix_density_fit

    x2c1e = pbchf.SCF.x2c1e

