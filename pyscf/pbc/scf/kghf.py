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
Generalized Hartree-Fock for periodic systems with k-point sampling
'''

from functools import reduce
import numpy as np
import scipy.linalg
import pyscf.scf.hf as mol_hf  # noqa
import pyscf.scf.ghf as mol_ghf  # noqa
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import ghf as pbcghf
from pyscf.pbc.scf import addons
from pyscf.pbc.df.df_jk import _format_jks
from pyscf import __config__


def get_jk(mf, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
           with_j=True, with_k=True, **kwargs):
    if cell is None: cell = mf.cell
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    nkpts = len(kpts)
    if kpts_band is None:
        nband = nkpts
    else:
        nband = len(kpts_band)

    dm_kpts = np.asarray(dm_kpts)
    nso = dm_kpts.shape[-1]
    nao = nso // 2
    dms = dm_kpts.reshape(-1,nkpts,nso,nso)

    dmaa = dms[:,:,:nao,:nao]
    dmab = dms[:,:,nao:,:nao]
    dmbb = dms[:,:,nao:,nao:]
    if with_k:
        if hermi:
            dms = np.stack((dmaa, dmbb, dmab))
        else:
            dmba = dms[:,:,nao:,:nao]
            dms = np.stack((dmaa, dmbb, dmab, dmba))
        _hermi = 0
    else:
        dms = np.stack((dmaa, dmbb))
        _hermi = 1
    nblocks, n_dm = dms.shape[:2]
    dms = dms.reshape(nblocks*n_dm, nkpts, nao, nao)

    j1, k1 = mf.with_df.get_jk(dms, _hermi, kpts, kpts_band, with_j, with_k,
                               exxdiv=mf.exxdiv)

    vj = vk = None
    if with_j:
        # j1 = (j1_aa, j1_bb, j1_ab)
        j1 = j1.reshape(nblocks,n_dm,nband,nao,nao)
        vj = np.zeros((n_dm,nband,nso,nso), j1.dtype)
        vj[:,:,:nao,:nao] = vj[:,:,nao:,nao:] = j1[0] + j1[1]
        vj = _format_jks(vj, dm_kpts, kpts_band, kpts)

    if with_k:
        k1 = k1.reshape(nblocks,n_dm,nband,nao,nao)
        vk = np.zeros((n_dm,nband,nso,nso), k1.dtype)
        vk[:,:,:nao,:nao] = k1[0]
        vk[:,:,nao:,nao:] = k1[1]
        vk[:,:,:nao,nao:] = k1[2]
        if hermi:
            # k1 = (k1_aa, k1_bb, k1_ab)
            vk[:,:,nao:,:nao] = k1[2].conj().transpose(0,1,3,2)
        else:
            # k1 = (k1_aa, k1_bb, k1_ab, k1_ba)
            vk[:,:,nao:,:nao] = k1[3]
        vk = _format_jks(vk, dm_kpts, kpts_band, kpts)

    return vj, vk

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts)
    nocc = mf.cell.nelectron * nkpts

    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double))

    if nocc < mo_energy.size:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[k][mo_occ_kpts[k]> 0],
                         mo_energy_kpts[k][mo_occ_kpts[k]==0])
        np.set_printoptions(threshold=1000)

    return mo_occ_kpts

def _cast_mol_init_guess(fn):
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = mol_ghf._from_rhf_init_dm(fn(cell))
        nkpts = len(kpts)
        dm_kpts = np.asarray([dm] * nkpts)
        return dm_kpts
    fn_init_guess.__name__ = fn.__name__
    fn_init_guess.__doc__ = (
        'Generates initial guess density matrix and the orbitals of the initial '
        'guess DM ' + fn.__doc__)
    return fn_init_guess

class KGHF(pbcghf.GHF, khf.KSCF):
    '''GHF class for PBCs.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)

    def get_hcore(self, cell=None, kpts=None):
        hcore = khf.KSCF.get_hcore(self, cell, kpts)
        return lib.asarray([scipy.linalg.block_diag(h, h) for h in hcore])

    def get_ovlp(self, cell=None, kpts=None):
        s = khf.KSCF.get_ovlp(self, cell, kpts)
        return lib.asarray([scipy.linalg.block_diag(x, x) for x in s])

    get_jk = get_jk
    get_occ = get_occ
    energy_elec = khf.KSCF.energy_elec

    def get_j(self, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, True, False)[0]

    def get_k(self, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, False, True)[1]

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band, True, True)
        vhf = vj - vk
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        def grad(mo, mo_occ, fock):
            occidx = mo_occ > 0
            viridx = ~occidx
            g = reduce(np.dot, (mo[:,viridx].conj().T, fock, mo[:,occidx]))
            return g.ravel()

        grad_kpts = [grad(mo, mo_occ_kpts[k], fock[k])
                     for k, mo in enumerate(mo_coeff_kpts)]
        return np.hstack(grad_kpts)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        raise NotImplementedError

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g after %d cycles',
                        self.e_tot, self.max_cycle)
        return self

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        addons.convert_to_ghf(mf, self)
        return self

    get_init_guess = khf.KRHF.get_init_guess
    init_guess_by_minao = _cast_mol_init_guess(mol_hf.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(mol_hf.init_guess_by_atom)
    init_guess_by_chkfile = mol_ghf.init_guess_by_chkfile

    density_fit = khf.KSCF.density_fit
    rs_density_fit = khf.KSCF.rs_density_fit
    newton = khf.KSCF.newton

    def x2c1e(self):
        '''X2C with spin-orbit coupling effects in spin-orbital basis'''
        from pyscf.pbc.x2c.x2c1e import x2c1e_gscf
        return x2c1e_gscf(self)
    x2c = x2c1e
    sfx2c1e = khf.KSCF.sfx2c1e


if __name__ == '__main__':
    from pyscf.pbc import gto

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

    kpts = cell.make_kpts([2,1,1])
    mf = KGHF(cell, kpts=kpts)
    mf.kernel()

    # x2c1e decoractor to KGHF class.
    #mf = KGHF(cell, kpts=kpts).x2c1e()
    # or
    #mf = KGHF(cell, kpts=kpts).sfx2c1e()
    #mf.kernel()
