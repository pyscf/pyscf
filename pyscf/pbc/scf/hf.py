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
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import sys
import time
import numpy as np
import h5py
from pyscf.scf import hf as mol_hf
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf.hf import make_rdm1
from pyscf.pbc import tools
from pyscf.pbc.gto import ecp
from pyscf.pbc.gto.pseudo import get_pp
from pyscf.pbc.scf import chkfile
from pyscf.pbc import df
from pyscf.pbc.scf import addons
from pyscf import __config__


def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
# Avoid pbcopt's prescreening in the lattice sum, for better accuracy
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt,
                       pbcopt=lib.c_null_ptr())
    cond = np.max(lib.cond(s))
    if cond * cell.precision > 1e2:
        prec = 1e2 / cond
        rmin = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
        if cell.rcut < rmin:
            logger.warn(cell, 'Singularity detected in overlap matrix.  '
                        'Integral accuracy may be not enough.\n      '
                        'You can adjust  cell.precision  or  cell.rcut  to '
                        'improve accuracy.  Recommended values are\n      '
                        'cell.precision = %.2g  or smaller.\n      '
                        'cell.rcut = %.4g  or larger.', prec, rmin)
    return s


def get_hcore(cell, kpt=np.zeros(3)):
    '''Get the core Hamiltonian AO matrix.
    '''
    hcore = get_t(cell, kpt)
    if cell.pseudo:
        hcore += get_pp(cell, kpt)
    else:
        hcore += get_nuc(cell, kpt)
    if len(cell._ecpbas) > 0:
        hcore += ecp.ecp_int(cell, kpt)
    return hcore


def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.
    '''
    return cell.pbc_intor('int1e_kin', hermi=1, kpts=kpt)


def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    return df.FFTDF(cell).get_nuc(kpt)


def get_j(cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            An arbitrary "band" k-point at which J is evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpts_band, with_k=False)[0]


def get_jk(mf, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            An arbitrary "band" k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpts_band, exxdiv=mf.exxdiv)


def get_bands(mf, kpts_band, cell=None, dm=None, kpt=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points.

    Returns:
        mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    kpts_band = np.asarray(kpts_band)
    single_kpt_band = (hasattr(kpts_band, 'ndim') and kpts_band.ndim == 1)
    kpts_band = kpts_band.reshape(-1,3)

    fock = mf.get_hcore(cell, kpts_band)
    fock = fock + mf.get_veff(cell, dm, kpt=kpt, kpts_band=kpts_band)
    s1e = mf.get_ovlp(cell, kpts_band)
    nkpts = len(kpts_band)
    mo_energy = []
    mo_coeff = []
    for k in range(nkpts):
        e, c = mf.eig(fock[k], s1e[k])
        mo_energy.append(e)
        mo_coeff.append(c)

    if single_kpt_band:
        mo_energy = mo_energy[0]
        mo_coeff = mo_coeff[0]
    return mo_energy, mo_coeff


def init_guess_by_chkfile(cell, chkfile_name, project=None, kpt=None):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, (nao,nao) ndarray
    '''
    from pyscf.pbc.scf import uhf
    dm = uhf.init_guess_by_chkfile(cell, chkfile_name, project, kpt)
    return dm[0] + dm[1]

get_fock = mol_hf.get_fock
get_occ = mol_hf.get_occ
get_grad = mol_hf.get_grad
make_rdm1 = mol_hf.make_rdm1
energy_elec = mol_hf.energy_elec


class SCF(mol_hf.SCF):
    '''SCF base class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.

        exxdiv : str
            Exchange divergence treatment, can be one of

            | None : ignore G=0 contribution in exchange integral
            | 'ewald' : Ewald summation for G=0 in exchange integral

        with_df : density fitting object
            Default is the FFT based DF model. For all-electron calculation,
            MDF model is favored for better accuracy.  See also :mod:`pyscf.pbc.df`.

        direct_scf : bool
            When this flag is set to true, the J/K matrices will be computed
            directly through the underlying with_df methods.  Otherwise,
            depending the available memory, the 4-index integrals may be cached
            and J/K matrices are computed based on the 4-index integrals.
    '''

    direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', False)

    def __init__(self, cell, kpt=np.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        self.kpt = kpt

        self._keys = self._keys.union(['cell', 'exxdiv', 'with_df'])

    @property
    def kpt(self):
        if 'kpt' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            self.kpt = self.__dict__.pop('kpt')
        return self.with_df.kpts.reshape(3)
    @kpt.setter
    def kpt(self, x):
        self.with_df.kpts = np.reshape(x, (-1,3))

    def build(self, cell=None):
        if 'kpt' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            self.kpt = self.__dict__.pop('kpt')
        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def dump_flags(self):
        mol_hf.SCF.dump_flags(self)
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'kpt = %s', self.kpt)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        if (self.cell.dimension == 3 and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(self.cell, [self.kpt])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung/cell.vol = %.12g',
                        madelung*self.cell.nelectron * -.5)
        logger.info(self, 'DF object = %s', self.with_df)
        if not hasattr(self.with_df, 'build'):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags()
        return self

    def check_sanity(self):
        mol_hf.SCF.check_sanity(self)
        self.with_df.check_sanity()
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF or MDF',
                        self.exxdiv)
        return self

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt)
        else:
            nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            nuc += ecp.ecp_int(cell, kpt)
        return nuc + cell.pbc_intor('int1e_kin', 1, 1, kpt)

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_ovlp(cell, kpt)

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None):
        r'''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.
        for particular k-point (kpt).

        When kpts_band is given, the J, K matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]
            K_{pq} = \sum_{rs} (pr|sq) dm[r,s]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (time.clock(), time.time())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (kpts_band is None and
            (self.exxdiv == 'ewald' or not self.exxdiv) and
            (self._eri is not None or cell.incore_anyway or
             (not self.direct_scf and self._is_mem_enough()))):
            if self._eri is None:
                logger.debug(self, 'Building PBC AO integrals incore')
                self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            vj, vk = mol_hf.dot_eri_dm(self._eri, dm, hermi)

            if self.exxdiv == 'ewald':
                from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
                # G=0 is not inculded in the ._eri integrals
                _ewald_exxdiv_for_G0(self.cell, kpt, dm.reshape(-1,nao,nao),
                                     vk.reshape(-1,nao,nao))
        else:
            vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi,
                                         kpt, kpts_band, exxdiv=self.exxdiv)

        logger.timer(self, 'vj and vk', *cpu0)
        vj = _format_jks(vj, dm, kpts_band)
        vk = _format_jks(vk, dm, kpts_band)
        return vj, vk

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None):
        r'''Compute J matrix for the given density matrix and k-point (kpt).
        When kpts_band is given, the J matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        #return self.get_jk(cell, dm, hermi, kpt, kpts_band)[0]
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (time.clock(), time.time())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (kpts_band is None and
            (self._eri is not None or cell.incore_anyway or
             (not self.direct_scf and self._is_mem_enough()))):
            if self._eri is None:
                logger.debug(self, 'Building PBC AO integrals incore')
                self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            vj, vk = mol_hf.dot_eri_dm(self._eri, dm.reshape(-1,nao,nao), hermi)
        else:
            vj = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi,
                                     kpt, kpts_band, with_k=False)[0]
        logger.timer(self, 'vj', *cpu0)
        return _format_jks(vj, dm, kpts_band)

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpts_band)[1]

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        return vj - vk * .5

    def get_jk_incore(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG, kpt=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.
        '''
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if self._eri is None:
            self._eri = self.with_df.get_ao_eri(kpt, compact=True)
        return self.get_jk(cell, dm, hermi, verbose, kpt)

    def energy_nuc(self):
        return self.cell.energy_nuc()

    get_bands = get_bands

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        # skip dipole memont for crystal
        return

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        if cell.dimension < 3:
            ne = np.einsum('ij,ji', dm, self.get_ovlp(cell))
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
        return mol_hf.SCF.init_guess_by_1e(self, cell)

    def init_guess_by_chkfile(self, chk=None, project=None, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return init_guess_by_chkfile(self.cell, chk, project, kpt)
    def from_chk(self, chk=None, project=None, kpt=None):
        return self.init_guess_by_chkfile(chk, project, kpt)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpt'] = self.kpt
        return self

    def _is_mem_enough(self):
        nao = self.cell.nao_nr()
        if abs(self.kpt).sum() < 1e-9:
            mem_need = nao**4*8/4/1e6
        else:
            mem_need = nao**4*16/1e6
        return mem_need + lib.current_memory()[0] < self.max_memory*.95

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df)

    def mix_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import mdf_jk
        return mdf_jk.density_fit(self, auxbasis, with_df)

    def sfx2c1e(self):
        from pyscf.pbc.x2c import sfx2c1e
        return sfx2c1e.sfx2c1e(self)
    x2c = x2c1e = sfx2c1e


class RHF(SCF, mol_hf.RHF):

    check_sanity = mol_hf.RHF.check_sanity
    stability = mol_hf.RHF.stability

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF'''
        addons.convert_to_rhf(mf, self)
        return self

    def nuc_grad_method(self):
        raise NotImplementedError


def _format_jks(vj, dm, kpts_band):
    if kpts_band is None:
        vj = vj.reshape(dm.shape)
    elif kpts_band.ndim == 1:  # a single k-point on bands
        vj = vj.reshape(dm.shape)
    elif hasattr(dm, "ndim") and dm.ndim == 2:
        vj = vj[0]
    return vj
