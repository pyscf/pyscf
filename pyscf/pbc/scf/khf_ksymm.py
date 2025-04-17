#!/usr/bin/env python
# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
import h5py
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf as mol_hf
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.scf import khf

@lib.with_doc(khf.get_occ.__doc__)
def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    if mo_energy_kpts is None:
        mo_energy_kpts = mf.mo_energy
    cell = mf.cell
    kpts = mf.kpts
    assert isinstance(kpts, libkpts.KPoints)

    nocc = cell.tot_electrons(kpts.nkpts) // 2
    mo_energy_kpts = kpts.transform_mo_energy(mo_energy_kpts)
    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)

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
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts, kpts_in_ibz=False)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
        np.set_printoptions(threshold=1000)

    mo_occ_kpts = kpts.check_mo_occ_symmetry(mo_occ_kpts)
    return mo_occ_kpts

@lib.with_doc(khf.energy_elec.__doc__)
def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    kpts_weights = mf.kpts.weights_ibz
    e1 = np.einsum('k,kij,kji', kpts_weights, dm_kpts, h1e_kpts)
    e_coul = np.einsum('k,kij,kji', kpts_weights, dm_kpts, vhf_kpts) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if khf.CHECK_COULOMB_IMAG and abs(e_coul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

@lib.with_doc(khf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpts=None):
    if isinstance(kpts, np.ndarray):
        return khf.get_rho(mf, dm, grids, kpts)
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts

    if isinstance(dm[0], np.ndarray) and dm[0].ndim == 3:
        ndm = len(dm[0])
    else:
        ndm = len(dm)
    if ndm != kpts.nkpts_ibz:
        raise RuntimeError("Number of input density matrices does not \
                           match the number of IBZ kpts: %d vs %d."
                           % (ndm, kpts.nkpts_ibz))
    dm = kpts.transform_dm(dm)
    return khf.get_rho(mf, dm, grids, kpts.kpts)

def eig(kmf, h_kpts, s_kpts):
    from pyscf.scf.hf_symm import eig as eig_symm
    cell = kmf.cell
    symm_orb = cell.symm_orb
    irrep_id = cell.irrep_id

    nkpts = len(h_kpts)
    assert len(symm_orb) == nkpts
    eig_kpts = []
    mo_coeff_kpts = []

    for k in range(nkpts):
        e, c = eig_symm(kmf, h_kpts[k], s_kpts[k], symm_orb[k], irrep_id[k])
        eig_kpts.append(e)
        mo_coeff_kpts.append(c)
    return eig_kpts, mo_coeff_kpts

def ksymm_scf_common_init(kmf, cell, kpts, use_ao_symmetry=True):
    kmf._kpts = None
    kmf.use_ao_symmetry = (cell.dimension == 3 and
                           use_ao_symmetry and
                           not kpts.time_reversal and
                           kpts.symmorphic and
                           len(kpts.little_cogroup_ops) > 0)
    if kmf.use_ao_symmetry and cell.symm_orb is None:
        cell._build_symmetry(kpts)
    return kmf


class KsymAdaptedKSCF(khf.KSCF):
    """
    KRHF with k-point symmetry
    """

    _keys = {'use_ao_symmetry'}

    get_occ = get_occ
    get_rho = get_rho
    energy_elec = energy_elec

    def __init__(self, cell, kpts=libkpts.KPoints(),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 use_ao_symmetry=True):
        ksymm_scf_common_init(self, cell, kpts, use_ao_symmetry)
        khf.KSCF.__init__(self, cell, kpts=kpts, exxdiv=exxdiv)

    @property
    def kpts(self):
        if 'kpts' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            kpts_ibz = self.__dict__.pop('kpts')
            if len(kpts_ibz) != self._kpts.nkpts_ibz:
                raise RuntimeError("chkfile is not consistent with the current system.")
        return self._kpts

    @kpts.setter
    def kpts(self, kpts):
        if isinstance(kpts, np.ndarray):
            logger.warn(self, "Input kpts is ndarray, building kpts object without symmetry.")
            kpts = libkpts.make_kpts(self.cell, kpts=kpts)
        elif not isinstance(kpts, libkpts.KPoints):
            raise TypeError("Input kpts have wrong type: %s" % type(kpts))
        kpts_bz = kpts.kpts
        self.with_df.kpts = np.reshape(kpts_bz, (-1,3))
        self._kpts = kpts

    @property
    def kmesh(self):
        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kpts_bz = self._kpts.kpts
        kmesh = kpts_to_kmesh(kpts_bz)
        if len(kpts_bz) != np.prod(kmesh):
            logger.WARN(self, 'K-points specified in %s are not Monkhorst-Pack %s grids',
                        self, kmesh)
        return kmesh

    @kmesh.setter
    def kmesh(self, x):
        self.kpts = self.cell.make_kpts(x)

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'N kpts (BZ) = %d', self.kpts.nkpts)
        logger.debug(self, 'kpts (BZ) = %s', self.kpts.kpts)
        logger.debug(self, 'kpts weights (BZ) = %s', self.kpts.weights)
        logger.info(self, 'N kpts (IBZ) = %d', self.kpts.nkpts_ibz)
        logger.debug(self, 'kpts (IBZ) = %s', self.kpts.kpts_ibz)
        logger.debug(self, 'kpts weights (IBZ) = %s', self.kpts.weights_ibz)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, [self.kpts.kpts])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            # FIXME: consider the fractional num_electron or not? This maybe
            # relates to the charged system.
            nelectron = float(self.cell.tot_electrons(self.kpts.nkpts)) / self.kpts.nkpts
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*nelectron * -.5)
        logger.info(self, 'DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            self.with_df.dump_flags(verbose)
        return self

    @lib.with_doc(khf.get_ovlp.__doc__)
    def get_ovlp(self, cell=None, kpts=None):
        if isinstance(kpts, np.ndarray):
            return khf.KSCF.get_ovlp(self, cell, kpts)
        if kpts is None: kpts = self.kpts
        return khf.KSCF.get_ovlp(self, cell, kpts.kpts_ibz)

    @lib.with_doc(khf.get_hcore.__doc__)
    def get_hcore(self, cell=None, kpts=None):
        if isinstance(kpts, np.ndarray):
            return khf.KSCF.get_hcore(self, cell, kpts)
        if kpts is None: kpts = self.kpts
        return khf.KSCF.get_hcore(self, cell, kpts.kpts_ibz)

    @lib.with_doc(khf.get_jk.__doc__)
    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if isinstance(kpts, np.ndarray):
            return super().get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                                                       with_j, with_k, omega, **kwargs)
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        #get dms for each kpt in BZ
        if isinstance(dm_kpts[0], np.ndarray) and dm_kpts[0].ndim == 3:
            ndm = len(dm_kpts[0])
        else:
            ndm = len(dm_kpts)
        if ndm != kpts.nkpts_ibz:
            raise RuntimeError("Number of input density matrices does not \
                               match the number of IBZ kpts: %d vs %d."
                               % (ndm, kpts.nkpts_ibz))
        dm_kpts = kpts.transform_dm(dm_kpts)
        if kpts_band is None: kpts_band = kpts.kpts_ibz
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            raise NotImplementedError('rsjk with k-points symmetry')
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts.kpts, kpts_band,
                                         with_j, with_k, omega, exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def init_guess_by_chkfile(self, chk=None, project=None, kpts=None):
        if isinstance(kpts, np.ndarray):
            return super().init_guess_by_chkfile(chk, project, kpts)
        if kpts is None: kpts = self.kpts
        return super().init_guess_by_chkfile(chk, project, kpts.kpts_ibz)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = self.kpts.kpts_ibz #FIXME Shall we rebuild kpts? If so, more info is needed.
        return self

    def eig(self, h_kpts, s_kpts):
        if self.use_ao_symmetry:
            return eig(self, h_kpts, s_kpts)
        else:
            return khf.KSCF.eig(self, h_kpts, s_kpts)

    def get_orbsym(self, mo_coeff=None, s=None):
        if not self.use_ao_symmetry:
            raise RuntimeError("AO symmetry not initiated")
        from pyscf.scf.hf_symm import get_orbsym
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if s is None:
            s = self.get_ovlp()

        cell = self.cell
        symm_orb = cell.symm_orb
        irrep_id = cell.irrep_id
        orbsym = []
        for k in range(len(mo_coeff)):
            orbsym_k = np.asarray(get_orbsym(cell, mo_coeff[k], s=s[k],
                                             symm_orb=symm_orb[k], irrep_id=irrep_id[k]))
            orbsym.append(orbsym_k)
        return orbsym

    orbsym = property(get_orbsym)

    def _finalize(self):
        khf.KSCF._finalize(self)
        if not self.use_ao_symmetry:
            return self

        orbsym = self.get_orbsym()
        for k, mo_e in enumerate(self.mo_energy):
            idx = np.argsort(mo_e.round(9), kind='stable')
            self.mo_energy[k] = self.mo_energy[k][idx]
            self.mo_occ[k] = self.mo_occ[k][idx]
            self.mo_coeff[k] = lib.tag_array(self.mo_coeff[k][:,idx], orbsym=orbsym[k][idx])

        self.dump_chk({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                       'mo_coeff': self.mo_coeff, 'mo_occ': self.mo_occ})
        return self

    def to_khf(self):
        '''transform to non-symmetry object
        '''
        from pyscf.pbc.scf import kuhf_ksymm, kghf_ksymm
        from pyscf.pbc.scf import khf, kuhf, kghf
        from pyscf.pbc.dft import krks, krks_ksymm, kuks, kuks_ksymm
        from pyscf.scf import addons as mol_addons

        def update_mo_(mf, mf1):
            kpts = mf.kpts
            if mf.mo_energy is not None:
                mo_energy = kpts.transform_mo_energy(mf.mo_energy)
                mo_occ = kpts.transform_mo_occ(mf.mo_occ)

                if isinstance(mf, kghf_ksymm.KGHF):
                    mo_coeff = np.asarray(mf.mo_coeff)
                    nao = mo_coeff.shape[1] // 2
                    mo_coeff_alpha = kpts.transform_mo_coeff(mo_coeff[:,:nao])
                    mo_coeff_beta = kpts.transform_mo_coeff(mo_coeff[:,nao:])
                    mo_coeff = []
                    for k in range(len(mo_coeff_alpha)):
                        mo_coeff.append(np.vstack((mo_coeff_alpha[k], mo_coeff_beta[k])))
                    mo_coeff = np.asarray(mo_coeff)
                else:
                    mo_coeff = kpts.transform_mo_coeff(mf.mo_coeff)

                mf1.mo_coeff = mo_coeff
                mf1.mo_occ = mo_occ
                mf1.mo_energy = mo_energy
            return mf1

        known_cls = {KsymAdaptedKRHF : khf.KRHF,
                     kuhf_ksymm.KUHF : kuhf.KUHF,
                     kghf_ksymm.KGHF : kghf.KGHF,
                     krks_ksymm.KRKS : krks.KRKS,
                     kuks_ksymm.KUKS : kuks.KUKS}

        out = mol_addons._object_without_soscf(self, known_cls, False)
        out.__dict__.pop('kpts', None)
        return update_mo_(self, out)

    def sfx2c1e(self):
        raise NotImplementedError
    x2c = x2c1e = sfx2c1e


class KsymAdaptedKRHF(KsymAdaptedKSCF, khf.KRHF):

    to_ks = khf.KRHF.to_ks
    convert_from_ = khf.KRHF.convert_from_

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm_kpts = mol_hf.SCF.get_init_guess(self, cell, key)
        if dm_kpts.ndim == 2:
            dm_kpts = np.asarray([dm_kpts]*self.kpts.nkpts_ibz)
        elif len(dm_kpts) != self.kpts.nkpts_ibz:
            dm_kpts = dm_kpts[self.kpts.ibz2bz]

        ne = lib.einsum('k,kij,kji', self.kpts.weights_ibz, dm_kpts, s1e).real
        nkpts = self.kpts.nkpts
        ne *= nkpts
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 0.01*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

KRHF = KsymAdaptedKRHF
