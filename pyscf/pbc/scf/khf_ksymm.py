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
    if khf.CHECK_COULOMB_IMAG and abs(e_coul.imag > mf.cell.precision*10):
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

class KsymAdaptedKSCF(khf.KSCF):
    """
    KRHF with k-point symmetry
    """
    def __init__(self, cell, kpts=libkpts.KPoints(),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        self._kpts = None
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

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        dm_kpts = None
        key = key.lower()
        if key == '1e' or key == 'hcore':
            dm_kpts = self.init_guess_by_1e(cell)
        elif getattr(cell, 'natm', 0) == 0:
            logger.info(self, 'No atom found in cell. Use 1e initial guess')
            dm_kpts = self.init_guess_by_1e(cell)
        elif key == 'atom':
            dm = self.init_guess_by_atom(cell)
        elif key[:3] == 'chk':
            try:
                dm_kpts = self.from_chk()
            except (IOError, KeyError):
                logger.warn(self, 'Fail to read %s. Use MINAO initial guess',
                            self.chkfile)
                dm = self.init_guess_by_minao(cell)
        else:
            dm = self.init_guess_by_minao(cell)

        if dm_kpts is None:
            dm_kpts = lib.asarray([dm]*self.kpts.nkpts_ibz)

        ne = np.einsum('k,kij,kji', self.kpts.weights_ibz, dm_kpts, self.get_ovlp(cell)).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relate to the charged system.
        nkpts = self.kpts.nkpts
        ne *= nkpts
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 1e-7*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

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
            return super(KsymAdaptedKSCF, self).get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
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
        vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts.kpts, kpts_band,
                                     with_j, with_k, omega, exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def init_guess_by_chkfile(self, chk=None, project=None, kpts=None):
        if isinstance(kpts, np.ndarray):
            return super(KsymAdaptedKSCF, self).init_guess_by_chkfile(chk, project, kpts)
        if kpts is None: kpts = self.kpts
        return super(KsymAdaptedKSCF, self).init_guess_by_chkfile(chk, project, kpts.kpts_ibz)

    def dump_chk(self, envs):
        if self.chkfile:
            mol_hf.SCF.dump_chk(self, envs)
            with h5py.File(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = self.kpts.kpts_ibz #FIXME Shall we rebuild kpts? If so, more info is needed.
        return self

    get_rho = get_rho
    energy_elec = energy_elec

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

        out = mol_addons._object_without_soscf(self, known_cls, remove_df=False)
        out.__dict__.pop('kpts', None)
        out.with_df = self.with_df
        return update_mo_(self, out)


class KsymAdaptedKRHF(KsymAdaptedKSCF, khf.KRHF):
    def nuc_grad_method(self):
        raise NotImplementedError()

KRHF = KsymAdaptedKRHF
