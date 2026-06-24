#!/usr/bin/env python
# Copyright 2020-2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.scf import hf as mol_hf
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.scf import khf_ksymm, khf, kuhf
from pyscf.pbc.lib.kpts import KPoints

@lib.with_doc(kuhf.get_occ.__doc__)
def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    if mo_energy_kpts is None:
        mo_energy_kpts = mf.mo_energy
    kpts = mf.kpts
    assert isinstance(kpts, KPoints)

    nocc_a, nocc_b = mf.nelec
    mo_energy_kpts = kpts.transform_mo_energy(mo_energy_kpts)
    mo_energy_a = np.sort(np.hstack(mo_energy_kpts[0]))
    nmo = len(mo_energy_a)
    if nocc_a > nmo or nocc_b > nmo:
        raise RuntimeError('Failed to assign mo_occ. '
                           f'nelec ({nocc_a}, {nocc_b}) > Nmo ({nmo})')
    fermi_a = mo_energy_a[nocc_a-1]
    mo_occ_kpts = [[], []]
    for mo_e in mo_energy_kpts[0]:
        mo_occ_kpts[0].append((mo_e <= fermi_a).astype(np.double))

    if nocc_b > 0:
        mo_energy_b = np.sort(np.hstack(mo_energy_kpts[1]))
        fermi_b = mo_energy_b[nocc_b-1]
        for mo_e in mo_energy_kpts[1]:
            mo_occ_kpts[1].append((mo_e <= fermi_b).astype(np.double))
    else:
        for mo_e in mo_energy_kpts[1]:
            mo_occ_kpts[1].append(np.zeros_like(mo_e))

    if nocc_a < nmo and nocc_b < nmo:
        homo = homo_a = fermi_a
        homo_b = None
        if nocc_b > 0:
            homo = max(homo, fermi_b)
        lumo = lumo_b = mo_energy_b[nocc_b]
        lumo_a = None
        if nocc_a < nmo:
            lumo_a = mo_energy_a[nocc_a]
            lumo = min(lumo, lumo_a)
        gap = (lumo - homo) * nist.HARTREE2EV
        mf.scf_summary['gap'] = gap

        if lumo_a is not None:
            logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', homo_a, lumo_a)
        else:
            logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', homo_a)
        if homo_b is not None:
            logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g', homo_b, lumo_b)
        else:
            logger.info(mf, 'beta               LUMO = %.12g', lumo_b)
        if homo+1e-3 > lumo:
            logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', homo, lumo)
        else:
            logger.info(mf, '  HOMO = %.12g  LUMO = %.12g  gap/eV = %.5f',
                        homo, lumo, gap)

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=nmo)
        logger.debug(mf, '     k-point                  alpha mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(kpts, kpts_in_ibz=False)):
            if (np.count_nonzero(mo_occ_kpts[0][k]) > 0 and
                np.count_nonzero(mo_occ_kpts[0][k] == 0) > 0):
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                             k, kpt[0], kpt[1], kpt[2],
                             np.sort(mo_energy_kpts[0][k][mo_occ_kpts[0][k]> 0]),
                             np.sort(mo_energy_kpts[0][k][mo_occ_kpts[0][k]==0]))
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[0][k])
        logger.debug(mf, '     k-point                  beta  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(kpts, kpts_in_ibz=False)):
            if (np.count_nonzero(mo_occ_kpts[1][k]) > 0 and
                np.count_nonzero(mo_occ_kpts[1][k] == 0) > 0):
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                             k, kpt[0], kpt[1], kpt[2],
                             np.sort(mo_energy_kpts[1][k][mo_occ_kpts[1][k]> 0]),
                             np.sort(mo_energy_kpts[1][k][mo_occ_kpts[1][k]==0]))
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[1][k])
        np.set_printoptions(threshold=1000)

    if isinstance(kpts, KPoints):
        mo_occ_kpts[0] = kpts.check_mo_occ_symmetry(mo_occ_kpts[0], tol=1e-4)
        mo_occ_kpts[1] = kpts.check_mo_occ_symmetry(mo_occ_kpts[1], tol=1e-4)
    return np.array(mo_occ_kpts)

@lib.with_doc(kuhf.energy_elec.__doc__)
def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None or getattr(vhf_kpts, 'ecoul', None) is None:
        vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    wtk = mf.kpts.weights_ibz

    e1 = np.einsum('k,skij,kji->', wtk, dm_kpts, h1e_kpts)
    e2 = np.einsum('k,skij,skji->', wtk, dm_kpts, vhf_kpts) * 0.5
    ecoul = vhf_kpts.ecoul
    exx = e2 - ecoul
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e2.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exx.real
    logger.debug(mf, 'E1 = %s  E2 = %s  E_coul = %s  Exc = %s', e1, e2, ecoul, exx)
    if kuhf.CHECK_COULOMB_IMAG and abs(e2.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e2.imag)
    return (e1+e2).real, e2.real

get_rho = khf_ksymm.get_rho


class KsymAdaptedKUHF(khf_ksymm.KsymAdaptedKSCF, kuhf.KUHF):
    """
    KUHF with k-point symmetry
    """

    get_occ = get_occ
    energy_elec = energy_elec
    get_rho = get_rho

    to_ks = kuhf.KUHF.to_ks
    convert_from_ = kuhf.KUHF.convert_from_

    def __init__(self, cell, kpts=libkpts.KPoints(),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 use_ao_symmetry=True):
        khf_ksymm.ksymm_scf_common_init(self, cell, kpts, use_ao_symmetry)
        kuhf.KUHF.__init__(self, cell, kpts, exxdiv)

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            cell = self.cell
            nkpts = self.kpts.nkpts
            ne = cell.tot_electrons(nkpts)
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
        khf_ksymm.KsymAdaptedKSCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per unit cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm_kpts = mol_hf.SCF.get_init_guess(self, cell, key)
        assert dm_kpts.shape[0]==2
        if dm_kpts.ndim != 4:
            nkpts = self.kpts.nkpts_ibz
            # dm[spin,nao,nao] at gamma point -> dm_kpts[spin,nkpts,nao,nao]
            dm_kpts = np.repeat(dm_kpts[:,None,:,:], nkpts, axis=1)
        elif dm_kpts.shape[1] != self.kpts.nkpts_ibz:
            dm_kpts = dm_kpts[:,self.kpts.ibz2bz]

        ne = lib.einsum('k,xkij,kji->x', self.kpts.weights_ibz, dm_kpts, s1e).real
        nkpts = self.kpts.nkpts
        ne *= nkpts
        nelec = np.asarray(self.nelec)
        if np.any(abs(ne - nelec) > 0.01*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne.mean()/nkpts, nelec/nkpts)
            dm_kpts *= (nelec / ne).reshape(2,-1,1,1)
        return dm_kpts

    def eig(self, h_kpts, s_kpts, overwrite=False, x=None):
        e_a, c_a = khf_ksymm.KsymAdaptedKSCF.eig(self, h_kpts[0], s_kpts)
        e_b, c_b = khf_ksymm.KsymAdaptedKSCF.eig(self, h_kpts[1], s_kpts)
        return np.array((e_a,e_b)), (c_a,c_b)

    def get_orbsym(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if s is None:
            s = self.get_ovlp()

        orbsym_a = khf_ksymm.KsymAdaptedKSCF.get_orbsym(self, mo_coeff[0], s)
        orbsym_b = khf_ksymm.KsymAdaptedKSCF.get_orbsym(self, mo_coeff[1], s)
        return (orbsym_a, orbsym_b)

    orbsym = property(get_orbsym)

    def _finalize(self):
        from pyscf.scf import chkfile as mol_chkfile
        kuhf.KUHF._finalize(self)
        if not self.use_ao_symmetry:
            return

        orbsym = self.get_orbsym()
        for s in range(2):
            for k, mo_e in enumerate(self.mo_energy[s]):
                idx = np.argsort(mo_e.round(9), kind='stable')
                self.mo_energy[s][k] = self.mo_energy[s][k][idx]
                self.mo_occ[s][k] = self.mo_occ[s][k][idx]
                self.mo_coeff[s][k] = lib.tag_array(self.mo_coeff[s][k][:,idx],
                                                    orbsym=orbsym[s][k][idx])
        self.dump_chk({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                       'mo_coeff': self.mo_coeff, 'mo_occ': self.mo_occ})
        return self

KUHF = KsymAdaptedKUHF
