#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
PBC gamma-point spin-restricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""

from functools import reduce
import numpy as np

from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import current_memory, logger
from pyscf.pbc import df, scf
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri

from pyscf.gw.gw_ac import GWAC as GWAC_mol


class GWAC(GWAC_mol):
    def __init__(self, mf, frozen=None, auxbasis=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        warn_pbc2d_eri(mf)

        GWAC_mol.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.fc = False

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('frozen orbitals = %s', self.frozen)
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('finite-size correction = %s', self.fc)
        log.info('outcore for self-energy= %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('broadening parameter = %.3e', self.eta)
        if self.nw2 is None:
            log.info('number of grids = %d', self.nw)
        else:
            log.info('grid size for W is %d', self.nw)
            log.info('grid size for self-energy is %d', self.nw2)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %.1f', self.ac_iw_cutoff)
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('use perturbative linearized QP eqn = %s', self.qpe_linearized)
        if self.qpe_linearized is True:
            log.info('linearized factor range = %s', self.qpe_linearized_range)
        else:
            log.info('QPE max iter = %d', self.qpe_max_iter)
            log.info('QPE tolerance = %.1e', self.qpe_tol)
        log.info('')
        return

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        if getattr(self._scf, 'with_df', None):
            self.with_df = self._scf.with_df
        else:
            self.with_df = df.DF(self._scf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=False)
        self._keys.update(['with_df'])
        return

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None

        Returns
        -------
        Lpq : double 3d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        max_memory = max(2000, self._scf.max_memory - current_memory()[0] - nao**2 * naux * 8 / 1e6)

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)

        kptijkl = _format_kpts(kpts)
        eri_3d = []
        for LpqR, _, _ in self._scf.with_df.sr_loop(kptijkl[:2], max_memory=0.3 * max_memory, compact=False):
            Lpq = None
            Lpq = nr_e2(LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        eri_3d = np.vstack(eri_3d).reshape(-1, nmo, nmo)

        return eri_3d

    def loop_ao2mo(self, mo_coeff=None, ijslice=None):
        """Transform density-fitting integral from AO to MO by block.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        ijslice : tuple, optional
            tuples for (1st idx start, 1st idx end, 2nd idx start, 2nd idx end), by default None

        Returns
        -------
        eri_3d : double 3d array
            three-center density-fitting matrix in MO in a block
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        max_memory = max(2000, self._scf.max_memory - current_memory()[0] - nao**2 * naux * 8 / 1e6)

        mo = np.asarray(mo_coeff, order='F')
        if ijslice is None:
            ijslice = (0, nmo, 0, nmo)
        nislice = ijslice[1] - ijslice[0]
        njslice = ijslice[3] - ijslice[2]

        kptijkl = _format_kpts(kpts)
        eri_3d = []
        for LpqR, _, _ in self._scf.with_df.sr_loop(kptijkl[:2], max_memory=0.2 * max_memory, compact=False):
            Lpq = None
            Lpq = nr_e2(LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        eri_3d = np.vstack(eri_3d).reshape(-1, nislice, njslice)

        return eri_3d

    def get_sigma_exchange(self, mo_coeff):
        """Get exchange self-energy (EXX).

        Parameters
        ----------
        mo_coeff : double 2d array
            orbital coefficient

        Returns
        -------
        vk : double 2d array
            exchange self-energy
        """
        dm = self._scf.make_rdm1()
        if isinstance(self._scf.with_df, df.GDF):
            rhf = scf.RHF(self.mol).density_fit()
        elif isinstance(self._scf.with_df, df.RSDF):
            rhf = scf.RHF(self.mol).rs_density_fit()
        if hasattr(self._scf, 'sigma'):
            rhf = scf.addons.smearing_(rhf, sigma=self._scf.sigma, method=self._scf.smearing_method)
        rhf.exxdiv = None
        rhf.with_df = self.with_df
        vk = rhf.get_veff(self.mol, dm) - rhf.get_j(self.mol, dm)
        vk = reduce(np.matmul, (mo_coeff.T, vk, mo_coeff))

        if self.fc:
            vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / self.mol.vol) ** (1.0 / 3.0)
            for i in range(self.nocc):
                vk[i, i] = vk[i, i] + vk_corr
        return vk
