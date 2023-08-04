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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory for
arbitrary moment consistency
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.agf2 import ragf2
from pyscf.agf2 import aux_space as aux


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Kwargs:
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Returns:
        :class:`SelfEnergy`
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction

    nmo = agf2.nmo
    nocc = gf_occ.naux
    nvir = gf_vir.naux
    naux = nocc * nocc * nvir
    tol = agf2.weight_tol

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = ragf2.get_frozen_mask(agf2)
        nmo -= np.sum(~mask)

    e = np.zeros((naux))
    v = np.zeros((nmo, naux))

    fpos = np.sqrt(0.5 * os_factor)
    fneg = np.sqrt(0.5 * os_factor + ss_factor)
    fdia = np.sqrt(os_factor)

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)

    coeffs = (gf_occ.coupling, gf_occ.coupling, gf_vir.coupling)
    qeri = _make_qmo_eris_incore(agf2, eri, coeffs)

    p1 = 0
    for i in range(nocc):
        xija = qeri[:,i,:i].reshape(nmo, -1)
        xjia = qeri[:,:i,i].reshape(nmo, -1)
        xiia = qeri[:,i,i].reshape(nmo, -1)
        eija = gf_occ.energy[i] + eja[:i+1]

        p0, p1 = p1, p1 + i*nvir
        e[p0:p1] = eija[:i].ravel()
        v[:,p0:p1] = fneg * (xija - xjia)

        p0, p1 = p1, p1 + i*nvir
        e[p0:p1] = eija[:i].ravel()
        v[:,p0:p1] = fpos * (xija + xjia)

        p0, p1 = p1, p1 + nvir
        e[p0:p1] = eija[i].ravel()
        v[:,p0:p1] = fdia * xiia

    se = aux.SelfEnergy(e, v, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        coupling = np.zeros((agf2.nmo, se.naux))
        coupling[mask] = se.coupling
        se = aux.SelfEnergy(se.energy, coupling, chempot=se.chempot)

    log.timer('se part', *cput0)

    return se


class RAGF2(ragf2.RAGF2):
    ''' Restricted AGF2 with canonical HF reference for arbitrary
        moment consistency

    Attributes:
        nmom : tuple of int
            Compression level of the Green's function and
            self-energy, respectively
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-8.
        conv_tol_nelec : float
            Convergence threshold for the number of electrons. Default
            value is 1e-6.
        max_cycle : int
            Maximum number of AGF2 iterations. Default value is 50.
        max_cycle_outer : int
            Maximum number of outer Fock loop iterations. Default
            value is 20.
        max_cycle_inner : int
            Maximum number of inner Fock loop iterations. Default
            value is 50.
        weight_tol : float
            Threshold in spectral weight of auxiliaries to be considered
            zero. Default 1e-11.
        fock_diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        fock_diis_min_space :
            Minimum space of DIIS. Default value is 1.
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        damping : float
            Damping factor for the self-energy. Default value is 0.0

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_init : float
            Initial correlation energy (truncated MP2)
        converged : bool
            Whether convergence was successful
        se : SelfEnergy
            Auxiliaries of the self-energy
        gf : GreensFunction
            Auxiliaries of the Green's function
    '''

    def __init__(self, mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):

        ragf2.RAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        self.nmom = nmom

        self._keys.update(['nmom'])

    build_se_part = build_se_part

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        ''' Builds the auxiliaries of the self-energy.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : GreensFunction
                Auxiliaries of the Green's function

        Kwargs:
            os_factor : float
                Opposite-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            ss_factor : float
                Same-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            se_prev : SelfEnergy
                Previous self-energy for damping. Default value is None

        Returns:
            :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        fock = None
        if self.nmom[0] is not None:
            fock = self.get_fock(eri=eri, gf=gf)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_occ = se_occ.compress(n=(None, self.nmom[1]))

        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)
        se_vir = se_vir.compress(n=(None, self.nmom[1]))

        se = aux.combine(se_vir, se_occ)
        se = se.compress(phys=fock, n=(self.nmom[0], None))

        if se_prev is not None and self.damping != 0.0:
            se.coupling *= np.sqrt(1.0-self.damping)
            se_prev.coupling *= np.sqrt(self.damping)
            se = aux.combine(se, se_prev)
            se = se.compress(n=self.nmom)

        return se

    def dump_flags(self, verbose=None):
        ragf2.RAGF2.dump_flags(self, verbose=verbose)
        logger.info(self, 'nmom = %s', repr(self.nmom))
        return self

    def run_diis(self, se, diis=None):
        return se


class _ChemistsERIs(ragf2._ChemistsERIs):
    pass

_make_qmo_eris_incore = ragf2._make_qmo_eris_incore



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=3)
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-11
    rhf.run()

    agf2 = RAGF2(rhf, nmom=(None,0))
    agf2.run()

    agf2 = ragf2.RAGF2(rhf)
    agf2.run()
