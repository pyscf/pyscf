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
unrestricted references for arbitrary moment consistency
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.agf2 import ragf2, uagf2, ragf2_slow
from pyscf.agf2 import aux_space as aux


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped,
        for a single spin.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : tuple of GreensFunction
            Occupied Green's function for each spin
        gf_vir : tuple of GreensFunction
            Virtual Green's function for each spin

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

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_occ[1]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction
    assert type(gf_vir[1]) is aux.GreensFunction

    tol = agf2.weight_tol

    def _build_se_part_spin(spin=0):
        ''' Perform the build for a single spin

        spin = 0: alpha
        spin = 1: beta
        '''

        if spin == 0:
            ab = slice(None)
        else:
            ab = slice(None, None, -1)

        nmoa, nmob = agf2.nmo[ab]
        gfo_a, gfo_b = gf_occ[ab]
        gfv_a, gfv_b = gf_vir[ab]
        noa, nob = gfo_a.naux, gfo_b.naux
        nva, nvb = gfv_a.naux, gfv_b.naux
        naux = nva*noa*(noa-1)//2 + nvb*noa*nob

        if not (agf2.frozen is None or agf2.frozen == 0):
            mask = uagf2.get_frozen_mask(agf2)
            nmoa -= np.sum(~mask[ab][0])
            nmob -= np.sum(~mask[ab][1])

        e = np.zeros((naux))
        v = np.zeros((nmoa, naux))

        falph = np.sqrt(ss_factor)
        fbeta = np.sqrt(os_factor)

        eja_a = lib.direct_sum('j,a->ja', gfo_a.energy, -gfv_a.energy)
        eja_b = lib.direct_sum('j,a->ja', gfo_b.energy, -gfv_b.energy)

        ca = (gf_occ[0].coupling, gf_occ[0].coupling, gf_vir[0].coupling)
        cb = (gf_occ[1].coupling, gf_occ[1].coupling, gf_vir[1].coupling)
        qeri = _make_qmo_eris_incore(agf2, eri, ca, cb, spin=spin)
        qeri_aa, qeri_ab = qeri

        p1 = 0
        for i in range(noa):
            xija_aa = qeri_aa[:,i,:i].reshape(nmoa, -1)
            xjia_aa = qeri_aa[:,:i,i].reshape(nmoa, -1)
            xija_ab = qeri_ab[:,i].reshape(nmoa, -1)

            eija_aa = gfo_a.energy[i] + eja_a[:i]
            eija_ab = gfo_a.energy[i] + eja_b

            p0, p1 = p1, p1 + i*nva
            e[p0:p1] = eija_aa.ravel()
            v[:,p0:p1] = falph * (xija_aa - xjia_aa)

            p0, p1 = p1, p1 + nob*nvb
            e[p0:p1] = eija_ab.ravel()
            v[:,p0:p1] = fbeta * xija_ab

        se = aux.SelfEnergy(e, v, chempot=gfo_a.chempot)
        se.remove_uncoupled(tol=tol)

        if not (agf2.frozen is None or agf2.frozen == 0):
            coupling = np.zeros((agf2.nmo[ab][0], se.naux))
            coupling[mask[ab][0]] = se.coupling
            se = aux.SelfEnergy(se.energy, coupling, chempot=gfo_a.chempot)

        return se

    se_a = _build_se_part_spin(0)

    cput0 = log.timer('se part (alpha)', *cput0)

    se_b = _build_se_part_spin(1)

    cput0 = log.timer('se part (beta)', *cput0)

    return (se_a, se_b)


class UAGF2(uagf2.UAGF2):
    ''' Unrestricted AGF2 with canonical HF reference for arbitrary
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
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
    '''

    _keys = {'nmom'}

    def __init__(self, mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):

        uagf2.UAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        self.nmom = nmom

    build_se_part = build_se_part

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        ''' Builds the auxiliaries of the self-energy.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : tuple of GreensFunction
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

        Returns
            :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        focka = fockb = None
        if self.nmom[1] is not None:
            focka, fockb = self.get_fock(eri=eri, gf=gf)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = {'os_factor': os_factor, 'ss_factor': ss_factor}
        gf_occ = (gf[0].get_occupied(), gf[1].get_occupied())
        gf_vir = (gf[0].get_virtual(), gf[1].get_virtual())

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_occ = (se_occ[0].compress(n=(None, self.nmom[1])),
                  se_occ[1].compress(n=(None, self.nmom[1])))

        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)
        se_vir = (se_vir[0].compress(n=(None, self.nmom[1])),
                  se_vir[1].compress(n=(None, self.nmom[1])))

        se_a = aux.combine(se_occ[0], se_vir[0])
        se_a = se_a.compress(phys=focka, n=(self.nmom[0], None))

        se_b = aux.combine(se_occ[1], se_vir[1])
        se_b = se_b.compress(phys=fockb, n=(self.nmom[0], None))

        if se_prev is not None and self.damping != 0.0:
            se_a_prev, se_b_prev = se_prev
            se_a.coupling *= np.sqrt(1.0-self.damping)
            se_b.coupling *= np.sqrt(1.0-self.damping)
            se_a_prev.coupling *= np.sqrt(self.damping)
            se_b_prev.coupling *= np.sqrt(self.damping)
            se_a = aux.combine(se_a, se_a_prev)
            se_b = aux.combine(se_b, se_b_prev)
            se_a = se_a.compress(n=(None,0))
            se_b = se_b.compress(n=(None,0))

        return (se_a, se_b)

    def dump_flags(self, verbose=None):
        uagf2.UAGF2.dump_flags(self, verbose=verbose)
        logger.info(self, 'nmom = %s', repr(self.nmom))
        return self

    def run_diis(self, se, diis=None):
        return se


class _ChemistsERIs(uagf2._ChemistsERIs):
    pass

_make_qmo_eris_incore = uagf2._make_qmo_eris_incore



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', charge=-1, spin=1, verbose=3)
    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-11
    uhf.run()

    agf2 = UAGF2(uhf)
    agf2.run()

    agf2 = uagf2.UAGF2(uhf)
    agf2.run()
