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

import numpy

from pyscf import lib
from pyscf.cc import rccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.pbc import mp

class RCCSD(rccsd.RCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(self._scf)
        if mbpt2:
            pt = mp.RMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = numpy.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2
        return rccsd.RCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        # _scf.exxdiv affects eris.fock. HF exchange correction should be
        # excluded from the Fock matrix.
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        # eris.mo_energy so far is just the diagonal part of the Fock matrix
        # without the exxdiv treatment. Here to add the exchange correction to
        # get better orbital energies. It is important for the low-dimension
        # systems since their occupied and the virtual orbital energies may
        # overlap which may lead to numerical issue in the CCSD iterations.
        #if mo_coeff is self._scf.mo_coeff:
        #    eris.mo_energy = self._scf.mo_energy[self.get_frozen_mask()]
        #else:

        # Add the HFX correction of Ewald probe charge method.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the
        # correct value.
        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        return eris

class UCCSD(uccsd.UCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(self._scf)
        if mbpt2:
            pt = mp.UMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            self.t1 = (numpy.zeros((nocca,nvira)), numpy.zeros((noccb,nvirb)))
            return self.e_corr, self.t1, self.t2
        return uccsd.UCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        # _scf.exxdiv affects eris.fock. HF exchange correction should be
        # excluded from the Fock matrix.
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        #if mo_coeff is self._scf.mo_coeff:
        #    idxa, idxb = self.get_frozen_mask()
        #    mo_e_a, mo_e_b = self._scf.mo_energy
        #    eris.mo_energy = (mo_e_a[idxa], mo_e_b[idxb])
        #else:
        nocca, noccb = eris.nocc
        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        eris.mo_energy = (_adjust_occ(eris.mo_energy[0], nocca, -madelung),
                          _adjust_occ(eris.mo_energy[1], noccb, -madelung))
        return eris

class GCCSD(gccsd.GCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(self._scf)
        if mbpt2:
            from pyscf.pbc.mp import mp2
            pt = mp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = numpy.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2
        return gccsd.GCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        with_df = self._scf.with_df
        kpt = self._scf.kpt
        def ao2mofn(mo_coeff):
            nao, nmo = mo_coeff.shape
            mo_a = mo_coeff[:nao//2]
            mo_b = mo_coeff[nao//2:]
            orbspin = getattr(mo_coeff, 'orbspin', None)
            if orbspin is None:
                eri  = with_df.ao2mo(mo_a, kpt, compact=False)
                eri += with_df.ao2mo(mo_b, kpt, compact=False)
                eri1 = with_df.ao2mo((mo_a,mo_a,mo_b,mo_b), kpt, compact=False)
                eri += eri1
                eri += eri1.T
                eri = eri.reshape([nmo]*4)
            else:
                # If GHF orbitals have orbspin labels, alpha and beta orbitals
                # occupy different columns. Here merging them into one set of
                # orbitals then zero out spin forbidden MO integrals
                mo = mo_a + mo_b
                eri  = with_df.ao2mo(mo, kpt, compact=False).reshape([nmo]*4)
                sym_forbid = (orbspin[:,None] != orbspin)
                eri[sym_forbid,:,:] = 0
                eri[:,:,sym_forbid] = 0
            return eri

        # _scf.exxdiv affects eris.fock. HF exchange correction should be
        # excluded from the Fock matrix.
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        #if mo_coeff is self._scf.mo_coeff:
        #    eris.mo_energy = self._scf.mo_energy[self.get_frozen_mask()]
        #else:
        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        return eris

def _adjust_occ(mo_energy, nocc, shift):
    '''Modify occupied orbital energy'''
    mo_energy = mo_energy.copy()
    mo_energy[:nocc] += shift
    return mo_energy


from pyscf.pbc import scf
scf.hf.RHF.CCSD = lib.class_as_method(RCCSD)
scf.uhf.UHF.CCSD = lib.class_as_method(UCCSD)
scf.ghf.GHF.CCSD = lib.class_as_method(GCCSD)
scf.rohf.ROHF.CCSD = None

