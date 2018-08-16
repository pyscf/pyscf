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

from pyscf import lib
from pyscf.lib import logger

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
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        return rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class UCCSD(uccsd.UCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(self._scf)
        if mbpt2:
            pt = mp.UMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocca, nvira = self.nocc
            nmoa, nmoa = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            self.t1 = (numpy.zeros((nocca,nvira)), numpy.zeros((noccb,nvirb)))
            return self.e_corr, self.t1, self.t2
        return uccsd.UCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        return uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

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
                mo = mo_a + mo_b
                eri  = with_df.ao2mo(mo, kpt, compact=False).reshape([nmo]*4)
                sym_forbid = (orbspin[:,None] != orbspin)
                eri[sym_forbid,:,:] = 0
                eri[:,:,sym_forbid] = 0
            return eri
        return gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

