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
from pyscf.ci import cisd
from pyscf.ci import ucisd
from pyscf.ci import gcisd
from pyscf.pbc import mp

class RCISD(cisd.RCISD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        cisd.RCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        from pyscf.cc import rccsd
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        if mo_coeff is self._scf.mo_coeff:
            eris.mo_energy = self._scf.mo_energy[self.get_frozen_mask()]
        else:
            madelung = tools.madelung(self._scf.cell, self._scf.kpt)
            eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        return eris

class UCISD(ucisd.UCISD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        ucisd.UCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = ucisd.uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        if mo_coeff is self._scf.mo_coeff:
            idxa, idxb = self.get_frozen_mask()
            mo_e_a, mo_e_b = self._scf.mo_energy
            eris.mo_energy = (mo_e_a[idxa], mo_e_b[idxb])
        else:
            nocca, noccb = eris.nocc
            madelung = tools.madelung(self._scf.cell, self._scf.kpt)
            eris.mo_energy = (_adjust_occ(eris.mo_energy[0], nocca, -madelung),
                              _adjust_occ(eris.mo_energy[1], noccb, -madelung))
        return eris

class GCISD(gcisd.GCISD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        gcisd.GCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
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

        with lib.temporary_env(self._scf, exxdiv=None):
            eris = gcisd.gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

        if mo_coeff is self._scf.mo_coeff:
            eris.mo_energy = self._scf.mo_energy[self.get_frozen_mask()]
        else:
            madelung = tools.madelung(self._scf.cell, self._scf.kpt)
            eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        return eris


from pyscf.pbc import scf
scf.hf.RHF.CISD = lib.class_as_method(RCISD)
scf.uhf.UHF.CISD = lib.class_as_method(UCISD)
scf.ghf.GHF.CISD = lib.class_as_method(GCISD)
scf.rohf.ROHF.CISD = None

