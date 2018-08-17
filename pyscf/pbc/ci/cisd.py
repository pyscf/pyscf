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
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        cisd.RCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        from pyscf.cc import rccsd
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        return rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class UCISD(ucisd.UCISD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        ucisd.UCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        return ucisd.uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class GCISD(gcisd.GCISD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        gcisd.GCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)

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
        return gcisd.gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)
