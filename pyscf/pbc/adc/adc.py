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
from pyscf.adc import radc
from pyscf.adc import radc_ao2mo
from pyscf.adc import uadc
from pyscf.pbc import mp
from pyscf.cc import rccsd

class RADC(radc.RADC):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        #if abs(mf.kpt).max() > 1e-9:
        #    raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)
        radc.RADC.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def kernel(self, nroots=1, guess=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
    
        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)
    
         

#        if self.verbose >= logger.WARN:
#            self.check_sanity()
#        self.dump_flags_gs()
    
#        nmo = self._nmo
#        nao = self.mo_coeff.shape[0]
#        nmo_pair = nmo * (nmo+1) // 2
#        nao_pair = nao * (nao+1) // 2
#        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
#        mem_now = lib.current_memory()[0]
#
#        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
#           if getattr(self, 'with_df', None): 
#               self.with_df = self.with_df
#           else :
#               self.with_df = self._scf.with_df
#
#           def df_transform():
#              return radc_ao2mo.transform_integrals_df(self)
#           self.transform_integrals = df_transform
#        elif (self._scf._eri is None or
#            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
#           def outcore_transform():
#               return radc_ao2mo.transform_integrals_outcore(self)
#           self.transform_integrals = outcore_transform

        #transform_integrals = radc_ao2mo.transform_integrals_incore
        #eris = self.transform_integrals() 

        mo_coeff = self._scf.mo_coeff
        nocc = self._nocc

        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)        

    
        exit()

        self.e_corr, self.t1, self.t2 = radc.compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = radc.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            e_exc, v_exc, spec_fac, x, adc_es = radc.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, x


#def _adjust_occ(mo_energy, nocc, shift):
#    '''Modify occupied orbital energy'''
#    mo_energy = mo_energy.copy()
#    mo_energy[:nocc] += shift
#    return mo_energy


from pyscf.pbc import scf
scf.hf.RHF.ADC = lib.class_as_method(RADC)
#scf.uhf.UHF.CCSD = lib.class_as_method(UCCSD)
#scf.ghf.GHF.CCSD = lib.class_as_method(GCCSD)
#scf.rohf.ROHF.CCSD = None
