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
from pyscf.pbc import mp
from pyscf.cc import rccsd

class RADC(radc.RADC):
    def kernel(self, nroots=1, guess=None, eris=None):

        #transform_integrals = radc_ao2mo.transform_integrals_incore
        eris = self.transform_integrals_incore()
        #eris = self.transform_integrals()
        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, x

        #return radc.RADC.kernel(self,nroots,guess,eris)
        return e_exc, v_exc, spec_fac, x

    def transform_integrals_incore(self,mo_coeff=None):
    #def transform_integrals(self,mo_coeff=None):
        print ("Sam")
        from pyscf.pbc import tools
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)

        occ = self.mo_coeff[:,:self._nocc]
        vir = self.mo_coeff[:,self._nocc:]

        nocc = occ.shape[1]
        nvir = vir.shape[1]
        nmo = self._nmo
    
        eris = lambda:None
        eri1 = ao2mofn(self.mo_coeff).reshape([nmo]*4)
        
        # TODO: check if myadc._scf._eri is not None

        with lib.temporary_env(self._scf, exxdiv=None):
            eris.oooo = eri1[:nocc, :nocc,:nocc,:nocc].copy()
            #eris.ovoo = eri1[:nocc, nocc:,:nocc,:nocc).copy()  # noqa: E501
            #eris.oovv = eri1[:nocc, :nocc, nocc:, nvir).copy()  # noqa: E501
            #eris.ovvo = eri1[:nocc, nocc:, nocc:, nocc).copy()  # noqa: E501
            #eris.ovvv = eri1[:nocc, nocc:, -1).copy()  # noqa: E501
#
#        if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
#            eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
#            eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3))
#            eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)
#
#        log.timer('ADC integral transformation', *cput0)
#        return eris


            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            #with lib.temporary_env(self._scf, exxdiv=None):
            #    eris = rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)
        #exit()
        return eris

from pyscf.pbc import scf
scf.hf.RHF.ADC = lib.class_as_method(RADC)
#scf.uhf.UHF.CCSD = lib.class_as_method(UCCSD)
#scf.ghf.GHF.CCSD = lib.class_as_method(GCCSD)
#scf.rohf.ROHF.CCSD = None
