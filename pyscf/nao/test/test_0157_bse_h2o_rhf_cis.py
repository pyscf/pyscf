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

from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf import gto, scf, tddft
from pyscf.data.nist import HARTREE2EV
from pyscf.nao import bse_iter
from pyscf.nao.m_polariz_inter_ave import polariz_freq_osc_strength

class KnowValues(unittest.TestCase):

  def test_157_bse_h2o_rhf_cis(self):
    """ This  """
    mol = gto.M(verbose=1,atom='O 0 0 0; H 0 0.489 1.074; H 0 0.489 -1.074',basis='cc-pvdz')

    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gto_td = tddft.TDDFT(gto_mf)
    gto_td.nstates = 190
    gto_td.kernel()

    omegas = np.arange(0.0, 2.0, 0.01) + 1j*0.03
    p_ave = -polariz_freq_osc_strength(gto_td.e, gto_td.oscillator_strength(), omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_ave])
    np.savetxt('test_0157_bse_h2o_rhf_cis_pyscf.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0157_bse_h2o_rhf_cis_pyscf.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
    
    nao_td  = bse_iter(mf=gto_mf, gto=mol, verbosity=0, xc_code='CIS')

    polariz = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, polariz])
    np.savetxt('test_0157_bse_h2o_rhf_cis_nao.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0157_bse_h2o_rhf_cis_nao.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=5e-5, rtol=5e-2), 
      msg="{}".format(abs(data_ref-data).sum()/data.size ) )
    
if __name__ == "__main__": unittest.main()
