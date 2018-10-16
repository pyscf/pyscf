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
from pyscf.nao import bse_iter
from pyscf import gto, scf, tddft, dft
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_polariz_inter_ave import polariz_freq_osc_strength
from pyscf.dft.libxc import  xc_type, parse_xc_name, is_hybrid_xc, is_gga, is_lda
class KnowValues(unittest.TestCase):

  def test_170_bse_h2o_rks_b3lyp_pb(self):
    """ This  """
    mol = gto.M(verbose=1,atom='O 0 0 0; H 0 0.489 1.074; H 0 0.489 -1.074',basis='cc-pvdz')
    gto_mf = scf.RKS(mol)
    gto_mf.xc = 'PBE'
    gto_mf.kernel()
    #print(__name__, 'veff')
    veff = gto_mf.get_veff()
    #print(veff.shape)
    #print(veff.sum())
    #gto_td = tddft.TDDFT(gto_mf)
    #gto_td.nstates = 190
    #gto_td.kernel()
    
    #print(__name__, parse_xc_name(gto_mf.xc) )
    #print(__name__, xc_type(gto_mf.xc) )
    #print(__name__, is_hybrid_xc(gto_mf.xc) )
    #print(__name__, is_gga(gto_mf.xc) )
    #print(__name__, is_lda(gto_mf.xc) )
    
    return 
    
    raise RuntimeError('check ')
    
    omegas = np.arange(0.0, 2.0, 0.01) + 1j*0.03
    p_ave = -polariz_freq_osc_strength(gto_td.e, gto_td.oscillator_strength(), omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_ave])
    np.savetxt('test_0170_bse_h2o_rks_pbe_pyscf.txt', data.T, fmt=['%f','%f'])
    #data_ref = np.loadtxt('test_0170_bse_h2o_rks_pbe_pyscf.txt-ref').T
    #self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
    
    nao_td  = bse_iter(mf=gto_mf, gto=mol, verbosity=1)
    
    fxc = nao_td.comp_fxc_pack()
    print(fxc.shape)
    print(__name__, fxc.sum())
    
    polariz = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, polariz])
    np.savetxt('test_0170_bse_h2o_rks_pbe_nao.txt', data.T, fmt=['%f','%f'])
    #data_ref = np.loadtxt('test_0170_bse_h2o_rks_pbe_nao.txt-ref').T
    #self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3), \
    #  msg="{}".format(abs(data_ref-data).sum()/data.size))
    
if __name__ == "__main__": unittest.main()
