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
from pyscf.nao import tddft_iter
from pyscf.nao.m_polariz_inter_ave import polariz_nonin_ave

class KnowValues(unittest.TestCase):

  def test_155_tddft_h2b_uks_nonin(self):
    """ This  example is with discrepancies... """
    mol = gto.M(verbose=1,atom='B 0 0 0; H 0 0.489 1.074; H 0 0.489 -1.074',basis='cc-pvdz',spin=3)

    gto_mf = scf.UKS(mol)
    gto_mf.kernel()

    omegas = np.arange(0.0, 2.0, 0.01) + 1j*0.03
    p_ave = -polariz_nonin_ave(gto_mf, mol, omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_ave])
    np.savetxt('test_0155_tddft_h2b_uks_nonin_pyscf.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0155_tddft_h2b_uks_nonin_pyscf.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
    
    nao_td  = tddft_iter(mf=gto_mf, gto=mol, verbosity=0, xc_code='RPA')

    polariz = -nao_td.polariz_nonin_ave_matelem(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, polariz])
    np.savetxt('test_0155_tddft_h2b_uks_nonin_matelem.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0155_tddft_h2b_uks_nonin_matelem.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))

    polariz = -nao_td.comp_polariz_nonin_ave(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, polariz])
    np.savetxt('test_0155_tddft_h2b_uks_nonin_nao.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0155_tddft_h2b_uks_nonin_nao.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
    
if __name__ == "__main__": unittest.main()
