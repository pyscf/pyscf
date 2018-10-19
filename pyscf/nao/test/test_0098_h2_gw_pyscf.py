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
from pyscf.nao import gw
from pyscf import gto, tddft, scf
from pyscf.gw import GW
from pyscf.nao.m_polariz_inter_ave import polariz_inter_xx

dname = os.path.dirname(os.path.abspath(__file__))

class KnowValues(unittest.TestCase):

  def test_nao_in_gw_pyscf(self):
    """ This is interacting polarizability with SIESTA starting point """
    from pyscf import gto, tddft, scf
    
    mol = gto.M( verbose = 0,
      atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)

    gto_hf = scf.RKS(mol)
    gto_hf.xc = 'hf'
    gto_hf.kernel()
    nocc = mol.nelectron//2
    nmo = gto_hf.mo_energy.size
    nvir = nmo-nocc
    gto_td = tddft.dRPA(gto_hf)
    gto_td.nstates = min(100, nocc*nvir)
    gto_td.kernel()

    gto_gw = GW(gto_hf, gto_td)
    gto_gw.kernel()

    ww = np.arange(0.0,4.0,0.01)+1j*0.02
    pxx_xpy = -polariz_inter_xx(gto_hf, mol, gto_td, ww).imag
    data = np.array([ww.real*27.2114, pxx_xpy])
    fname = dname+'/h2_rpa_xpy_0098.omega.inter.pxx.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(fname+'-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=0.1, atol=1e-05))
 
    gw_nao = gw(gto=mol, mf=gto_hf, tdscf=gto_td)
    x  = gw_nao.moms1[:,0]
    
    rf_nao = gw_nao.rf( ww )
    pxx_nao = np.einsum('p,wpq,q->w', x, -rf_nao.imag, x)
    data = np.array([ww.real*27.2114, pxx_nao])
    fname = dname+'/h2_rpa_nao_0098.omega.inter.pxx.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(fname+'-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=0.1, atol=1e-05))

    rf_gto = gw_nao.rf_pyscf( ww )
    pxx_gto = np.einsum('p,wpq,q->w', x, -rf_gto.imag, x)
    data = np.array([ww.real*27.2114, pxx_gto])
    fname = dname+'/h2_rpa_gto_0098.omega.inter.pxx.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(fname+'-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=0.1, atol=1e-05))

if __name__ == "__main__": unittest.main()
