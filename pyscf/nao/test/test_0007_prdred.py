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
import unittest
from pyscf import gto
import numpy as np

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_prdred(self):
    """  """
    from pyscf.nao import system_vars_c
    from pyscf.nao.m_prod_talman import prod_talman_c
    sv = system_vars_c().init_pyscf_gto(mol)
    self.assertEqual(sv.sp2charge[0], 1)
    pt = prod_talman_c(sv.ao_log)
    jtb,clbdtb,lbdtb=pt.prdred_terms(2,4)
    self.assertEqual(len(jtb), 565)
    self.assertEqual(pt.lbdmx, 14)


  def test_prdred_eval(self):
    from pyscf.nao.m_prod_talman import prod_talman_c
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao as csphar_jt
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao import ao_log_c
    from numpy import sqrt, zeros, array

    import os
    
    dname = os.path.dirname(os.path.abspath(__file__))
    sp2ion = []
    sp2ion.append(siesta_ion_xml(dname+'/O.ion.xml'))

    aos = ao_log_c().init_ao_log_ion(sp2ion)
    jmx = aos.jmx
    pt = prod_talman_c(aos)

    spa,spb=0,0
    rav,rbv,rcv = array([0.0,0.0,-1.0]),array([0.0,0.0,1.0]), zeros(3)
    coord = np.array([0.4, 0.1, 0.22]) # Point at which we will compute the expansion and the original product
    ylma,ylmb,ylmc = csphar_jt(coord-rav, jmx), csphar_jt(coord-rbv, jmx),csphar_jt(coord+rcv, pt.lbdmx)
    rcs = sqrt(sum((coord+rcv)**2))
 
    serr = 0.0
    merr = 0.0
    nval = 0
    for la,phia in zip(aos.sp_mu2j[spa], aos.psi_log[spa]):
      fa = pt.log_interp(phia, sqrt(sum((coord-rav)**2)))
      for lb,phib in zip(aos.sp_mu2j[spb], aos.psi_log[spb]):
        fb = pt.log_interp(phib, sqrt(sum((coord-rbv)**2)))

        jtb,clbdtb,lbdtb=pt.prdred_terms(la,lb)
        jtb,clbdtb,lbdtb,rhotb = pt.prdred_libnao(phib,lb,rbv,phia,la,rav,rcv)
        rhointerp = pt.log_interp(rhotb, rcs)
        
        for ma in range(-la,la+1):
          aovala = ylma[la*(la+1)+ma]*fa
          for mb in range(-lb,lb+1):
            aovalb = ylmb[lb*(lb+1)+mb]*fb

            ffr,m = pt.prdred_further_scalar(la,ma,lb,mb,rcv,jtb,clbdtb,lbdtb,rhointerp)
            prdval = 0.0j
            for j,fr in enumerate(ffr): prdval = prdval + fr*ylmc[j*(j+1)+m]

            derr = abs(aovala*aovalb-prdval)
            nval = nval + 1
            serr = serr + derr
            merr = max(merr, derr)

    self.assertTrue(merr<1.0e-05)
    self.assertTrue(serr/nval<1.0e-06)
    

if __name__ == "__main__":
  unittest.main()
