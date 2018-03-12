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

class KnowValues(unittest.TestCase):

  def test_bse_rpa(self):
    """ Compute polarization with RPA via 2-point non-local potentials (BSE solver)  """
    from timeit import default_timer as timer
    from pyscf.nao import system_vars_c, prod_basis_c, bse_iter_c
    from timeit import default_timer as timer

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    pb = prod_basis_c().init_prod_basis_pp(sv)
    bse = bse_iter_c(pb.sv, pb, iter_broadening=1e-2)
    omegas = np.linspace(0.0,2.0,150)+1j*bse.eps
    
    pxx = np.zeros(len(omegas))
    for iw,omega in enumerate(omegas):
      for ixyz in range(1):
        vab = bse.apply_l(bse.dab[ixyz], omega)
        pxx[iw] = pxx[iw] - (vab.imag*bse.dab[ixyz]).sum()
        
    data = np.array([omegas.real*27.2114, pxx])
    #np.savetxt('water.bse_iter_rpa.omega.inter.pxx.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.bse_iter_rpa.omega.inter.pxx.txt-ref')
    #print('    bse.l0_ncalls ', bse.l0_ncalls)
    #self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
