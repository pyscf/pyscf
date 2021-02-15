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

class KnowValues(unittest.TestCase):

  def test_bse_iter_nonin(self):
    """ Compute polarization with LDA TDDFT  """
    from timeit import default_timer as timer
    
    dname = os.path.dirname(os.path.abspath(__file__))
    bse = bse_iter(label='water', cd=dname, iter_broadening=1e-2, xc_code='RPA', verbosity=0)
    omegas = np.linspace(0.0,2.0,500)+1j*bse.eps
    
    pxx = np.zeros(len(omegas))
    for iw,omega in enumerate(omegas):
      for ixyz in range(1):
        vab = bse.apply_l0(bse.dip_ab[ixyz], omega)
        pxx[iw] = pxx[iw] - (vab.imag*bse.dip_ab[ixyz].reshape(-1)).sum()
        
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('water.bse_iter_rpa.omega.nonin.pxx.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.bse_iter_rpa.omega.nonin.pxx.txt-ref')
    #print('    bse.l0_ncalls ', bse.l0_ncalls)
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
