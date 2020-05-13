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
from pyscf.nao import tddft_iter

class KnowValues(unittest.TestCase):

  def test_tddft_iter_lda(self):
    """ Compute polarization with LDA TDDFT  """
    from timeit import default_timer as timer
    
    dname = os.path.dirname(os.path.abspath(__file__))
    td = tddft_iter(label='water', cd=dname, jcutoff=7, iter_broadening=1e-2, xc_code='LDA,PZ', level=0)
    omegas = np.linspace(0.0,2.0,150)+1j*td.eps
    pxx = -td.comp_polariz_inter_xx(omegas).imag
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('water.tddft_iter_lda.omega.inter.pxx.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.tddft_iter_lda.omega.inter.pxx.txt-ref')
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
