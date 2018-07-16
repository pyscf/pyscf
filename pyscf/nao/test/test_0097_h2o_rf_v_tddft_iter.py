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

dname = os.path.dirname(os.path.abspath(__file__))
td = gw(label='water', cd=dname, jcutoff=7, iter_broadening=1e-2)
x  = td.moms1[:,0]

class KnowValues(unittest.TestCase):

  def test_chi0_v_rf0(self):
    """ This is non-interacting polarizability with SIESTA starting point """
    omegas = np.arange(0.0,1.0,0.01)+1j*0.02
    rf0 = -td.rf0(omegas).imag
    pxx_rf = np.einsum('p,wpq,q->w', x, rf0, x)
    data_rf = np.array([omegas.real*27.2114, pxx_rf])
    np.savetxt(dname+'/water.tddft_iter_fm_0097.omega.nonin.pxx.txt', data_rf.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.tddft_iter_fm_0097.omega.nonin.pxx.txt-ref')
    self.assertTrue(np.allclose(data_ref,data_rf.T, rtol=1.0, atol=1e-05))

  def test_tddft_iter_v_rpa_rf(self):
    """ This is interacting polarizability with SIESTA starting point """
    omegas = np.arange(0.0,1.0,0.01)+1j*0.02
    rf = -td.rf(omegas).imag
    pxx_rf = np.einsum('p,wpq,q->w', x, rf, x)
    data_rf = np.array([omegas.real*27.2114, pxx_rf])
    np.savetxt(dname+'/water.tddft_iter_rf_0097.omega.inter.pxx.txt', data_rf.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.tddft_iter_rf_0097.omega.inter.pxx.txt-ref')
    self.assertTrue(np.allclose(data_ref,data_rf.T, rtol=1.0, atol=1e-05))

if __name__ == "__main__": unittest.main()
