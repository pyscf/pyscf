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
import unittest, numpy as np

from pyscf.nao.m_log_mesh import log_mesh
from pyscf.nao.m_log_interp import log_interp_c

class KnowValues(unittest.TestCase):

  def test_log_interp_sca(self):
    """ Test the interpolation facility from the class log_interp_c """
    rr,pp = log_mesh(1024, 0.01, 20.0)
    log_interp = log_interp_c(rr)
    gc = 1.2030
    ff = np.array([np.exp(-gc*r**2) for r in rr])
    for r in np.linspace(0.009, 25.0, 100):
      y = log_interp(ff, r)
      yrefa = np.exp(-gc*r**2)
      self.assertAlmostEqual(y, yrefa)

  def test_log_interp_vec(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = log_mesh(1024, 0.01, 20.0)
    log_interp = log_interp_c(rr)
    gcs = np.array([1.2030, 3.2030, 0.7, 10.0])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])
    for r in np.linspace(0.009, 25.0, 100):
      yyref, yy = np.exp(-gcs*r**2), log_interp(ff, r)
      for y,yref in zip(yy, yyref): self.assertAlmostEqual(y,yref)

  def test_log_interp_diff(self):
    """ Test the differentiation facility from the class log_interp_c """
    import matplotlib.pyplot as plt
    rr,pp = log_mesh(1024, 0.001, 20.0)
    logi = log_interp_c(rr)
    gc = 1.2030
    ff = np.array([np.exp(-gc*r**2) for r in rr])
    ffd_ref = np.array([np.exp(-gc*r**2)*(-2.0*gc*r) for r in rr])
    ffd = logi.diff(ff)
    ffd_np = np.gradient(ff, rr)
    s = 3
    for r,d,dref,dnp in zip(rr[s:],ffd[s:],ffd_ref[s:],ffd_np[s:]):
      self.assertAlmostEqual(d,dref)
      
    #plt.plot(rr, ff, '-', label='ff')
    #plt.plot(rr, ffd, '--', label='ffd')
    #plt.legend()
    #plt.show()
    
if __name__ == "__main__":
  unittest.main()
