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
import numpy as np

class KnowValues(unittest.TestCase):

  def test_csphar(self):
    """  """
    from pyscf.nao.m_csphar import csphar
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao, talman2world
    
    rvec = np.array([0.1, 0.2, -0.4])
    lmax = 3
    ylm_py = csphar(rvec, lmax)
    ylm_jt = csphar_talman_libnao(rvec, lmax)
    self.assertEqual(len(ylm_py), (lmax+1)**2)
    self.assertEqual(len(ylm_jt), (lmax+1)**2)
    
    self.assertAlmostEqual(ylm_py[1], 0.075393004386513446-0.15078600877302686j)

    rvecs = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.2, -0.4], [5.1, 2.2, -9.4], [0.9, 0.6, -0.2]]
    for rvec in rvecs:
       ylm_py_ref = csphar(rvec, lmax)
       ylm_py = talman2world(csphar_talman_libnao(rvec, lmax))
       for y1,y2 in zip(ylm_py_ref, ylm_py):
         self.assertAlmostEqual(y1,y2)

if __name__ == "__main__":
  unittest.main()
