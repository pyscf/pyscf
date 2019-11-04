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
from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
from pyscf.nao.ao_log import ao_log

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_ao_log_sp2ion(self):
    """ This is for initializing with SIESTA radial orbitals """
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sp2ion = []
    sp2ion.append(siesta_ion_xml(dname+'/H.ion.xml'))
    sp2ion.append(siesta_ion_xml(dname+'/O.ion.xml'))
    ao = ao_log(sp2ion=sp2ion, nr=512, rmin=0.0025)
    self.assertEqual(ao.nr, 512)
    self.assertAlmostEqual(ao.rr[0], 0.0025)
    self.assertAlmostEqual(ao.rr[-1], 11.105004591662)
    self.assertAlmostEqual(ao.pp[-1], 63.271890905445957)
    self.assertAlmostEqual(ao.pp[0], 0.014244003769469984)
    self.assertEqual(len(ao.sp2nmult), 2)
    self.assertEqual(len(ao.sp_mu2j[1]), 5)
    self.assertEqual(ao.sp2charge[0], 1)

  def test_ao_log_gto(self):
    """ This is indeed for initializing with auxiliary basis set"""
    from pyscf.nao import nao
    sv = nao(gto=mol)
    ao = ao_log(gto=mol, nao=sv)
    #print(__name__, dir(ao))
    self.assertEqual(ao.nr, 1024)
    self.assertEqual(ao.jmx, 2)
    for a,b in zip(sv.sp2charge, ao.sp2charge): self.assertEqual(a,b)
    

if __name__ == "__main__": unittest.main()
