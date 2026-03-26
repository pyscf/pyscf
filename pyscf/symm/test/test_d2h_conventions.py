#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

# Validate the implementation of axis label conventions by making sure the MOs
# of ethylene get the same, correct irreps regardless of how its geometry is
# entered
import unittest
import numpy as np
from pyscf import gto, scf

def setUpModule():
    global get_inp, ref
    yh = 0.92229064
    zc = 0.66690396
    zh = 1.22952195
    el = ['C',]*2 + ['H',]*4
    x = [0.0,]*6
    y = [0.0,0.0,yh,yh,-yh,-yh]
    z = [zc,-zc,zh,-zh,zh,-zh]
    xyz = np.asarray ([x,y,z])
    def get_inp (perm):
        carts = xyz[perm,:].T.tolist ()
        print (carts)
        return '\n'.join (['{:s} {} {} {}'.format (*([e,] + c))
                           for e,c in zip (el, carts)])
    ref = tuple ([0, 5, 0, 5, 6, 0, 3, 7, 2, 6, 0, 5, 3, 5])

def tearDownModule ():
    global get_inp, ref
    del get_inp, ref

class KnownValues(unittest.TestCase):
    def case_perm (self, perm):
        print (get_inp(perm))
        mol = gto.M (atom=get_inp(perm), basis='sto-3g', verbose=0, output='/dev/null',
                     symmetry=True)
        mf = scf.RHF (mol).run ()
        orbsym = tuple (mf.mo_coeff.orbsym)
        self.assertEqual (orbsym, ref)

    def test_xyz (self): self.case_perm ((0,1,2))
    def test_xzy (self): self.case_perm ((0,2,1))
    def test_zxy (self): self.case_perm ((2,0,1))
    def test_yxz (self): self.case_perm ((1,0,2))
    def test_yzx (self): self.case_perm ((1,2,0))
    def test_zyx (self): self.case_perm ((2,1,0))

if __name__ == "__main__":
    print("Full Tests for D2h axis conventions")
    unittest.main()

