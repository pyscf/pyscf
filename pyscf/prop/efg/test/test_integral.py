#!/usr/bin/env python
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

import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.prop.efg import rhf as rhf_efg

class KnownValues(unittest.TestCase):
    def test_sfx2c_field_gradients(self):
        mol = gto.M(atom='H1 0.5 -0.6 0.4; H2 -0.5, 0.4, -0.3; H -0.4 -0.3 0.5; H 0.3 0.5 -0.6',
                    unit='B',
                    basis={'H': [[0,[2., 1]]], 'H1':[[1,[.5, 1]]], 'H2':[[1,[1,1]]]})
        grids = dft.gen_grid.Grids(mol)
        grids.build()
        ao = mol.eval_gto('GTOval_ip', grids.coords)
        r0 = mol.atom_coord(0)
        dr = grids.coords - r0
        dd = numpy.linalg.norm(dr, axis=1)
        rr = 3 * numpy.einsum('ix,iy->ixy', dr, dr)
        for i in range(3):
            rr[:,i,i] -= dd**2
        h1ref = lib.einsum('i,ixy,dip,diq->xypq', grids.weights/dd**5, rr, ao, ao)

        h1ao = rhf_efg._get_sfx2c_quadrupole_integrals(mol, 0)
        self.assertAlmostEqual(abs(h1ref - h1ao).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for efg integrals")
    unittest.main()

