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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import dft
from pyscf.pbc import tools
from pyscf.pbc.x2c import sfx2c1e

cell = gto.Cell()
cell.build(unit = 'B',
           a = numpy.eye(3)*4,
           mesh = [11]*3,
           atom = 'H 0 0 0; H 0 0 1.8',
           verbose = 0,
           basis='sto3g')

class KnownValues(unittest.TestCase):
    def test_hf(self):
        with lib.light_speed(2) as c:
            mf = scf.RHF(cell).sfx2c1e()
            mf.with_df = df.AFTDF(cell)
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.47578184212352159+0j, 8)
            kpts = cell.make_kpts([3,1,1])
            h1 = mf.get_hcore(kpt=kpts[1])
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.09637799091491725+0j, 8)

    def test_khf(self):
        with lib.light_speed(2) as c:
            mf = scf.KRHF(cell).sfx2c1e()
            mf.with_df = df.AFTDF(cell)
            mf.kpts = cell.make_kpts([3,1,1])
            dm = mf.get_init_guess()
            h1 = mf.get_hcore()
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]),-0.47578184212352159+0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]),-0.09637799091491725+0j, 8)
            self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]),-0.09637799091491725+0j, 8)

    def test_pnucp(self):
        cell1 = gto.Cell()
        cell1.atom = '''
        He   1.3    .2       .3
        He    .1    .1      1.1 '''
        cell1.basis = {'He': [[0, [0.8, 1]],
                              [1, [0.6, 1]]
                             ]}
        cell1.mesh = [15]*3
        cell1.a = numpy.array(([2.0,  .9, 0. ],
                               [0.1, 1.9, 0.4],
                               [0.8, 0  , 2.1]))
        cell1.build()

        charge = -cell1.atom_charges()
        Gv = cell1.get_Gv(cell1.mesh)
        SI = cell1.get_SI(Gv)
        rhoG = numpy.dot(charge, SI)

        coulG = tools.get_coulG(cell1, mesh=cell1.mesh, Gv=Gv)
        vneG = rhoG * coulG
        vneR = tools.ifft(vneG, cell1.mesh).real

        coords = cell1.gen_uniform_grids(cell1.mesh)
        aoR = dft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        vne_ref = numpy.einsum('p,xpi,xpj->ij', vneR, aoR[1:4], aoR[1:4])

        mydf = df.AFTDF(cell1)
        dat = sfx2c1e.get_pnucp(mydf)
        self.assertAlmostEqual(abs(dat-vne_ref).max(), 0, 7)

        mydf.eta = 0
        dat = sfx2c1e.get_pnucp(mydf)
        self.assertAlmostEqual(abs(dat-vne_ref).max(), 0, 7)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.x2c")
    unittest.main()
