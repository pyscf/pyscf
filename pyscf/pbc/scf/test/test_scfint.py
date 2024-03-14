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
import numpy as np
from pyscf import lib
from pyscf.pbc.scf import scfint
from pyscf.pbc import gto as pbcgto
import pyscf.pbc.dft as pdft
import pyscf.pbc.scf.hf as phf
import pyscf.pbc.scf.khf as pkhf
from pyscf.pbc.df import fft_jk


def make_cell1(L, n):
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.a = ((L,0,0),(0,L,0),(0,0,L))
    cell.mesh = [n,n,n]

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.build(False, False)
    return cell

def make_cell2(L, n):
    cell = pbcgto.Cell()
    cell.build(False, False,
               unit = 'B',
               verbose = 0,
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]] })
    return cell

numpy.random.seed(1)
k = numpy.random.random(3)

def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, kpt)
    ngrids = len(aoR)
    s = (cell.vol/ngrids) * np.dot(aoR.T.conj(), aoR)
    return s

def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.

    Note: Evaluated in real space using orbital gradients, for improved accuracy.
    '''
    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, kpt, deriv=1)
    ngrids = aoR.shape[1]  # because we requested deriv=1, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
             np.dot(aoR[2].T.conj(), aoR[2]) +
             np.dot(aoR[3].T.conj(), aoR[3]))
    t *= (cell.vol/ngrids)
    return t



class KnowValues(unittest.TestCase):
    def test_olvp(self):
        cell = make_cell1(4, 41)
        s0 = get_ovlp(cell)
        s1 = scfint.get_ovlp(cell)
        self.assertAlmostEqual(numpy.linalg.norm(s0-s1), 0, 8)
        self.assertAlmostEqual(lib.fp(s1), 1.3229918679678208, 10)

        s0 = get_ovlp(cell, kpt=k)
        s1 = scfint.get_ovlp(cell, kpt=k)
        self.assertAlmostEqual(numpy.linalg.norm(s0-s1), 0, 8)

    def test_t(self):
        cell = make_cell1(4, 41)
        t0 = get_t(cell, kpt=k)
        t1 = scfint.get_t(cell, kpt=k)
        self.assertAlmostEqual(numpy.linalg.norm(t0-t1), 0, 8)

if __name__ == '__main__':
    print("Full Tests for scfint")
    unittest.main()
