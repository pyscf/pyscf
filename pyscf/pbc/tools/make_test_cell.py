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
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

#import unittest
import numpy as np

from pyscf import gto, scf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.ao2mo import eris
import pyscf.pbc.tools
import pyscf.pbc.cc


def make_cell(L, mesh):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    cell.a = L * np.identity(3)

    cell.basis = 'sto-3g'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_n0(L=5, mesh=[9]*3):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    cell.a = L * np.identity(3)
    cell.a[1,0] = 5.0

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade-q2'
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_n1(L=5, mesh=[9]*3):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    cell.a = L * np.identity(3)

    cell.basis = 'sto-3g'
    cell.pseudo = 'gth-pade-q2'
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_n2(L=5, mesh=[9]*3):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['O', (L/2., L/2., L/2.)],
                      ['H', (L/2.-0.689440, L/2.+0.578509, L/2.)],
                      ['H', (L/2.+0.689440, L/2.-0.578509, L/2.)],
        ])
    cell.a = L * np.identity(3)

    cell.basis = 'sto-3g'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_n3(mesh=[9]*3):
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell

def test_cell_n3_diffuse():
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = {'C': [[0, (0.1, 1.0)],
                        [1, (0.1, 1.0)]]}
    cell.pseudo = "gth-pade"

    cell.verbose = 7
    # cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_cu_metallic(mesh=[9]*3):
    """
    Copper unit cell w/ special basis giving non-equal number of occupied orbitals per k-point
    """
    cell = pbcgto.Cell()
    cell.pseudo = 'gth-pade'
    cell.atom='''
    Cu 0.        , 0.        , 0.
    Cu 1.6993361, 1.6993361, 1.6993361
    '''
    cell.a = '''
    0.        , 3.39867219, 3.39867219
    3.39867219, 0.        , 3.39867219
    3.39867219, 3.39867219, 0.
    '''
    cell.basis = { 'Cu': [[0, (0.8, 1.0)],
                          [1, (1.0, 1.0)],
                          [2, (1.2, 1.0)]] }
    cell.unit = 'B'
    cell.mesh = mesh
    cell.verbose = 9
    cell.incore_anyway = True
    cell.output = '/dev/null'
    cell.build()
    return cell
