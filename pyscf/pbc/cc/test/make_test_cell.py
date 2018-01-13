#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
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
    #cell.atom.extend([['Be', (0.0,0.0,0.0)]])
    cell.a = L * np.identity(3)

    #cell.basis = 'gth-szv'
    cell.basis = 'sto-3g'
    #cell.pseudo = None
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    #cell.verbose = 4
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
    #cell.basis = "gth-dzvp"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = mesh

    cell.output = '/dev/null'
    cell.build()
    return cell
