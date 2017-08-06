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

import ase
import ase.lattice
import ase.dft.kpoints

ANG2BOHR = 1.889725989

def make_cell(L, ngs):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    #cell.atom.extend([['Be', (0.0,0.0,0.0)]])
    cell.a = L * np.identity(3)

    #cell.basis = 'gth-szv'
    cell.basis = 'sto-3g'
    #cell.pseudo = None
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    #cell.verbose = 4
    cell.output = '/dev/null'
    cell.build()
    return cell

def test_cell_n0( L = 5.0, ngs = 4):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    cell.a = L * np.identity(3)
    cell.a[1,0] = 5.0

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    cell.output = '/dev/null'
    cell.build()
    return cell

def test_cell_n1( L = 5.0, ngs = 4):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
    cell.a = L * np.identity(3)

    cell.basis = 'sto-3g'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    cell.output = '/dev/null'
    cell.build()
    return cell

def test_cell_n2( L = 5.0, ngs = 4):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom.extend([['O', (L/2., L/2., L/2.)],
                      ['H', (L/2.-0.689440, L/2.+0.578509, L/2.)],
                      ['H', (L/2.+0.689440, L/2.-0.578509, L/2.)],
        ])
    cell.a = L * np.identity(3)

    cell.basis = 'sto-3g'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    cell.output = '/dev/null'
    cell.build()
    return cell


def test_cell_n3(ngs=4):
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    import ase
    import pyscf.pbc.tools.pyscf_ase as pyscf_ase
    import ase.lattice
    from ase.lattice import bulk
    ase_atom = bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    #cell.basis = "gth-dzvp"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.gs = np.array([ngs,ngs,ngs])

    cell.output = '/dev/null'
    cell.build()
    return cell
