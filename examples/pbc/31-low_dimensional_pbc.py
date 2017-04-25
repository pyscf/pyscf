#!/usr/bin/env python

'''
Low dimensional PBC systems
'''

import numpy
from pyscf import scf
from pyscf.pbc import df as pdf
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools

##################################################
#
# 0D PBC (molecule)
#
##################################################

e = []
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = numpy.eye(3)*20,
           gs = [10]*3,
           atom = '''H 0 0 0; H 0 0 1.8''',
           dimension = 0,
           verbose = 0,
           basis = 'sto3g')
mf = pbchf.RHF(cell)
mf.with_df = pdf.AFTDF(cell)
e.append(mf.kernel())

mf = pbchf.RHF(cell)
mf.with_df = pdf.DF(cell)
mf.with_df.auxbasis = 'weigend'
e.append(mf.kernel())

mf = pbchf.RHF(cell)
mf.with_df = pdf.MDF(cell)
mf.with_df.auxbasis = 'weigend'
e.append(mf.kernel())

mol = cell.to_mol()
mf = scf.RHF(mol)
e.append(mf.kernel())

print('0D:  AFT      DF       MDF       super-mole')
print(e)

##################################################
#
# 1D PBC
#
##################################################

e = []
L = 4
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[L,0,0],[0,20,0],[0,0,20]],
           gs = [5,10,10],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=1,
           verbose = 0,
           basis='sto3g')
mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts([4,1,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell)
mf.with_df = pdf.DF(cell)
mf.with_df.auxbasis = 'weigend'
mf.kpts = cell.make_kpts([4,1,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell)
mf.with_df = pdf.MDF(cell)
mf.with_df.auxbasis = 'weigend'
mf.kpts = cell.make_kpts([4,1,1])
e.append(mf.kernel())

mol = tools.super_cell(cell, [4,1,1]).to_mol()
mf = scf.RHF(mol)
e.append(mf.kernel()/4)

print('1D:  AFT      DF       MDF       super-mole')
print(e)

##################################################
#
# 2D PBC
#
##################################################

e = []
L = 4
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[L,0,0],[0,L,0],[0,0,20]],
           gs = [5,5,10],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=2,
           verbose = 0,
           basis='sto3g')
mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell)
mf.with_df = pdf.DF(cell)
mf.with_df.auxbasis = 'weigend'
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell)
mf.with_df = pdf.MDF(cell)
mf.with_df.auxbasis = 'weigend'
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mol = tools.super_cell(cell, [4,4,1]).to_mol()
mf = scf.RHF(mol)
e.append(mf.kernel()/16)

print('2D:  AFT      DF       MDF       super-mole')
print(e)

