#!/usr/bin/env python

'''
Low dimensional PBC systems

About the all-electron AFTDF, DF and MDF:

For all-electron systems, AFTDF (analytical Fourier transformation) is
less accurate than DF (density fitting) and MDF (mixed density fitting)
methods.

DF uses one center Gaussian functions to expand the orbital pair products.
For all-electron problem (regardless of the boundary conditions) DF is more
accurate than AFTDF and less accurate than MDF, in most scenario.

MDF is a combination of DF and AFTDF. It uses one center Gaussian with the
planewaves to expand the orbital pair products.  Typically, it has better
accuracy but worse performance than DF.  If the auxiliary Gaussians in DF
have good quality, the DF scheme may have better accuracy than MDF due to
the linear dependency between Gaussian and planewaves.  So choose DF or MDF
based on your needs.

You may see warnings for vacuum size in the 2D calculations. When evaluating
integrals, truncated Coulomb interactions were used by default for 2D system.
This requires a proper vacuum to remove the Coulomb interactions between
slabs. Large errors may be observed if the vacuum size is too small.
See also the discussions in issue #558
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
           a = numpy.eye(3),
           mesh = [20]*3,
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
# The above two lines of initialization can be replaced by density_fit method
# mf = pbchf.RHF(cell).density_fit(auxbasis='weigend')
e.append(mf.kernel())

mf = pbchf.RHF(cell)
mf.with_df = pdf.MDF(cell)
mf.with_df.auxbasis = 'weigend'
# The above two lines of initialization can be replaced by density_fit method
# mf = pbchf.RHF(cell).mix_density_fit(auxbasis='weigend')
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
           a = [[L,0,0],[0,1,0],[0,0,1]],
           mesh = [10,20,20],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=1,
           low_dim_ft_type='inf_vacuum', # This setting must be specified for 1D system
           verbose = 0,
           basis='sto3g')
mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts([4,1,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell).density_fit(auxbasis='weigend')
mf.kpts = cell.make_kpts([4,1,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell).mix_density_fit(auxbasis='weigend')
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
           a = [[L,0,0],[0,L,0],[0,0,8.]],
           mesh = [10,10,20],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=2,
           verbose = 0,
           basis='sto3g')

mf = pbchf.KRHF(cell.copy())
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell, cell.make_kpts([4,4,1])).density_fit(auxbasis='weigend')
e.append(mf.kernel())

mf = pbchf.KRHF(cell, cell.make_kpts([4,4,1])).mix_density_fit(auxbasis='weigend')
e.append(mf.kernel())

mol = tools.super_cell(cell, [4,4,1]).to_mol()
mf = scf.RHF(mol)
e.append(mf.kernel()/16)

print('2D:  FFT      AFT      DF       MDF       super-mole')
print(e)

