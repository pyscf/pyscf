#!/usr/bin/env python

'''
Write orbitals, electron density, molecular electrostatic potential in
Gaussian cube file format.
'''

from pyscf import gto, scf
from pyscf.tools import cubegen

mol = gto.M(atom='''
            O 0.0000000, 0.000000, 0.00000000
            H 0.761561 , 0.478993, 0.00000000
            H -0.761561, 0.478993, 0.00000000''', basis='6-31g*')
mf = scf.RHF(mol).run()

# electron density
cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())

# electron density slice
cubegen.density(mol, 'h2o_den_slice.cube', mf.make_rdm1(), nx=1, ny=1, nz=80)

# molecular electrostatic potential
cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

# molecular electrostatic potential slice
cubegen.mep(mol, 'h2o_pot_slice.cube', mf.make_rdm1(), nx=1, ny=1, nz=80)

# 1st MO
cubegen.orbital(mol, 'h2o_mo1.cube', mf.mo_coeff[:,0])

# 1st MO orbital slice
cubegen.orbital(mol, 'h2o_mo1_slice.cube', mf.mo_coeff[:,0], nx=1, ny=1, nz=80)
