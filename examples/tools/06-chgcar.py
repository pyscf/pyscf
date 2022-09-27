#!/usr/bin/env python

'''
Write orbitals, electron density in VASP CHGCAR format.
'''

import numpy as np
from pyscf.pbc import gto, scf
from pyscf.tools import chgcar

#
# Regular CHGCAR file for crystal cell
#
cell = gto.M(atom='H 0 0 0; H 0 0 1', a=np.eye(3)*3)
mf = scf.RHF(cell).run()

# electron density
chgcar.density(cell, 'cell_h2.CHGCAR', mf.make_rdm1())

# 1st MO
chgcar.orbital(cell, 'cell_h2_mo1.CHGCAR', mf.mo_coeff[:,0])


#
# Extended mode to support molecular system. In this mode, a lattic was
# generated and the molecule was placed in the center of the unit cell.
#
from pyscf import gto, scf
mol = gto.M(atom='H 0 0 0; H 0 0 1')
mf = scf.RHF(mol).run()

# electron density
chgcar.density(mol, 'mole_h2.CHGCAR', mf.make_rdm1())

# 2nd MO
chgcar.orbital(mol, 'mole_h2_mo2.CHGCAR', mf.mo_coeff[:,1])
