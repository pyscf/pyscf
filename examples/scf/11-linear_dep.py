#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf

'''
mol.atom attribute saves the molecular geometry.  Internally, the geomoetry is
saved as

        [(atom_type/nuc_charge, (coordinates:0.,0.,0.)),  ... ]

By default, the unit of coordinates is Angstrom.  To input the geometry in
Bohr unit, one method is to loop over the internal structure and multiply the
coordinates with 0.52917721092 (pyscf.lib.parameters) to convert Bohr to Angstrom.
We might add another attribute for Mole object to indicate the unit in the future.

Note in this example, the diffuse functions in the basis might cause linear
dependency in the AO space.  It can affect the accuracy of the SCF calculation.
See also example  42-remove_linear_dep.py  to remove the linear dependency.
'''

mol = gto.Mole()
mol.verbose = 3
mol.atom = [['H', (0, 0, i*1.0)] for i in range(10)]
mol.basis = 'aug-ccpvdz'
mol.build()

mf = scf.RHF(mol)
print('condition number', numpy.linalg.cond(mf.get_ovlp()))
mf.scf()
