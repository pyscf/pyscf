#!/usr/bin/env python

'''
Initialize a molecular system.

There are many methods to define/initialize a molecule.  This example presents
three methods to create/initialize the molecular object.  Mole object is a
Python object.  You can initialize the Mole object using any methods supported
by Python.

See also

pyscf/examples/pbc/06-load_mol_from_chkfile.py  to initialize mol from chkfile

pyscf/examples/pbc/00-input_cell.py  for initialization of crystal

'''

from pyscf import gto

#
# First method is to assign the geometry, basis etc. to Mole object, then
# call build() function to initialize the molecule
#
mol = gto.Mole()
mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
mol.basis = 'sto-3g'
mol.build()

#
# Shortcuts for initialization.
#
# Use the keyword arguments of mol.build() to initialize a molecule
#
mol = gto.Mole()
mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)

#
# Use shortcut function gto.M or pyscf.M to initialize a molecule
#
mol = gto.M(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)

import pyscf
mol = pyscf.M(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)



#
# Other parameters
# ================
#

mol.charge = 0
mol.spin = 0 # 2j == nelec_alpha - nelec_beta
mol.symmetry = 1  # Allow the program to apply point group symmetry if possible
# .unit can be 'bohr', 'ang' to indicate the coordinates unit of the input mol.atom
# If a number is assigned to unit, this number will be used as the length of
# 1 Bohr (in Angstrom).  Eg you can double the bond length of a system by
# setting mol.unit = 0.529*.5.
mol.unit = 'Ang'    # (New in version 1.1)

# Output
# ------
# To write output on disk, assign a filename to Mole.output
mol.output = 'path/to/my_out.txt'
# if Mole.output is not given, the default output would be stdout

# Print level
# -----------
# Mole.verbose is used to control print level.  The print level can be 0 (quite,
# no output) to 9 (very noise).  The default level is 1, which only outputs the
# error message, it works almost the same as level 0.  Level 4 (info), or 5 (debug)
# are recommended value if some calculation detials are needed.
mol.verbose = 4
# level 4 hides some details such as CPU timings, the orbital energies during
# the SCF iterations.

# max memory to use
# -----------------
mol.max_memory = 1000 # in MB
# or use evnrionment  PYSCF_MAX_MEMORY  to control the memory usage
# (New in PySCF-1.3) eg
#    export PYSCF_MAX_MEMORY=10000 # 10 GB
#    python 00-input_mole.py

# Whether to use Cartesian GTOs (New since version 1.5)
# -----------------------------------------------------
# default: False
mol.cart = True
