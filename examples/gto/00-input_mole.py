#!/usr/bin/env python

import numpy
import pyscf.lib
from pyscf import gto

# Initialization
# ==============
# There are two ways to define and build a molecule.  The first is to use the
# keyword arguments of mol.build() to initialize a molecule, as follows
mol = gto.Mole()
mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)
# There is a shortcut for this initialization method
mol = gto.M(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g',
)

# The second way is to assign the geometry, basis etc. to Mole object, then
# call build() function to initialize the molecule
mol = gto.Mole()
mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
mol.basis = 'sto-3g'
mol.build()


# Geometry
# ========
# There are two ways to insert the geometry. The internal format of atom is a
# python list:
# atom = [[atom1, (x, y, z)],
#         [atom2, (x, y, z)],
#         ...
#         [atomN, (x, y, z)]]
# You can input the geometry in this format.  Therefore, you are able to use
# all the possible feature provided by Python to make up the geometry, such as
# import geometry from external module, using Python httplib to download
# molecular geometries from database,  or defining a loop to scan energy surface
mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H',(0, 0, 1)]]
mol.atom.extend([['H', (i, i, i)] for i in range(1,5)])
# The second way is to assign mol.atom a string like
mol.atom = '''
O 0 0 0
H 0 1 0
H 0 0 1;
'''
mol.atom += ';'.join(['H '+(' %f'%i)*3 for i in range(1,5)])
# Z-matrix string is not supported yet.

# symbols and format
# ------------------
# To specify the atoms type, you can use the atomic symbol (case-insensitive),
# or the atomic nuclear charge.
mol.atom = [[8,(0, 0, 0)], ['h',(0, 1, 0)], ['H',(0, 0, 1)]]
# If you want to label an atom to distinguish it from the rest, you can prefix
# or suffix number or special characters 1234567890~!@#$%^&*()_+.?:<>[]{}|
# (execept "," and ";") to an atomic symbol.  With this decoration, you can
# specify different basis,
# mass, or nuclear model for the same type of atoms.
mol.atom = '''8 0 0 0; h:1 0 1 0; H@2 0 0'''
mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
mol.build()
# There is a few format requirements.  For the Python list input, you can
# replace it with tuple, or numpy.ndarray for the coordinate part
mol.atom = ((8,numpy.zeros(3)), ['H',(0, 1, 0)], ['H',[0, 0, 1]])
# The string input requires ";" or "\n" to partition atoms, and " " or "," to
# divide the atomic symbol and the coordinates.  Blank lines will be ignored
mol.atom = '''
O        0,   0, 0             ; 1 0.0 1 0

H@2,0 0 1
'''
# No matter which format or symbol used for the input, function Mole.build()
# will convert mol.atom to the internal format
mol.build()
print(mol.atom)


# Basis
# =====
# There are four ways to assign basis sets.  The first is to write in intenal
# format,  an example of internal format is pyscf/gto/basis/dzp_dunning.py
# basis = {atom_type1:[[angular_momentum
#                       (GTO-exp1, contract-coeff11, contract-coeff12),
#                       (GTO-exp2, contract-coeff21, contract-coeff22),
#                       (GTO-exp3, contract-coeff31, contract-coeff32),
#                       ...],
#                      [angular_momentum
#                       (GTO-exp1, contract-coeff11, contract-coeff12),
#                       ...],
#                      ...],
#          atom_type2:[[angular_momentum, (...),],
#                      ...],
mol.basis = {'H': [[0,
                    (19.2406000, 0.0328280),
                    (2.8992000, 0.2312080),
                    (0.6534000, 0.8172380),],
                   [0,
                    (0.1776000, 1.0000000),],
                   [1,
                    (1.0000000, 1.0000000),]],
            }
# This format is not easy to input.  So two functions are defined in gto.basis
# module to simplify the workload.  One is basis.load
mol.basis = {'H': gto.basis.load('sto3g', 'H')}
# the other is basis.parse to parse a basis string of NWChem format
# (see https://bse.pnl.gov/bse/portal), e.g.
mol.basis = {'O': gto.basis.parse('''
C    S
     71.6168370              0.15432897       
     13.0450960              0.53532814       
      3.5305122              0.44463454       
C    SP
      2.9412494             -0.09996723             0.15591627       
      0.6834831              0.39951283             0.60768372       
      0.2222899              0.70011547             0.39195739       
''')}
# These functions return the internal format.  Things can be more convenient
# by inputing name of the baiss
mol.basis = {'O': 'sto3g', 'H': '6-31g'}
# or specify one kind of basis function universally for all atoms
mol.basis = '6-31g'
# There is no support for default settings, i.e. specifying a default basis
# set for all atoms then change one atom to another set.  But as in python
# environment, it can simply be achieved, e.g.
mol.basis = dict([(a, 'sto-3g') for a in pyscf.lib.parameters.NUC.keys()])
mol.basis['C'] = '6-31g'
# where pyscf.lib.parameters.NUC.keys() returns all atomic symbols.
# However, the functions basis.load and basis.parse are defined for other good
# reason.  Since the basis-name input method only assigns the basis set
# according to the associated atomic symbol,  basis.load and basis.parse can
# assign different basis functions for that atom, e.g. to model BSSE
mol.basis = {'GHOST': gto.basis.load('cc-pvdz', 'O'), 'H': 'sto3g'}
# The package defined a 0-nuclear-charge atom, called "GHOST".  This phantom
# atom can be used to insert basis for BSSE correction

# symbols and format
# ------------------
# Like the requirements of geometry input, you can use atomic symbol
# (case-insensitive), the atomic nuclear charge, as the key of the mol.basis
# dict.  Prefix and suffix of numbers and special characters are allowed.
# If the decorated atomic symbol is appeared in mol.atom but not mol.basis,
# the basis parser will remove all decorations and seek the pure atomic symbol
# in mol.basis dict, e.g.  in the following example, 6-31G basis will be
# assigned to the second H atom, but STO-3G will be used for the third atom.
mol.atom = [[8,(0, 0, 0)], ['h1',(0, 1, 0)], ['H2',(0, 0, 1)]]
mol.basis = {'O': 'sto-3g', 'H': 'sto3g', 'H1': '6-31G'}


# Other parameters
# ================

mol.charge = 0
mol.spin = 0 # 2j == nelec_alpha - nelec_beta
mol.symmetry = 1
# can be 'bohr', 'ang' to indicate the coordinates unit of the input mol.atom
mol.unit = 'Ang'    # (New in version 1.1)

# nuclear model
# -------------
# 0 means point nuclear model, 1 means Gaussian nuclear model.  The nuclear
# model can be set globally
mol.nucmod = 1
# or specified in a dictionary, like mol.basis
mol.nucmod = {'O': 1}
# by default, the point nuclear model is used.
# The Gaussian nuclear model needs the mass of atoms.  It can be given in
mol.mass = {'O': 18}
# The requirements on symbol are similar to mol.basis, you can decorate the
# atomic symbol with numbers, special characters.

# Output
# ------
# To write output on disk, assign a filename to Mole.output
mol.output = 'path/to/my_out.txt'
# if Mole.output is not given, the default output would be stdout

# Print level
# -----------
# Mole.verbose is used to control print level.  The print level can be 0
# (quite, no output) to 9 (very noise).  The default level is 1, which only
# output the error message, it works almost the same as level 0.  Mostly, the
# useful print level is 4 (info), and 5 (debug)
mol.verbose = 4
# level 4 hides some details such as CPU timings, the orbital energies during
# the SCF iterations.

# max memory to use
# -----------------
mol.max_memory = 1000 # MB, default is 4000 MB
# The memory usage is NOT well controlled in many module.

