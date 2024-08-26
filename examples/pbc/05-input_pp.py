#!/usr/bin/env python

'''
Input pseudo potential using functions pbc.gto.pseudo.parse and pbc.gto.pseudo.load

It is allowed to mix the Quantum chemistry effective core potential (ECP) with
crystal pseudo potential (PP).  Input ECP with .ecp attribute and PP with
.pseudo attribute.

See also
pyscf/pbc/gto/pseudo/GTH_POTENTIALS for the GTH-potential format
pyscf/examples/gto/05-input_ecp.py for quantum chemistry ECP format
'''

import numpy
from pyscf.pbc import gto

cell = gto.M(atom='''
Si1 0 0 0
Si2 1 1 1''',
             a = '''3    0    0
                    0    3    0
                    0    0    3''',
             basis = {'Si1': 'gth-szv',  # Goedecker, Teter and Hutter single zeta basis
                      'Si2': 'lanl2dz'},
             pseudo = {'Si1': gto.pseudo.parse('''
Si
    2    2
     0.44000000    1    -6.25958674
    2
     0.44465247    2     8.31460936    -2.33277947
                                        3.01160535
     0.50279207    1     2.33241791
''')},
             ecp = {'Si2': 'lanl2dz'},  # ECP for second Si atom
            )

#
# Some elements have multiple PP definitions in GTH database.  Add suffix in
# the basis name to load the specific PP.
#
cell = gto.M(
    a = numpy.eye(3)*5,
    atom = 'Mg1 0 0 0; Mg2 0 0 1',
    pseudo = {'Mg1': 'gth-lda-q2', 'Mg2': 'gth-lda-q10'})

#
# Allow mixing quantum chemistry ECP (or BFD PP) and crystal PP in the same calculation.
#
cell = gto.M(
    a = '''4    0    0
           0    4    0
           0    0    4''',
    atom = 'Cl 0 0 1; Na 0 1 0',
    basis = {'na': 'gth-szv', 'Cl': 'bfd-vdz'},
    ecp = {'Cl': 'bfd-pp'},
    pseudo = {'Na': 'gthbp'})

#
# ECP can be input in the attribute .pseudo
#
cell = gto.M(
    a = '''4    0    0
           0    4    0
           0    0    4''',
    atom = 'Cl 0 0 1; Na 0 1 0',
    basis = {'na': 'gth-szv', 'Cl': 'bfd-vdz'},
    pseudo = {'Na': 'gthbp', 'Cl': 'bfd-pp'})

