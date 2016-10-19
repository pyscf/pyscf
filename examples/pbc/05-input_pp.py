#!/usr/bin/env python

'''
Input pseudo potential using functions pbc.gto.pseudo.parse and pbc.gto.pseudo.load

It is allowed to mix the Quantum chemistry effective core potentail (ECP) with
crystal pseudo potential (PP).  Input ECP with .ecp attribute and PP with
.pseudo attribute.

See also pyscf/pbc/gto/pseudo/GTH_POTENTIALS for the GTH-potential format
'''

from pyscf.pbc import gto

cell = gto.M(atom='''
Si1 0 0 0
Si2 1 1 1''',
             h = '''3    0    0
                    0    3    0
                    0    0    3''',
             gs = [5,5,5],
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

