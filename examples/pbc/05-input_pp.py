#!/usr/bin/env python

'''
Input pseudo potential using functions pbc.gto.pseudo.parse and pbc.gto.pseudo.load

See also
pyscf/pbc/gto/pseudo/GTH_POTENTIALS for the GTH-potential format
pyscf/examples/gto/05-input_ecp.py for quantum chemistry ECP format
'''

from pyscf.pbc import gto

cell = gto.M(atom='''
Si 0 0 0
Si 1 1 1''',
             h = '''3    0    0
                    0    3    0
                    0    0    3''',
             gs = [5,5,5],
             basis = 'gth-szv',  # Goedecker, Teter and Hutter single zeta basis
             pseudo = {'Si': gto.pseudo.parse('''
Si
    2    2
     0.44000000    1    -6.25958674
    2
     0.44465247    2     8.31460936    -2.33277947
                                        3.01160535
     0.50279207    1     2.33241791
''')})


#
# Allow mixing quantum chemistry ECP and crystal PP in the same calculation.
#
cell = gto.M(
    h = '''3    0    0
           0    3    0
           0    0    3''',
    gs = [5,5,5],
    atom = 'Cu 0 0 1; Na 0 1 0',
    basis = {'na': 'lan2ldz', 'Cu': 'gth-szv'},
    ecp = 'lanl2dz',
    pseudo = {'Cu': 'gthbp'})
