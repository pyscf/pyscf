#!/usr/bin/env python

'''
PySCF doesn't have its own input parser.  The input file is a Python program.

Before going throught the rest part, be sure the PySCF path is added in PYTHONPATH.
'''

import pyscf

# mol is an object to hold molecule information.
mol = pyscf.M(
    verbose = 4,
    output = 'out_h2o',
    atom = '''
      o     0    0       0
      h     0    -.757   .587
      h     0    .757    .587''',
    basis = '6-31g',
)
# For more details, see pyscf/gto/mole.py and pyscf/examples/gto

#
# The package follow the convention that each method has its class to hold
# control parameters.  The calculation can be executed by the kernel function.
# Eg, to do Hartree-Fock, (1) create HF object, (2) call kernel function
#
mf = mol.RHF()
print('E=%.15g' % mf.kernel())



#
# The above code can be simplified using stream operations.
# There are three stream functions ".set", ".run", ".apply" to pipe computing
# streams.  Stream operations allows writing multiple computing tasks in one
# line.
#
mf = pyscf.M(
    atom = '''
      O     0    0       0
      h     0    -.757   .587
     1      0    .757    .587''',
    basis = '6-31g'
).RHF().run()
print('E=%.15g' % mf.e_tot)


mp2 = mol.RHF().run().MP2().run()
print('E=%.15g' % mp2.e_tot)
