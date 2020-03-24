#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytical nuclear gradients of CASCI excited state.
'''

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
import inspect

mol = gto.M(
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g'
)
mf = scf.RHF(mol).run()

mc = mcscf.CASSCF(mf, 4, 4).state_average ([0.25, 0.25, 0.25, 0.25])
mc.run()
print (inspect.getmro (mc.Gradients ().__class__))

# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
mc = mcscf.CASSCF(mf, 4, 4).state_average ([0.25, 0.25, 0.25, 0.25]).run ()
print (inspect.getmro (mc.__class__))
print (inspect.getmro (mc.Gradients ().__class__))
g = mc.Gradients().kernel(state=3)
print('Gradients of the 3rd excited state')
print(g)

#
# Use gradients scanner.
#
# Note the returned gradients are based on atomic unit.
#
g_scanner = mc.nuc_grad_method().as_scanner(state=3)
e, g = g_scanner(mol)
print('Gradients of the 3rd excited state')
print(g)

#
# Specify state ID for the gradients of another state.
#
# Unless explicitly specified as an input argument of set_geom_ function,
# set_geom_ function will use the same unit as the one specified in mol.unit.
mol.set_geom_('''O   0.   0.      0.1
                 H   0.  -0.757   0.587
                 H   0.   0.757   0.587''')
e, g = g_scanner(mol, state=2)
print('Gradients of the 2nd excited state')
print(g)

