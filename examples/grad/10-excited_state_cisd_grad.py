#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytical nuclear gradients of CISD excited state.
'''

from pyscf import gto
from pyscf import scf
from pyscf import ci

mol = gto.M(
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g'
)
mf = scf.RHF(mol).run()

myci = ci.CISD(mf)
myci.nroots = 4
myci.run()

g = myci.nuc_grad_method().kernel(state=3)
print('Gradients of the 3rd excited state')
print(g)

# An equivalent way to specify the exicited state is to directly input the
# excited state wavefunction
g = myci.nuc_grad_method().kernel(myci.ci[3])
print('Gradients of the 3rd excited state')
print(g)


#
# Use gradients scanner.
#
g_scanner = myci.nuc_grad_method().as_scanner(state=3)
e, g = g_scanner(mol)
print('Gradients of the 3rd excited state')
print(g)

#
# Specify state ID for the gradients of another state.
#
mol.set_geom_('''O   0.   0.      0.1
                 H   0.  -0.757   0.587
                 H   0.   0.757   0.587''')
e, g = g_scanner(mol, state=2)
print('Gradients of the 2nd excited state')
print(g)

