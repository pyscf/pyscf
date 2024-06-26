#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF analytical nuclear hessian calculation.
'''

from pyscf import gto, scf, hessian

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = '631g')

mf = mol.RHF().run()
# The structure of h is
# h[Atom_1, Atom_2, Atom_1_XYZ, Atom_1_XYZ]
h = mf.Hessian().kernel()
print(h.shape)

# Use atmlst to specify the atoms to calculate the hessian
atmlst = [0, 1]
err = abs(h[atmlst][:, atmlst] - mf.Hessian().kernel(atmlst=atmlst)).max()
assert err < 1e-6

mf = mol.apply('UKS').x2c().run()
h = mf.Hessian().kernel()
