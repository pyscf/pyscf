#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF analytical nuclear hessian calculation. SCF hessian object can be created
by calling .Hessian() for pyscf-1.6.1 and newer.
'''

from pyscf import gto, scf, hessian

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = '631g')

mf = scf.RHF(mol).run()
# The stucture of h is
# h[Atom_1, Atom_2, Atom_1_XYZ, Atom_1_XYZ]
h = mf.Hessian().kernel()

mf = mol.apply('UKS').x2c().run()
h = mf.Hessian().kernel()
