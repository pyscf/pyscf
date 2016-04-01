#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Another input style.

There are three different methods to initialize a Mole object.
1. To use gto.M function as shown by 00-simple_hf.py.
2. As this script did, first create gto.Mole object, assign value to each
   attributes (see pyscf/gto/mole.py file for the details of attributes), then
   call mol.build() function to initialize the mol object.
3. First create Mole object, then call .build() function with keyword arguments
   eg

   >>> mol = gto.Mole()
   >>> mol.build(atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587', basis='ccpvdz', symmetry=1)

Method 2 and 3 can also be used to update the Mole object whenever the Mole
attributes are changed in your script. (Note to switch off the noise produced
by mol.build function, you can call mol.build(0,0))
'''

from pyscf import gto, scf

mol = gto.Mole()
mol.verbose = 5
#mol.output = 'out_h2o'
mol.atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''
mol.basis = 'ccpvdz'
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.kernel()
