#!/usr/bin/env python

'''
Save and load Mole object from/to chkfile
'''

from pyscf import gto
from pyscf import lib

mol = gto.M(atom = '''O 0 0 0; H 0 1 0; H 0 0 1''',
            basis = 'ccpvdz')
print('cc-pvdz basis, nbas = %d' % mol.nbas)

#
# h2o.chk is in HDF5 format.  The mol object is stored in string format under
# key '/mol' 
#
lib.chkfile.save_mol(mol, 'h2o.chk')
mol = lib.chkfile.load_mol('h2o.chk')


#
# mol.update_from_chk read the mol from chkfile and overwrite the mol
# attributes
#
mol = gto.M(atom = '''O 0 0 0; H 0 1 0; H 0 0 1''',
            basis = 'sto3g')
print('mol.basis s STO-3G. nbas = %d' % mol.nbas)

mol.update_from_chk('h2o.chk')
print('After calling update_from_chk, mol.basis becomes cc-pvdz. nbas = %d' % mol.nbas)

