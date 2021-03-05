import pyscf

'''There are three different methods to initialize a Mole object.
1. Using the function pyscf.M or gto.M:

'''
mol_HF_ccpvdz = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True
)

'''2. First creating a gto.Mole object, then assigning a value to each
of the attributes (see pyscf/gto/mole.py file for the details of
attributes), then calling mol.build() to initialize the mol object.

'''
mol_H2O_ccpvdz = gto.Mole()
mol_H2O_ccpvdz.verbose = 5
mol_H2O_ccpvdz.atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''
mol_H2O_ccpvdz.basis = 'ccpvdz'
mol_H2O_ccpvdz.symmetry = 1
mol_H2O_ccpvdz.build()

''' 3. First creating a Mole object, then calling .build() function
with keyword arguments e.g.

   >>> mol = gto.Mole()
   >>> mol.build(atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587', basis='ccpvdz', symmetry=1)

mol.build() should be called to update the Mole object whenever Mole's
attributes are changed in your script. Note to mute the noise produced by
mol.build function, you can execute mol.build(0,0) instead.
'''
