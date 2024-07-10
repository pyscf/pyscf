#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Pass initial guess to SCF calculation.

SCF kernel function takes one argument dm (density matrix) as the initial
guess.  If not given, SCF constructs an atomic density superposition for the
initial guess.
'''

import tempfile
from pyscf import gto
from pyscf import scf


mol0 = gto.M(
    atom = '''
        C     0.   0.   0.625
        C     0.   0.  -0.625 ''',
    basis = 'cc-pVDZ',
)

mf = scf.RHF(mol0).run()

#
# Save the density matrix as the initial guess for the next calculation
#
dm_init_guess = mf.make_rdm1()


mol = gto.M(
    atom = '''
        C     0.   0.   0.7
        C     0.   0.  -0.7 ''',
    basis = 'cc-pVDZ',
)

tmp_chkfile = tempfile.NamedTemporaryFile()
chkfile_name = tmp_chkfile.name
mf = scf.RHF(mol)
mf.chkfile = chkfile_name
mf.kernel(dm_init_guess)

# If a numpy array is assigned to the attribute .init_guess, it will be used
# as the initial guess density matrix
mf.init_guess = dm_init_guess
mf.kernel()

#
# Initial guess from Hcore
#
mf = scf.RHF(mol)
mf.init_guess = '1e'
mf.kernel()

#
# This is the default initial guess.  It is the superpostion from atomic
# density.  The atomic density is projected from MINAO basis.
#
mf = scf.RHF(mol)
mf.init_guess = 'minao'
mf.kernel()

#
# Another way to build the atomic density superpostion.  The atomic density is
# generated from occupantion averaged atomic HF calculation.
#
mf = scf.RHF(mol)
mf.init_guess = 'atom'
mf.kernel()

#
# Also a Huckel guess based on on-the-fly atomic HF calculations is possible.
#
mf = scf.RHF(mol)
mf.init_guess = 'huckel'
mf.kernel()
#
# Another variant also exists, where an updated GWH rule is used
#
mf = scf.RHF(mol)
mf.init_guess = 'mod_huckel'
mf.kernel()

#
# Superposition of atomic potentials (SAP) can be used as initial guess for DFT
# methods.
#
mf = mol.RKS().set(xc='b3lyp')
mf.init_guess = 'vsap'
mf.kernel()

#
# Gaussian fitted version of SAP can be used with
# any method.
#
mf = scf.RHF(mol)
mf.init_guess = 'sap'
mf.kernel()
#
# The SAP fit basis can be changed using the SCF object attribute sap_basis.
# sap_basis accepts either python dictionary (basis set in internal
# format) or filename/pathname. If BSE API is installed with pip, 
# the implementation will also look through the BSE catalog for basis sets.
#
mf.sap_basis = 'sapgraspsmall'
mf.kernel() # Will use PySCF SAP_ALIAS and find it
# in the local files
mf.sap_basis = 'sap_helfem_large'
mf.kernel() # Will be found from BSE if installed
mf.sap_basis = {
    'C': [[0,
  [70.00376965910681, -1.461395066555269],
  [35.71620900974838, 2.081985464785248],
  [18.22255561721856, -3.934427793137729],
  [9.297222253682943, 2.482153296237811],
  [4.743480741674971, -2.270305056357756],
  [2.420143235548455, -1.625287881586701],
  [1.234766956912478, -2.814362858422101],
  [0.6299831412818762, -0.9315848803380504],
  [0.3214199700417737, -0.2828736323863268],
  [0.163989780633558, -0.4608879213337786],
  [0.08366825542528468, -0.4659958068514243],
  [0.04268788542106363, -0.3170178640539234]]]
}
mf.kernel() # Will use the above basis for C atoms


#
# Initial guess can be read and projected from another chkfile.
# In this initial guess method, mf.chkfile attribute needs to be specified.
# It is allowed to use the chkfile of different molecule because the density
# matrix is projected to the target molecule.
#
mf = scf.RHF(mol)
mf.chkfile = chkfile_name
mf.init_guess = 'chkfile'
mf.kernel()

#
# The last initial guess method is identical to the following operations.
# mf.from_chk function reads the density matrix from the given chkfile, and
# project it to the basis that defined by mol.
#
mf = scf.RHF(mol)
dm = mf.from_chk(chkfile_name)
mf.kernel(dm)

