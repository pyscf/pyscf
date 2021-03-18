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
# Superposition of atomic potentials can be used as initial guess for DFT
# methods.
#
mf = mol.RKS().set(xc='b3lyp')
mf.init_guess = 'vsap'
mf.kernel()

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

