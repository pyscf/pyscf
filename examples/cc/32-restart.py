#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restart a CCSD calculation.

To restore a CCSD calculation, some parameters need to be saved. They are the
SCF orbitals and the CCSD amplitudes.  Note the SCF orbitals may be different
in different runs.  To ensure the restored CCSD amplitudes are meaningful, SCF
orbitals should be saved for the purpose of restarting.
'''

from pyscf import lib, gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.3',
    basis = 'ccpvdz')

mf = scf.RHF(mol)
mf.chkfile = 'hf.chk'  # Save SCF orbitals
mf.kernel()

mycc = cc.CCSD(mf)
mycc.max_cycle = 5
mycc.verbose = 4
# CCSD amplitudes can be restored from DIIS object
mycc.diis_file = 'ccdiis.h5'
mycc.kernel()


# In an independent run, first restore the system and SCF orbitals
mol = lib.chkfile.load_mol('hf.chk')
mf = scf.RHF(mol)
mf.__dict__.update(lib.chkfile.load('hf.chk', 'scf'))

#
# There are two methods to restore the CCSD calculations
#
# Method 1
# Construct the amplitudes based on the DIIS file, and use the amplitudes as
# the initial guess of CCSD calculation
mycc = cc.CCSD(mf)
mycc.verbose = 4
ccvec = lib.diis.restore('ccdiis.h5')
t1, t2 = mycc.vector_to_amplitudes(ccvec)
mycc.kernel(t1, t2)

# Method 2
# Construct the diis object based on the DIIS file, then use the DIIS object
# in the CCSD calculation which will include the informations from previous
# unfinished calculations.
mycc = cc.CCSD(mf)
mycc.verbose = 4
mycc.diis = lib.diis.DIIS(mol)
ccvec = mycc.diis.restore('ccdiis.h5')
t1, t2 = mycc.vector_to_amplitudes(ccvec)
mycc.kernel(t1, t2)
