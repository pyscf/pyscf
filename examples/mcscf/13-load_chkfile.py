#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import h5py
from pyscf import gto, scf, mcscf
from pyscf import lib

'''
chkfile is an HDF5 file which store MCSCF and SCF results and intermediates in
key:value format.  Keys in chkfile are the same to the attributes of SCF and
MCSCF objects.
'''

tmpchk = tempfile.NamedTemporaryFile()

mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mf = scf.RHF(mol)
mf.chkfile = tmpchk.name
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 6)
mc.chkfile = tmpchk.name
mc.max_cycle_macro = 1
mc.kernel()


#
# Scenario 1: Using h5py to read quantities in chkfile
#

with h5py.File(tmpchk.name) as f:
    print('Keys in chkfile', f.keys)
    print('Keys in mcscf group', f['mcscf'].keys)
    mcscf_orb = f['mcscf/mo_coeff'].value


#
# Scenario 2: Using lib.chkfile module
#
mol = lib.chkfile.load_mol(tmpchk.name)
mcscf_orb = lib.chkfile.load(tmpchk.name, 'mcscf/mo_coeff')

#
# Scenario 3: Using Python trick to quickly load scf/mcscf
# intermediates/results
#
mc = mcscf.CASSCF(mf, 6, 6)
mc.__dict__.update(lib.chkfile.load(tmpchk.name, 'mcscf'))
