#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use callback hook to save SCF informations in each iteration.

See also pyscf/examples/mcscf/24-callback.py
'''

import numpy
from pyscf import gto, scf

mol = gto.M(
    atom = [
        ["F", (0., 0., 0.)],
        ["H", (0., 0., 1.6)],],
    basis = 'cc-pvdz')

mf = scf.RHF(mol)
mf.chkfile = 'hf1.chk'

def save_scf_iteration(envs):
    cycle = envs['cycle']
    info = {'fock'    : envs['fock'],
            'dm'      : envs['dm'],
            'mo_coeff': envs['mo_coeff'],
            'mo_energy':envs['mo_energy'],
            'e_tot'   : envs['e_tot']}
    scf.chkfile.save(mf.chkfile, 'HF-iteration/%d' % cycle, info)

mf.callback = save_scf_iteration
mf.kernel()

import h5py
with h5py.File('hf1.chk', 'r') as f:
    print(f['HF-iteration'].keys())
    print(f['HF-iteration/1'].keys())
