#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
import pyscf.gto
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def load_scf(chkfile):
    return load_mol(chkfile), load(chkfile, 'scf')

def dump_scf(mol, chkfile, e_tot, mo_energy, mo_coeff, mo_occ,
             overwrite_mol=True):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile) as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
            elif overwrite_mol:
                del(fh5['mol'])
                fh5['mol'] = mol.dumps()
            if 'scf' in fh5:
                del(fh5['scf'])
            fh5['scf/e_tot']     = e_tot
            fh5['scf/mo_energy'] = mo_energy
            fh5['scf/mo_occ'   ] = mo_occ
            fh5['scf/mo_coeff' ] = mo_coeff
    else:
        with h5py.File(chkfile, 'w') as fh5:
            fh5['mol'] = mol.dumps()
            fh5['scf/e_tot']     = e_tot
            fh5['scf/mo_energy'] = mo_energy
            fh5['scf/mo_occ'   ] = mo_occ
            fh5['scf/mo_coeff' ] = mo_coeff

