#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.lib.chkfile import load_mol, save_mol

def load_scf(chkfile):
    return load_mol(chkfile), load(chkfile, 'scf')

def dump_scf(mol, chkfile, e_tot, mo_energy, mo_coeff, mo_occ,
             overwrite_mol=True):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile) as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    scf_dic = {'e_tot'    : e_tot,
               'mo_energy': mo_energy,
               'mo_occ'   : mo_occ,
               'mo_coeff' : mo_coeff}
    save(chkfile, 'scf', scf_dic)

