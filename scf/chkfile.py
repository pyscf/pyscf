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
    with h5py.File(chkfile) as fh5:
        scf_rec = {
            'hf_energy': fh5['scf/hf_energy'].value,
            'mo_energy': fh5['scf/mo_energy'].value,
            'mo_occ'   : fh5['scf/mo_occ'   ].value,
            'mo_coeff' : fh5['scf/mo_coeff' ].value, }
    return load_mol(chkfile), scf_rec

def dump_scf(mol, chkfile, hf_energy, mo_energy, mo_coeff, mo_occ):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile) as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = format(mol.pack())
            if 'scf' in fh5:
                del(fh5['scf'])
            fh5['scf/hf_energy'] = hf_energy
            fh5['scf/mo_energy'] = mo_energy
            fh5['scf/mo_occ'   ] = mo_occ
            fh5['scf/mo_coeff' ] = mo_coeff
    else:
        with h5py.File(chkfile, 'w') as fh5:
            fh5['mol'] = format(mol.pack())
            fh5['scf/hf_energy'] = hf_energy
            fh5['scf/mo_energy'] = mo_energy
            fh5['scf/mo_occ'   ] = mo_occ
            fh5['scf/mo_coeff' ] = mo_coeff

