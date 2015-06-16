#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
import pyscf.gto

def load_chkfile_key(chkfile, key):
    return load(chkfile, key)
def load(chkfile, key):
    fh5 = h5py.File(chkfile, 'r')
    if key in fh5:
        val = fh5[key].value
    else:
        val = None
    fh5.close()
    return val

def dump_chkfile_key(chkfile, key, value):
    dump(chkfile, key, value)
def dump(chkfile, key, value):
    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile, 'r+') as fh5:
            if key in fh5:
                del(fh5[key])
            fh5[key] = value
    else:
        with h5py.File(chkfile, 'w') as fh5:
            fh5[key] = value


###########################################
def load_scf(chkfile):
    with h5py.File(chkfile) as fh5:
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        moldic = eval(fh5['mol'].value)
        mol.build(False, False, **moldic)
        scf_rec = {
            'hf_energy': fh5['scf/hf_energy'].value,
            'mo_energy': fh5['scf/mo_energy'].value,
            'mo_occ'   : fh5['scf/mo_occ'   ].value,
            'mo_coeff' : fh5['scf/mo_coeff' ].value, }
    return mol, scf_rec

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

