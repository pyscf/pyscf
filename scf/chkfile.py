#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import h5py
from pyscf import gto

def load_chkfile_key(chkfile, key):
    return load(chkfile, key)
def load(chkfile, key):
    fh5 = h5py.File(chkfile, 'r')
    val = fh5[key].value
    fh5.close()
    return val

def dump_chkfile_key(chkfile, key, value):
    dump(chkfile, key, value)
def dump(chkfile, key, value):
    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile)
        if key in fh5:
            del(fh5[key])
    else:
        fh5 = h5py.File(chkfile, 'w')
    fh5[key] = value
    fh5.close()


###########################################
def load_scf(chkfile):
    fh5 = h5py.File(chkfile, 'r')
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    moldic = eval(fh5['mol'].value)
    mol.build(False, False, **moldic)
    scf_rec = {
        'hf_energy': fh5['scf/hf_energy'].value,
        'mo_energy': fh5['scf/mo_energy'].value,
        'mo_occ'   : fh5['scf/mo_occ'   ].value,
        'mo_coeff' : fh5['scf/mo_coeff' ].value,
    }
    fh5.close()
    return mol, scf_rec

def dump_scf(mol, chkfile, hf_energy, mo_energy, mo_occ, mo_coeff):
    '''save temporary results'''
    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile)
        if 'scf' in fh5:
            del(fh5['scf'])
    else:
        fh5 = h5py.File(chkfile, 'w')
    fh5['scf/hf_energy'] = hf_energy
    fh5['scf/mo_energy'] = mo_energy
    fh5['scf/mo_occ'   ] = mo_occ
    fh5['scf/mo_coeff' ] = mo_coeff
    fh5.close()


#################################
def _pickle2hdf5(chkfile, ext='.h5'):
    import cPickle as pickle
    rec = pickle.load(open(chkfile, 'r'))
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.atom     = rec['mol']['atom']
    mol.basis    = rec['mol']['basis']
    mol.etb      = rec['mol']['etb']
    mol.build(False, False)

    fh5 = h5py.File(chkfile+ext, 'w')
    fh5['mol'] = format(mol.pack())
    for k1, v1 in rec.items():
        if k1 not in ['mol']:
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    fh5['/'.join((k1,k2))] = v2
            else:
                fh5[k1] = v1
