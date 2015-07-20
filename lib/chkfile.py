#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
import pyscf.gto

def load_chkfile_key(chkfile, key):
    return load(chkfile, key)
def load(chkfile, key):
    def loadasdic(key, group):
        if key in group:
            val = group[key]
            if isinstance(val, h5py.Group):
                return dict([(k, loadasdic(k, val)) for k in val])
            else:
                return val.value
        else:
            return None
    with h5py.File(chkfile, 'r') as fh5:
        return loadasdic(key, fh5)

def dump_chkfile_key(chkfile, key, value):
    dump(chkfile, key, value)
def save(chkfile, key, value):
    dump(chkfile, key, value)
def dump(chkfile, key, value):
    def saveasgroup(key, value, root):
        if isinstance(value, dict):
            root1 = root.create_group(key)
            for k in value:
                saveasgroup(k, value[k], root1)
        else:
            root[key] = value
    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile, 'r+') as fh5:
            if key in fh5:
                del(fh5[key])
            saveasgroup(key, value, fh5)
    else:
        with h5py.File(chkfile, 'w') as fh5:
            saveasgroup(key, value, fh5)


def load_mol(chkfile):
    with h5py.File(chkfile) as fh5:
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        moldic = eval(fh5['mol'].value)
        mol.build(False, False, **moldic)
    return mol

def save_mol(mol, chkfile):
    dump(chkfile, 'mol', format(mol.pack()))


