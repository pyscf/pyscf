#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import json
import h5py
import pyscf.gto

def load_chkfile_key(chkfile, key):
    return load(chkfile, key)
def load(chkfile, key):
    '''Load array(s) from chkfile
    
    Args:
        chkfile : str
            Name of chkfile. The chkfile needs to be saved in HDF5 format.
        key : str
            HDF5.dataset name or group name.  If key is the HDF5 group name,
            the group will be loaded into an Python dict, recursively

    Returns:
        whatever read from chkfile

    Examples:

    >>> from pyscf import gto, scf, lib
    >>> mol = gto.M(atom='He 0 0 0')
    >>> mf = scf.RHF(mol)
    >>> mf.chkfile = 'He.chk'
    >>> mf.kernel()
    >>> mo_coeff = lib.chkfile.load('He.chk', 'scf/mo_coeff')
    >>> mo_coeff.shape
    (1, 1)
    >>> scfdat = lib.chkfile.load('He.chk', 'scf')
    >>> scfdat.keys()
    ['e_tot', 'mo_occ', 'mo_energy', 'mo_coeff']
    '''
    def load_as_dic(key, group):
        if key in group:
            val = group[key]
        elif key + '__from_list__' in group:
            key = key + '__from_list__'
            val = group[key]
        else:
            return None

        if isinstance(val, h5py.Group):
            if key.endswith('__from_list__'):
                return [load_as_dic(k, val) for k in val]
            else:
                return dict([(k.replace('__from_list__', ''),
                              load_as_dic(k, val)) for k in val])
        else:
            return val.value

    with h5py.File(chkfile, 'r') as fh5:
        return load_as_dic(key, fh5)

def dump_chkfile_key(chkfile, key, value):
    dump(chkfile, key, value)
def save(chkfile, key, value):
    dump(chkfile, key, value)
def dump(chkfile, key, value):
    '''Save array(s) in chkfile
    
    Args:
        chkfile : str
            Name of chkfile.
        key : str

        value : array, vector ... or dict
            If value is a python dict, the key/value of the dict will be saved
            recursively as the HDF5 group/dataset

    Returns:
        No return value

    Examples:

    >>> import h5py
    >>> from pyscf import lib
    >>> ci = {'Ci' : {'op': ('E', 'i'), 'irrep': ('Ag', 'Au')}}
    >>> lib.chkfile.save('symm.chk', 'symm', ci)
    >>> f = h5py.File('symm.chk')
    >>> f.keys()
    ['symm']
    >>> f['symm'].keys()
    ['Ci']
    >>> f['symm/Ci'].keys()
    ['op', 'irrep']
    >>> f['symm/Ci/op']
    <HDF5 dataset "op": shape (2,), type "|S1">
    '''
    def save_as_group(key, value, root):
        if isinstance(value, dict):
            root1 = root.create_group(key)
            for k in value:
                save_as_group(k, value[k], root1)
        elif isinstance(value, (tuple, list)):
            root1 = root.create_group(key + '__from_list__')
            for k, v in enumerate(value):
                save_as_group(str(k), v, root1)
        else:
            try:
                root[key] = value
            except (TypeError, ValueError) as e:
                if not (e.args[0] == "Object dtype dtype('O') has no native HDF5 equivalent" or
                        e.args[0].startswith('could not broadcast input array')):
                    raise e
                root1 = root.create_group(key + '__from_list__')
                for k, v in enumerate(value):
                    save_as_group(str(k), v, root1)

    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile, 'r+') as fh5:
            if key in fh5:
                del(fh5[key])
            elif key + '__from_list__' in fh5:
                del(fh5[key+'__from_list__'])
            save_as_group(key, value, fh5)
    else:
        with h5py.File(chkfile, 'w') as fh5:
            save_as_group(key, value, fh5)


def load_mol(chkfile):
    '''Load Mole object from chkfile.
    The save_mol/load_mol operation can be used a serialization method for Mole object.
    
    Args:
        chkfile : str
            Name of chkfile.

    Returns:
        A (initialized/built) Mole object

    Examples:

    >>> from pyscf import gto, lib
    >>> mol = gto.M(atom='He 0 0 0')
    >>> lib.chkfile.save_mol(mol, 'He.chk')
    >>> lib.chkfile.load_mol('He.chk')
    <pyscf.gto.mole.Mole object at 0x7fdcd94d7f50>
    '''
    try:
        with h5py.File(chkfile, 'r') as fh5:
            mol = pyscf.gto.loads(fh5['mol'].value)
    except:
# Compatibility to the old serialization format
# TODO: remove it in future release
        from numpy import array
        with h5py.File(chkfile, 'r') as fh5:
            mol = pyscf.gto.Mole()
            mol.output = '/dev/null'
            moldic = eval(fh5['mol'].value)
            for key in ('mass', 'grids', 'light_speed'):
                if key in moldic:
                    del(moldic[key])
            mol.build(False, False, **moldic)
    return mol

def save_mol(mol, chkfile):
    '''Save Mole object in chkfile

    Args:
        mol : an instance of :class:`Mole`.

        chkfile : str
            Name of chkfile.

    Returns:
        No return value
    '''
    dump(chkfile, 'mol', mol.dumps())
dump_mol = save_mol

