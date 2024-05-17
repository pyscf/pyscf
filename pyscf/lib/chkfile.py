#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import json
import h5py

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range

def load(chkfile, key):
    '''Load array(s) from chkfile

    Args:
        chkfile : str
            Name of chkfile. The chkfile needs to be saved in HDF5 format.
        key : str
            HDF5.dataset name or group name.  If key is the name of a HDF5
            group, the group will be loaded into a Python dict, recursively.

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
                return {k.replace('__from_list__', ''): load_as_dic(k, val) for k in val}
        else:
            return val[()]

    with h5py.File(chkfile, 'r') as fh5:
        return load_as_dic(key, fh5)
load_chkfile_key = load

def dump(chkfile, key, value):
    '''Save array(s) in chkfile

    Args:
        chkfile : str
            Name of chkfile.
        key : str
            key to be used in h5py object. It can contain "/" to represent the
            path in the HDF5 storage structure.
        value : array, vector, list ... or dict
            If value is a python dict or list, the key/value of the dict will
            be saved recursively as the HDF5 group/dataset structure.

    Returns:
        No return value

    Examples:

    >>> import h5py
    >>> from pyscf import lib
    >>> ci = {'Ci' : {'op': ('E', 'i'), 'irrep': ('Ag', 'Au')}}
    >>> lib.chkfile.save('symm.chk', 'symm', ci)
    >>> f = h5py.File('symm.chk', 'r')
    >>> f.keys()
    ['symm']
    >>> f['symm'].keys()
    ['Ci']
    >>> f['symm/Ci'].keys()
    ['op', 'irrep']
    >>> f['symm/Ci/op']
    <HDF5 dataset "op": shape (2,), type "|S1">
    '''
    from pyscf.lib import H5FileWrap
    def save_as_group(key, value, root):
        if isinstance(value, dict):
            root1 = root.create_group(key)
            for k in value:
                save_as_group(k, value[k], root1)
        elif isinstance(value, (tuple, list, RANGE_TYPE)):
            root1 = root.create_group(key + '__from_list__')
            for k, v in enumerate(value):
                save_as_group('%06d'%k, v, root1)
        else:
            try:
                root[key] = value
            except (TypeError, ValueError) as e:
                if not (e.args[0] == "Object dtype dtype('O') has no native HDF5 equivalent" or
                        e.args[0].startswith('could not broadcast input array')):
                    raise e
                root1 = root.create_group(key + '__from_list__')
                for k, v in enumerate(value):
                    save_as_group('%06d'%k, v, root1)

    if h5py.is_hdf5(chkfile):
        with H5FileWrap(chkfile, 'r+') as fh5:
            if key in fh5:
                del (fh5[key])
            elif key + '__from_list__' in fh5:
                del (fh5[key+'__from_list__'])
            save_as_group(key, value, fh5)
    else:
        with H5FileWrap(chkfile, 'w') as fh5:
            save_as_group(key, value, fh5)
dump_chkfile_key = save = dump


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
    from numpy import array  # noqa
    from pyscf import gto
    try:
        with h5py.File(chkfile, 'r') as fh5:
            mol = gto.loads(fh5['mol'][()])
    except Exception:
        # Compatibility to the old serialization format
        # TODO: remove it in future release
        with h5py.File(chkfile, 'r') as fh5:
            mol = gto.Mole()
            mol.output = '/dev/null'
            moldic = eval(fh5['mol'][()])
            for key in ('mass', 'grids', 'light_speed'):
                if key in moldic:
                    del (moldic[key])
            mol.build(False, False, **moldic)
    return mol

def save_mol(mol, chkfile):
    '''Save Mole object in chkfile

    Args:
        chkfile str:
            Name of chkfile.

    Returns:
        No return value

    '''
    dump(chkfile, 'mol', mol.dumps())
dump_mol = save_mol
