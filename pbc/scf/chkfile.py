#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import h5py
import pyscf.pbc.gto
import pyscf.lib.chkfile
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save

from numpy import array  # for eval() function

def load_scf(chkfile):
    return load_cell(chkfile), load(chkfile, 'scf')

def dump_scf(cell, chkfile, hf_energy, mo_energy, mo_coeff, mo_occ):
    return pyscf.lib.chkfile.dump_scf(cell, chkfile, hf_energy,
                                      mo_energy, mo_coeff, mo_occ)

def load_cell(chkfile):
    '''Load Cell object from chkfile.
    The save_cell/load_cell operation can be used a serialization method for Cell object.
    
    Args:
        chkfile : str
            Name of chkfile.

    Returns:
        An (initialized/built) Cell object

    Examples:

    >>> from pyscf.pbc import gto, scf
    >>> cell = gto.Cell()
    >>> cell.build(atom='He 0 0 0')
    >>> scf.chkfile.save_cell(cell, 'He.chk')
    >>> scf.chkfile.load_cell('He.chk')
    <pyscf.pbc.gto.cell.Cell object at 0x7fdcd94d7f50>
    '''
    with h5py.File(chkfile) as fh5:
        celldic = eval(fh5['mol'].value)
        cell = pyscf.pbc.gto.cell.unpack(celldic)
        cell.build(False, False)
    return cell

def save_cell(cell, chkfile):
    '''Save Cell object in chkfile under key "mol"
    
    Args:
        cell : an instance of :class:`Cell`.

        chkfile : str
            Name of chkfile.

    Returns:
        No return value
    '''
    dump(chkfile, 'mol', format(cell.pack()))
