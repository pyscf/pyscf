#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save
from pyscf.pbc.lib.chkfile import load_cell, save_cell
from pyscf.scf.chkfile import dump_scf

def load_scf(chkfile):
    return load_cell(chkfile), load(chkfile, 'scf')

