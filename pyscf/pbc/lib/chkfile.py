#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import h5py
import pyscf.pbc.gto
import pyscf.lib.chkfile
from pyscf.lib.chkfile import load_chkfile_key, load  # noqa
from pyscf.lib.chkfile import dump_chkfile_key, dump, save  # noqa

def load_cell(chkfile):
    '''Load Cell object from chkfile.

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
    with h5py.File(chkfile, 'r') as fh5:
        try:
            cell = pyscf.pbc.gto.loads(fh5['mol'][()])
        except Exception:
            from numpy import array  # noqa
            celldic = eval(fh5['mol'][()])
            cell = pyscf.pbc.gto.cell.unpack(celldic)
            cell.build(False, False)
    return cell

dump_cell = save_cell = pyscf.lib.chkfile.save_mol
