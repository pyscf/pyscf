# Copyright 2020-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
Interface to spglib
'''
import spglib
from packaging.version import parse as _parse_version
if _parse_version(spglib.__version__) < _parse_version('1.15.1'):
    msg = ('spglib outdated. Install or update with:\n\t pip install -U spglib')
    raise ImportError(msg)

def cell_to_spgcell(cell):
    '''
    Convert PySCF Cell object to spglib cell object
    '''
    a = cell.lattice_vectors()
    atm_pos = cell.get_scaled_positions()
    atm_num = []
    from pyscf.data import elements
    for symbol in cell.elements:
        if 'X-' in symbol or 'GHOST-' in symbol:
            raise NotImplementedError("Ghost atoms are not supported with symmetry.")
        atm_num.append(elements.NUC[symbol])
    for iatm in range(cell.natm):
        symb = cell.atom_symbol(iatm)
        idx = ''.join([i for i in symb if i.isnumeric()])
        if idx.isnumeric():
            atm_num[iatm] += int(idx) * 1000
    spg_cell = (a, atm_pos, atm_num, cell.magmom)
    return spg_cell

get_spacegroup = spglib.get_spacegroup
get_symmetry = spglib.get_symmetry
get_symmetry_dataset = spglib.get_symmetry_dataset
