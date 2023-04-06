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

'''
Parsers for basis set in Gaussian program format
'''

__all__ = ['parse', 'load']

try:
    from pyscf.gto.basis.parse_nwchem import optimize_contraction
    from pyscf.gto.basis.parse_nwchem import remove_zero
except ImportError:
    optimize_contraction = lambda basis: basis
    remove_zero = lambda basis: basis
from pyscf.lib.exceptions import BasisNotFoundError

MAXL = 12
SPDF = 'SPDFGHIJKLMN'
MAPSPDF = {key: l for l, key in enumerate(SPDF)}

DELIMETER = '****'

def parse(string, optimize=True):
    '''Parse the basis text which is in NWChem format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.
    '''
    raw_basis = []
    for dat in string.splitlines():
        x = dat.split('!', 1)[0].strip()
        if x and x != DELIMETER:
            raw_basis.append(x)
    return _parse(raw_basis, optimize)

def load(basisfile, symb, optimize=True):
    raw_basis = search_seg(basisfile, symb)
    #if not raw_basis:
    #    raise BasisNotFoundError('Basis not found for  %s  in  %s' % (symb, basisfile))
    return _parse(raw_basis, optimize)

def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:

        def _seek(test_str):
            raw_basis = []
            dat = fin.readline()
            while dat:
                if test_str in dat:
                    return True, raw_basis
                elif dat.strip():  # Skip empty lines
                    raw_basis.append(dat)
                dat = fin.readline()
            return False, raw_basis

        has_delimeter, raw_basis = _seek(DELIMETER)
        if has_delimeter:
            dat = fin.readline()
            while dat:
                if dat.strip().split(' ', 1)[0].upper() == symb.upper():
                    raw_basis = _seek(DELIMETER)[1]
                    break
                else:
                    _seek(DELIMETER)
                dat = fin.readline()
    return raw_basis

def _parse(raw_basis, optimize=True):
    basis_add = []
    for line in raw_basis:
        dat = line.strip()
        if dat.startswith('!'):
            continue
        elif dat[0].isalpha():
            key = dat.split()
            if len(key) == 2:
                # skip the line which has only two items. It's the line for
                # element symbol
                continue
            elif key[0] == 'SP':
                basis_add.append([0])
                basis_add.append([1])
            elif len(key[0])>2 and key[0][:2] in ['l=', 'L=']:
                # Angular momentum defined explicitly
                basis_add.append([int(key[0][2:])])
            else:
                basis_add.append([MAPSPDF[key[0]]])
        else:
            line = [float(x) for x in dat.replace('D','e').split()]
            if key[0] == 'SP':
                basis_add[-2].append([line[0], line[1]])
                basis_add[-1].append([line[0], line[2]])
            else:
                basis_add[-1].append(line)
    basis_sorted = []
    for l in range(MAXL):
        basis_sorted.extend([b for b in basis_add if b[0] == l])
    if not basis_sorted:
        raise BasisNotFoundError(f'Basis data not found in "{raw_basis}"')

    if optimize:
        basis_sorted = optimize_contraction(basis_sorted)

    basis_sorted = remove_zero(basis_sorted)
    return basis_sorted

if __name__ == '__main__':
    print(load('def2-qzvp-jkfit.gbs', 'C'))
