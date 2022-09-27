#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#         Timothy Berkelbach <tim.berkelbach@gmail.com> 

import os
from pyscf.gto import basis as _mol_basis
from pyscf.gto.basis import parse_ecp, load_ecp
from pyscf.pbc.gto.basis import parse_cp2k
from pyscf import __config__

ALIAS = {
    'gthaugdzvp'  : 'gth-aug-dzvp.dat',
    'gthaugqzv2p' : 'gth-aug-qzv2p.dat',
    'gthaugqzv3p' : 'gth-aug-qzv3p.dat',
    'gthaugtzv2p' : 'gth-aug-tzv2p.dat',
    'gthaugtzvp'  : 'gth-aug-tzvp.dat',
    'gthdzv'      : 'gth-dzv.dat',
    'gthdzvp'     : 'gth-dzvp.dat',
    'gthqzv2p'    : 'gth-qzv2p.dat',
    'gthqzv3p'    : 'gth-qzv3p.dat',
    'gthszv'      : 'gth-szv.dat',
    'gthtzv2p'    : 'gth-tzv2p.dat',
    'gthtzvp'     : 'gth-tzvp.dat',
    'gthccdzvp'   : 'gth-cc-dzvp.dat',
    'gthcctzvp'   : 'gth-cc-tzvp.dat',
    'gthccqzvp'   : 'gth-cc-qzvp.dat',
    'gthszvmolopt'      : 'gth-szv-molopt.dat',
    'gthdzvpmolopt'     : 'gth-dzvp-molopt.dat',
    'gthtzvpmolopt'     : 'gth-tzvp-molopt.dat',
    'gthtzv2pmolopt'    : 'gth-tzv2p-molopt.dat',
    'gthszvmoloptsr'    : 'gth-szv-molopt-sr.dat',
    'gthdzvpmoloptsr'   : 'gth-dzvp-molopt-sr.dat',
}

OPTIMIZE_CONTRACTION = getattr(__config__, 'pbc_gto_basis_parse_optimize', False)
def parse(string, optimize=OPTIMIZE_CONTRACTION):
    '''Parse the basis text in CP2K format, return an internal basis format
    which can be assigned to :attr:`Cell.basis`

    Args:
        string : Blank linke and the lines of "BASIS SET" and "END" will be ignored

    Examples:

    >>> cell = gto.Cell()
    >>> cell.basis = {'C': gto.basis.parse("""
    ... C DZVP-GTH
    ...   2
    ...   2  0  1  4  2  2
    ...         4.3362376436   0.1490797872   0.0000000000  -0.0878123619   0.0000000000
    ...         1.2881838513  -0.0292640031   0.0000000000  -0.2775560300   0.0000000000
    ...         0.4037767149  -0.6882040510   0.0000000000  -0.4712295093   0.0000000000
    ...         0.1187877657  -0.3964426906   1.0000000000  -0.4058039291   1.0000000000
    ...   3  2  2  1  1
    ...         0.5500000000   1.0000000000
    ... #
    ... """)}
    '''
    return parse_cp2k.parse(string)

def load(file_or_basis_name, symb, optimize=OPTIMIZE_CONTRACTION):
    '''Convert the basis of the given symbol to internal format

    Args:
        file_or_basis_name : str
            Case insensitive basis set name. Special characters will be removed.
        symb : str
            Atomic symbol, Special characters will be removed.

    Examples:
        Load DZVP-GTH of carbon 

    >>> cell = gto.Cell()
    >>> cell.basis = {'C': load('gth-dzvp', 'C')}
    '''
    if os.path.isfile(file_or_basis_name):
        try:
            return parse_cp2k.load(file_or_basis_name, symb)
        except RuntimeError:
            with open(file_or_basis_name, 'r') as fin:
                return parse_cp2k.parse(fin.read())

    name = _mol_basis._format_basis_name(file_or_basis_name)
    if '@' in name:
        split_name = name.split('@')
        assert len(split_name) == 2
        name = split_name[0]
        contr_scheme = _mol_basis._convert_contraction(split_name[1])
    else:
        contr_scheme = 'Full'

    if name not in ALIAS:
        return _mol_basis.load(file_or_basis_name, symb)

    basmod = ALIAS[name]
    symb = ''.join(i for i in symb if i.isalpha())
    b = parse_cp2k.load(os.path.join(os.path.dirname(__file__), basmod), symb)

    if contr_scheme != 'Full':
        b = _mol_basis._truncate(b, contr_scheme, symb, split_name)
    return b

del(OPTIMIZE_CONTRACTION)

