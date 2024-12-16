#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
parse CP2K format
'''

import re
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.gto.basis import parse_nwchem
from pyscf.gto.basis.parse_nwchem import _search_basis_block
from pyscf import __config__

DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)

MAXL = 8

def parse(string, symb=None, optimize=False):
    '''Parse the basis text which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.

    Examples:

    >>> cell = gto.Cell()
    >>> cell.basis = {'C': pyscf.gto.basis.parse_cp2k.parse("""
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
    if symb is not None:
        raw_data = list(filter(None, re.split(BASIS_SET_DELIMITER, string)))
        line_data = _search_basis_block(raw_data, symb)
        if not line_data:
            raise BasisNotFoundError(f'Basis not found for {symb}')
    else:
        line_data = string.splitlines()

    bastxt = []
    for dat in line_data:
        x = dat.split('#')[0].strip()
        if (x and not x.startswith('END') and not x.startswith('BASIS')):
            bastxt.append(x)
    return _parse(bastxt, optimize)

def load(basisfile, symb, optimize=False):
    return _parse(search_seg(basisfile, symb), optimize)

def _parse(blines, optimize=False):
    blines_iter = iter(blines)
    try:
        header_ln = next(blines_iter)  # noqa: F841
        nsets = int(next(blines_iter))
    except Exception:
        raise BasisNotFoundError('Not basis data')

    basis = []
    try:
        for n in range(nsets):
            comp = [int(p) for p in next(blines_iter).split()]
            lmin, lmax, nexps, ncontractions = comp[1], comp[2], comp[3], comp[4:]
            basis_n = [[l] for l in range(lmin,lmax+1)]
            for nexp in range(nexps):
                line = next(blines_iter)
                dat = line.split()
                try:
                    bfun = [float(x) for x in dat]
                except ValueError:
                    if DISABLE_EVAL:
                        raise ValueError('Failed to parse %s' % line)
                    else:
                        bfun = eval(','.join(dat))

                if len(bfun) != sum(ncontractions) + 1:
                    raise ValueError('Basis data incomplete')

                bfun_iter = iter(bfun)
                exp = next(bfun_iter)
                for i,l in enumerate(range(lmin,lmax+1)):
                    cl = [exp]
                    for c in range(ncontractions[i]):
                        cl.append(next(bfun_iter))
                    basis_n[i].append(cl)
            basis.extend(basis_n)
    except StopIteration:
        raise ValueError('Basis data incomplete')

    basis_sorted = []
    for l in range(MAXL):
        basis_sorted.extend([b for b in basis if b[0] == l])
    if not basis_sorted:
        raise BasisNotFoundError('Basis data not found')

    if optimize:
        basis_sorted = parse_nwchem.optimize_contraction(basis_sorted)

    basis_sorted = parse_nwchem.remove_zero(basis_sorted)
    return basis_sorted

BASIS_SET_DELIMITER = re.compile('# *BASIS SET.*\n')
def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        fdata = re.split(BASIS_SET_DELIMITER, fin.read())
    line_data = _search_basis_block(fdata[1:], symb)
    if not line_data:
        raise BasisNotFoundError(f'Basis for {symb} not found in {basisfile}')
    return [x for x in line_data if x and 'END' not in x]
