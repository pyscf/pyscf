#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
from pyscf.gto.basis import parse_nwchem
from pyscf import __config__

DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)

MAXL = 8

def parse(string, optimize=False):
    '''Parse the basis text which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.
    '''
    bastxt = []
    for dat in string.splitlines():
        x = dat.split('#')[0].strip()
        if (x and not x.startswith('END') and not x.startswith('BASIS')):
            bastxt.append(x)
    return _parse(bastxt, optimize)

def load(basisfile, symb, optimize=False):
    return _parse(search_seg(basisfile, symb), optimize)

def _parse(blines, optimize=False):
    header_ln = blines.pop(0)  # noqa: F841
    nsets = int(blines.pop(0))
    basis = []
    for n in range(nsets):
        comp = [int(p) for p in blines.pop(0).split()]
        lmin, lmax, nexps, ncontractions = comp[1], comp[2], comp[3], comp[4:]
        basis_n = [[l] for l in range(lmin,lmax+1)]
        for nexp in range(nexps):
            line = blines.pop(0)
            dat = line.split()
            try:
                bfun = [float(x) for x in dat]
            except ValueError:
                if DISABLE_EVAL:
                    raise ValueError('Failed to parse basis %s' % line)
                else:
                    bfun = eval(','.join(dat))
            exp = bfun.pop(0)
            for i,l in enumerate(range(lmin,lmax+1)):
                cl = [exp]
                for c in range(ncontractions[i]):
                    cl.append(bfun.pop(0))
                basis_n[i].append(tuple(cl))
        basis.extend(basis_n)
    basis_sorted = []
    for l in range(MAXL):
        basis_sorted.extend([b for b in basis if b[0] == l])

    if optimize:
        basis_sorted = parse_nwchem.optimize_contraction(basis_sorted)

    basis_sorted = parse_nwchem.remove_zero(basis_sorted)
    return basis_sorted

BASIS_SET_DELIMITER = re.compile('# *BASIS SET.*\n')
def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        fdata = re.split(BASIS_SET_DELIMITER, fin.read())
    for dat in fdata[1:]:
        dat0 = dat.split(None, 1)
        if dat0 and dat0[0] == symb:
            # remove blank lines
            return [x.strip() for x in dat.splitlines()
                    if x.strip() and 'END' not in x]
    raise RuntimeError('Basis not found for  %s  in  %s' % (symb, basisfile))


