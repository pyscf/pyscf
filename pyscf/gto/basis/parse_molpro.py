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
#

'''
Parses for basis set in the Molpro format
'''

import numpy

try:
    from pyscf.gto.basis.parse_nwchem import optimize_contraction
    from pyscf.gto.basis.parse_nwchem import remove_zero
except ImportError:
    optimize_contraction = lambda basis: basis
    remove_zero = lambda basis: basis

MAXL = 8
MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'K': 7}
COMMENT_KEYWORDS = '!*#'

# parse the basis text which is in Molpro format, return an internal basis
# format which can be assigned to gto.mole.basis
def parse(string, optimize=True):
    bastxt = []
    for x in string.splitlines():
        x = x.strip()
        if x and x[0] not in COMMENT_KEYWORDS:
            bastxt.append(x)
    return _parse(bastxt, optimize)

def load(basisfile, symb, optimize=True):
    return _parse(search_seg(basisfile, symb), optimize)

def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        rawbas = []
        dat = fin.readline()
        while dat:
            if dat[0] in COMMENT_KEYWORDS:
                dat = fin.readline()
                continue
            elif dat[0].isalpha():
                if dat.startswith(symb+' '):
                    rawbas.append(dat.splitlines()[0])
                elif rawbas:
                    return rawbas
                fin.readline()  # line for references
            elif rawbas:
                rawbas.append(dat.splitlines()[0])
            dat = fin.readline()
    raise RuntimeError('Basis not found for  %s  in  %s' % (symb, basisfile))


def _parse(raw_basis, optimize=True):
    # pass 1
    basis_add = []
    for dat in raw_basis:
        dat = dat.upper()
        if dat[0].isalpha():
            if ' ' not in dat:
                # Skip the line of comments
                continue
            status = dat
            val = []
            basis_add.append([status, val])
        else:
            val.append(dat)
    raw_basis = [[k, ' '.join(v)] for k,v in basis_add]

    # pass 2
    basis_add = []
    for status, valstring in raw_basis:
        tmp = status.split(':')
        key = tmp[0].split()
        l = MAPSPDF[key[1].upper()]
        #TODO if key[-1] == 'SV'
        val = tmp[1].split()
        np = int(val[0])
        nc = int(val[1])

        rawd = [float(x) for x in valstring.replace('D','e').split()]
        if nc == 0:
            for e in rawd:
                basis_add.append([l, [e, 1.]])
        else:
            exps = numpy.array(rawd[:np])
            coeff = numpy.zeros((np,nc))
            p1 = np
            for i in range(nc):
                start, end = val[2+i].split('.')
                start, end = int(start), int(end)
                nd = end - start + 1
                p0, p1 = p1, p1 + nd
                coeff[start-1:end,i] = rawd[p0:p1]

            bval = numpy.hstack((exps[:,None], coeff))
            basis_add.append([l] + bval.tolist())

    basis_sorted = []
    for l in range(MAXL):
        basis_sorted.extend([b for b in basis_add if b[0] == l])

    if optimize:
        basis_sorted = optimize_contraction(basis_sorted)

    basis_sorted = remove_zero(basis_sorted)
    return basis_sorted

if __name__ == '__main__':
    #print(search_seg('minao.libmol', 'C'))
    print(load('cc_pvdz.libmol', 'C'))
