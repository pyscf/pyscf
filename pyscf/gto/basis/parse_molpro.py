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
Parses for basis set in the Molpro format
'''

__all__ = ['parse', 'load']

import re
import numpy

try:
    from pyscf.gto.basis.parse_nwchem import optimize_contraction
    from pyscf.gto.basis.parse_nwchem import remove_zero
except ImportError:
    optimize_contraction = lambda basis: basis
    remove_zero = lambda basis: basis

from pyscf import __config__
DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)

MAXL = 12
SPDF = 'SPDFGHIKLMNO'
MAPSPDF = {key: l for l, key in enumerate(SPDF)}
COMMENT_KEYWORDS = '!*#'

# parse the basis text which is in Molpro format, return an internal basis
# format which can be assigned to gto.mole.basis
def parse(string, optimize=True):
    raw_basis = []
    for x in string.splitlines():
        x = x.strip()
        if x and x[0] not in COMMENT_KEYWORDS:
            raw_basis.append(x)
    return _parse(raw_basis, optimize)

def load(basisfile, symb, optimize=True):
    raw_basis = search_seg(basisfile, symb)
    #if not raw_basis:
    #    raise BasisNotFoundError('Basis not found for  %s  in  %s' % (symb, basisfile))
    return _parse(raw_basis, optimize)

def search_seg(basisfile, symb):
    raw_basis = []
    with open(basisfile, 'r') as fin:
        dat = fin.readline()
        while dat:
            if dat[0] in COMMENT_KEYWORDS:
                dat = fin.readline()
                continue
            elif dat[0].isalpha():
                if dat.startswith(symb+' '):
                    raw_basis.append(dat.splitlines()[0])
                elif raw_basis:
                    return raw_basis
                fin.readline()  # line for references
            elif raw_basis:
                raw_basis.append(dat.splitlines()[0])
            dat = fin.readline()
    return raw_basis


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

def parse_ecp(string):
    ecptxt = []
    for x in string.splitlines():
        x = x.strip()
        if x and x[0] not in COMMENT_KEYWORDS:
            ecptxt.append(x)
    return _parse_ecp(ecptxt)

def _parse_ecp(raw_ecp):
    symb, nelec, nshell, nso = re.split(',|;', raw_ecp[0])[1:5]
    nelec = int(nelec)
    nshell = int(nshell)
    nso = int(nso)
    assert len(raw_ecp) == (nshell + nso + 2), "ecp info doesn't match with data"

    def parse_terms(terms):
        r_orders = [[] for i in range(7)]  # up to r^6
        for term in terms:
            line = term.split(',')
            order = int(line[0])
            try:
                coef = [float(x) for x in line[1:]]
            except ValueError:
                if DISABLE_EVAL:
                    raise ValueError('Failed to parse ecp %s' % line)
                else:
                    coef = list(eval(','.join(line[1:])))
            r_orders[order].append(coef)
        return r_orders

    ecp_add = {}
    ul = [x.strip() for x in raw_ecp[1].replace('D', 'e').split(';') if x.strip()]
    assert int(ul[0]) + 1 == len(ul), "UL doesn't match data"
    ecp_add[-1] = parse_terms(ul[1:])

    for i, sf_terms in enumerate(raw_ecp[2:2+nshell]):
        terms = [x.strip() for x in sf_terms.replace('D', 'e').split(';') if x.strip()]
        assert int(terms[0]) + 1 == len(terms), \
                "ECP %s Shell doesn't match data" % SPDF[i]
        ecp_add[i] = parse_terms(terms[1:])

    if nso > 0:
        for i, so_terms in enumerate(raw_ecp[2+nshell:]):
            terms = [x.strip() for x in so_terms.replace('D', 'e').split(';') if x.strip()]
            assert int(terms[0]) + 1 == len(terms), \
                    "ECP-SOC Shell %s doesn't match data" % SPDF[i+1]
            soc_data = parse_terms(terms[1:])
            for order, coefs in enumerate(soc_data):
                if not coefs:
                    continue
                sf_coefs = ecp_add[i+1][order]
                assert ([x[0] for x in sf_coefs] == [x[0] for x in coefs]), \
                        "In ECP Shell %s order %d, SF and SOC do not match" % (SPDF[i+1], order)
                for j, c in enumerate(coefs):
                    sf_coefs[j].append(c[1])

    bsort = []
    for l in range(-1, MAXL):
        if l in ecp_add:
            bsort.append([l, ecp_add[l]])
    return [nelec, bsort]

if __name__ == '__main__':
    #print(search_seg('minao.libmol', 'C'))
    print(load('cc_pvdz.libmol', 'C'))
