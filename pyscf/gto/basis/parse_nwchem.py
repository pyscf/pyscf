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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parsers for basis set in the NWChem format
'''

__all__ = ['parse', 'load', 'parse_ecp', 'load_ecp',
           'convert_basis_to_nwchem', 'convert_ecp_to_nwchem',
           'optimize_contraction', 'remove_zero', 'to_general_contraction']

import re
import numpy
import numpy as np
import scipy.linalg
from pyscf.data.elements import _std_symbol
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.gto.basis.parse_nwchem_ecp import DISABLE_EVAL, MAXL, SPDF, MAPSPDF
from pyscf.gto.basis.parse_nwchem_ecp import parse as parse_ecp
from pyscf.gto.basis.parse_nwchem_ecp import load as load_ecp
from pyscf.gto.basis.parse_nwchem_ecp import convert_ecp_to_nwchem
from pyscf import __config__

BASIS_SET_DELIMITER = re.compile('# *BASIS SET.*\n|END\n')

def parse(string, symb=None, optimize=True):
    '''Parse the basis text which is in NWChem format. Return an internal
    basis format which can be assigned to attribute :attr:`Mole.basis`
    Empty lines, or the lines started with #, or the lines of "BASIS SET" and
    "END" will be ignored are ignored.

    Args:
        string : A string in NWChem basis format. Empty links and the lines of
        "BASIS SET" and "END" will be ignored

    Kwargs:
        optimize : Optimize basis contraction.  Convert the segment contracted
            basis to the general contracted basis.

    Examples:

    >>> mol = gto.Mole()
    >>> mol.basis = {'O': gto.basis.parse("""
    ... #BASIS SET: (6s,3p) -> [2s,1p]
    ... C    S
    ...      71.6168370              0.15432897
    ...      13.0450960              0.53532814
    ...       3.5305122              0.44463454
    ... C    SP
    ...       2.9412494             -0.09996723             0.15591627
    ...       0.6834831              0.39951283             0.60768372
    ...       0.2222899              0.70011547             0.39195739
    ... """)}

    >>> gto.basis.parse("""
    ... He    S
    ...      13.6267000              0.1752300
    ...       1.9993500              0.8934830
    ...       0.3829930              0.0000000
    ... He    S
    ...      13.6267000              0.0000000
    ...       1.9993500              0.0000000
    ...       0.3829930              1.0000000
    ... """, optimize=True)
    [[0, [13.6267, 0.17523, 0.0], [1.99935, 0.893483, 0.0], [0.382993, 0.0, 1.0]]]
    '''
    if symb is not None:
        symb = _std_symbol(symb)
        line_data = _search_basis_block(re.split(BASIS_SET_DELIMITER, string), symb)
        if not line_data:
            raise BasisNotFoundError('Basis set not found for %s' % symb)
    else:
        line_data = string.splitlines()

    raw_basis = []
    for dat in line_data:
        dat = dat.split('#')[0].strip()  # Use # to start comments
        dat_upper = dat.upper()
        if (dat and not dat_upper.startswith('END') and not dat_upper.startswith('BASIS')):
            raw_basis.append(dat)
    return _parse(raw_basis, optimize)

def load(basisfile, symb, optimize=True):
    '''Load basis for atom of symb from file'''
    raw_basis = search_seg(basisfile, symb)
    if not raw_basis:
        raise BasisNotFoundError('Basis set not found for %s' % symb)
    return _parse(raw_basis, optimize)

def _parse(raw_basis, optimize=True):
    basis_parsed = [[] for l in range(MAXL)]
    key = None
    for line in raw_basis:
        dat = line.strip()
        if not dat or dat.startswith('#'):
            continue
        elif dat[0].isalpha():
            keys = dat.split()
            if len(keys) == 1:
                key = keys[0].upper()
            else:
                key = keys[1].upper()
            if key == 'SP':
                basis_parsed[0].append([0])
                basis_parsed[1].append([1])
            elif key in MAPSPDF:
                l = MAPSPDF[key]
                current_basis = [l]
                basis_parsed[l].append(current_basis)
            else:
                raise BasisNotFoundError('Not basis data')
        else:
            dat = dat.replace('D','e').split()
            try:
                dat = [float(x) for x in dat]
            except ValueError:
                if DISABLE_EVAL:
                    raise ValueError('Failed to parse %s' % line)
                else:
                    dat = list(eval(','.join(dat)))
            except Exception as e:
                raise BasisNotFoundError('\n' + str(e) +
                                         '\nor the required basis file not existed.')
            if key is None:
                raise BasisNotFoundError('Not basis data')
            elif key == 'SP':
                basis_parsed[0][-1].append([dat[0], dat[1]])
                basis_parsed[1][-1].append([dat[0], dat[2]])
            else:
                current_basis.append(dat)
    basis_sorted = [b for bs in basis_parsed for b in bs]
    if not basis_sorted:
        raise BasisNotFoundError('Basis data not found')

    if optimize:
        basis_sorted = optimize_contraction(basis_sorted)

    basis_sorted = remove_zero(basis_sorted)
    return basis_sorted

def search_seg(basisfile, symb):
    symb = _std_symbol(symb)
    with open(basisfile, 'r') as fin:
        fdata = re.split(BASIS_SET_DELIMITER, fin.read())
    line_data = _search_basis_block(fdata, symb)
    return [x for x in line_data if x and 'END' not in x]

def _search_basis_block(raw_data, symb):
    line_data = []
    for dat in raw_data:
        dat0 = dat.split(None, 1)
        if not dat0:
            continue

        if dat0[0] == symb:
            return dat.splitlines()

        elif dat0[0][0] == '#':
            basis_lines = dat.splitlines()
            for i, line in enumerate(basis_lines):
                # Skip all leading '# xxx' lines and empty lines
                if not line or line.lstrip()[0] == '#':
                    continue
                elif line.split(None, 1)[0] == symb:
                    return basis_lines[i:]
                else:
                    break
    return line_data

def convert_basis_to_nwchem(symb, basis):
    '''Convert the internal basis format to NWChem format string'''
    res = []
    symb = _std_symbol(symb)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('#BASIS SET: (%s) -> [%s]' % (','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    for bas in basis:
        res.append('%-2s    %s' % (symb, SPDF[bas[0]]))
        for dat in bas[1:]:
            res.append(' '.join('%15.9f'%x for x in dat))
    return '\n'.join(res)

def optimize_contraction(basis):
    '''Search the basis segments which have the same exponents then merge them
    to the general contracted sets.

    Note the difference to the function :func:`to_general_contraction`. The
    return value of this function may still have multiple segments for each
    angular moment section.
    '''
    basdic = {}
    for b in basis:
        if isinstance(b[1], int):  # kappa = b[1]
            key = tuple(b[:2])
            ec = numpy.array(b[2:]).T
        else:
            key = tuple(b[:1])
            ec = numpy.array(b[1:]).T
        es = ec[0]
        cs = list(ec[1:])

        if key not in basdic:
            basdic[key] = []

        if basdic[key]:
            for e_cs in basdic[key]:
                if numpy.array_equal(e_cs[0], es):
                    e_cs.extend(cs)
                    break
            else:
                basdic[key].append([es] + cs)
        else:
            basdic[key].append([es] + cs)

    basis = []
    for key in sorted(basdic.keys()):
        l_kappa = list(key)
        for e_cs in basdic[key]:
            b = l_kappa + numpy.array(e_cs).T.tolist()
            basis.append(b)
    return basis

def to_general_contraction(basis):
    '''Segmented contracted basis -> general contracted basis.

    Combine multiple basis segments to one segment for each angular moment
    section.

    Examples:

    >>> gto.contract(gto.uncontract(gto.load('sto3g', 'He')))
    [[0, [6.36242139, 1.0, 0.0, 0.0], [1.158923, 0.0, 1.0, 0.0], [0.31364979, 0.0, 0.0, 1.0]]]
    '''
    basdic = {}
    for b in basis:
        if isinstance(b[1], int):  # kappa = b[1]
            key = tuple(b[:2])
            ec = numpy.array(b[2:])
        else:
            key = tuple(b[:1])
            ec = numpy.array(b[1:])
        if key in basdic:
            basdic[key].append(ec)
        else:
            basdic[key] = [ec]

    basis = []
    for key in sorted(basdic.keys()):
        l_kappa = list(key)

        es = numpy.hstack([ec[:,0] for ec in basdic[key]])
        cs = scipy.linalg.block_diag(*[ec[:,1:] for ec in basdic[key]])

        es, e_idx, rev_idx = numpy.unique(es.round(9), True, True)
        es = es[::-1]  # sort the exponents from large to small
        bcoeff = numpy.zeros((e_idx.size, cs.shape[1]))
        for i, j in enumerate(rev_idx):
            bcoeff[j] += cs[i]
        bcoeff = bcoeff[::-1]
        ec = numpy.hstack((es[:,None], bcoeff))

        basis.append(l_kappa + ec.tolist())

    return basis

def remove_zero(basis):
    '''
    Remove exponents if their contraction coefficients are all zeros.
    '''
    new_basis = []
    for b in basis:
        if isinstance(b[1], int):  # kappa = b[1]
            key = list(b[:2])
            ec = b[2:]
        else:
            key = list(b[:1])
            ec = b[1:]

        new_ec = [e_c for e_c in ec if any(c!=0 for c in e_c[1:])]
        if new_ec:
            new_basis.append(key + new_ec)
    return new_basis

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O', basis='6-31g')
    print(load_ecp('lanl2dz.dat', 'Na'))
    b = load('ano.dat', 'Na')
    print(convert_basis_to_nwchem('Na', b))
    b = load_ecp('lanl2dz.dat', 'Na')
    print(convert_ecp_to_nwchem('Na', b))
