#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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

__all__ = ['parse', 'load', 'convert_ecp_to_nwchem']

import numpy
import numpy as np
import re
from pyscf.data.elements import _std_symbol
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf import __config__

DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)

MAXL = 15
SPDF = 'SPDFGHIKLMNORTU'
MAPSPDF = {key: l for l, key in enumerate(SPDF)}

ECP_DELIMITER = re.compile('\n *ECP *\n')

def parse(string, symb=None):
    '''Parse ECP text which is in NWChem format. Return an internal
    basis format which can be assigned to attribute :attr:`Mole.ecp`
    Empty lines, or the lines started with #, or the lines of "BASIS SET" and
    "END" will be ignored are ignored.
    '''

    if symb is not None:
        symb = _std_symbol(symb)
        raw_data = string.splitlines()
        for i, dat in enumerate(raw_data):
            dat0 = dat.split(None, 1)
            if dat0 and dat0[0] == symb:
                break
        if i+1 == len(raw_data):
            raise BasisNotFoundError('ECP not found for  %s' % symb)
        seg = []
        for dat in raw_data[i:]:
            dat = dat.strip()
            if dat: # remove empty lines
                if ((dat[0].isalpha() and dat.split(None, 1)[0].upper() != symb.upper())):
                    break
                else:
                    seg.append(dat)
    else:
        seg = string.splitlines()

    ecptxt = []
    for dat in seg:
        dat = dat.split('#')[0].strip()
        dat_upper = dat.upper()
        if (dat and not dat_upper.startswith('END') and not dat_upper.startswith('ECP')):
            ecptxt.append(dat)
    return _parse_ecp(ecptxt)

def _parse_ecp(raw_ecp):
    ecp_add = []
    nelec = None
    for line in raw_ecp:
        dat = line.strip()
        if not dat or dat.startswith('#'): # comment line
            continue
        elif dat[0].isalpha():
            keys = dat.split()
            if len(keys) == 1:
                key = keys[0].upper()
            else:
                key = keys[1].upper()
            if key == 'NELEC':
                nelec = int(dat.split()[2])
                continue
            elif key == 'UL':
                ecp_add.append([-1])
            elif key in MAPSPDF:
                ecp_add.append([MAPSPDF[key]])
            else:
                raise BasisNotFoundError('Not basis data')
            # up to r^6
            by_ang = [[] for i in range(7)]
            ecp_add[-1].append(by_ang)
        else:
            line = dat.replace('D','e').split()
            l = int(line[0])
            try:
                coef = [float(x) for x in line[1:]]
            except ValueError:
                if DISABLE_EVAL:
                    raise ValueError('Failed to parse ecp %s' % line)
                else:
                    coef = list(eval(','.join(line[1:])))
            except Exception:
                raise BasisNotFoundError('Not basis data')
            if any(x != 0 for x in coef[1:]):
                by_ang[l].append(coef)

    if nelec is None:
        return []

    if not ecp_add:
        raise BasisNotFoundError(f'ECP data not found in "{raw_ecp}"')

    ecp_sorted = sorted(ecp_add, key=lambda shell: shell[0])
    return [nelec, ecp_sorted]

def load(basisfile, symb):
    '''Load ECP for atom of symb from file'''
    return _parse_ecp(_search_ecp(basisfile, symb))

def _search_ecp(basisfile, symb):
    symb = _std_symbol(symb)
    with open(basisfile, 'r') as fin:
        fdata = re.split(ECP_DELIMITER, fin.read())
    if len(fdata) <= 1:
        return []

    fdata = fdata[1].splitlines()
    for i, dat in enumerate(fdata):
        dat0 = dat.split(None, 1)
        if dat0 and dat0[0] == symb:
            break
    seg = []
    for dat in fdata[i:]:
        dat = dat.strip()
        if dat:  # remove empty lines
            if ((dat[0].isalpha() and dat.split(None, 1)[0].upper() != symb.upper())):
                return seg
            else:
                seg.append(dat)
    return []

def convert_ecp_to_nwchem(symb, ecp):
    '''Convert the internal ecp format to NWChem format string'''
    symb = _std_symbol(symb)
    res = ['%-2s nelec %d' % (symb, ecp[0])]

    for ecp_block in ecp[1]:
        l = ecp_block[0]
        if l == -1:
            res.append('%-2s ul' % symb)
        else:
            res.append('%-2s %s' % (symb, SPDF[l].lower()))
        for r_order, dat in enumerate(ecp_block[1]):
            for e,c in dat:
                res.append('%d    %15.9f  %15.9f' % (r_order, e, c))
    return '\n'.join(res)
