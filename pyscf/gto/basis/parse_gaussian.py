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
Parsers for basis set in Gaussian program format
'''

MAXL = 8
MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'K': 7}

def parse(string):
    '''Parse the basis text which is in NWChem format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.
    '''
    bastxt = []
    for dat in string.splitlines():
        x = dat.split('!')[0].strip()
        if x and x != '****':
            bastxt.append(x)
    return _parse(bastxt)

def load(basisfile, symb):
    return _parse(search_seg(basisfile, symb))

def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline()
        dat = fin.readline()
        _seek(fin, '****')

        dat = fin.readline()
        while dat:
            dat = dat.strip().upper()
            if dat.split(' ', 1)[0] == symb.upper():
                seg = []
                dat = fin.readline().strip()
                while dat:
                    if dat == '****':
                        break
                    seg.append(dat)
                    dat = fin.readline().strip()
                return seg
            else:
                _seek(fin, '****')
            dat = fin.readline()
    return []

def _parse(raw_basis):
    basis_add = []
    for line in raw_basis:
        dat = line.strip()
        if dat.startswith('!'):
            continue
        elif dat[0].isalpha():
            key = dat.split()[0]
            if key == 'SP':
                basis_add.append([0])
                basis_add.append([1])
            else:
                basis_add.append([MAPSPDF[key]])
        else:
            line = [float(x) for x in dat.replace('D','e').split()]
            if key == 'SP':
                basis_add[-2].append([line[0], line[1]])
                basis_add[-1].append([line[0], line[2]])
            else:
                basis_add[-1].append(line)
    bsort = []
    for l in range(MAXL):
        bsort.extend([b for b in basis_add if b[0] == l])
    return bsort

def _seek(fbasis, test_str):
    dat = fbasis.readline()
    while dat:
        if test_str in dat:
            return True
        dat = fbasis.readline()
    return False

if __name__ == '__main__':
    print(load('def2-qzvp-jkfit.gbs', 'C'))
