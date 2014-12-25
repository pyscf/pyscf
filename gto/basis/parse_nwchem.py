#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# parse NWChem format
#

MAXL = 8
MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'J': 7}

# parse the basis text which is in NWChem format, return an internal basis
# format which can be assigned to gto.mole.basis
def parse_str(string):
    bastxt = [x for x in string.split('\n') \
              if x.strip() and 'END' not in x and '#BASIS SET' not in x]

    basis_add = []
    for dat in bastxt:
        key = dat.split()[1].upper()
        if key == 'SP':
            basis_add.append([0])
            basis_add.append([1])
        elif key in MAPSPDF:
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

def parse(basisfile, symb):
    basis_add = []
    for dat in search_seg(basisfile, symb):
        if symb in dat:
            key = dat.split()[1].upper()
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

def search_seg(basisfile, symb):
    fin = open(basisfile, 'r')
    fdata = fin.read().split('#BASIS SET')
    for dat in fdata[1:]:
        if symb+' ' in dat:
            break
    fin.close()
    # remove blank lines
    return [x for x in dat.split('\n')[1:] if x.strip() and 'END' not in x]

