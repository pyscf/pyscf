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

def parse(string):
    '''Parse the basis text which is in NWChem format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.
    '''
    bastxt = []
    for dat in string.split('\n'):
        x = dat.strip()
        if (x and not x.startswith('#') and not x.startswith('END') and
            not x.startswith('BASIS')):
            bastxt.append(dat)
    return _parse(bastxt)

def load(basisfile, symb):
    return _parse(search_seg(basisfile, symb))

def parse_ecp(string):
    ecptxt = []
    for dat in string.split('\n'):
        x = dat.strip()
        if (x and not x.startswith('#') and not x.startswith('END') and
            not x.startswith('ECP')):
            ecptxt.append(dat)
    return _parse_ecp(ecptxt)

def load_ecp(basisfile, symb):
    return _parse_ecp(search_ecp(basisfile, symb))

def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline().lstrip()
        while not dat.startswith('#BASIS SET:'):
            dat = fin.readline().lstrip()
        # searching
        dat = fin.readline().lstrip()
        while not dat.startswith('END'):
            if symb+' ' in dat:
                seg = []
                while '#BASIS SET:' not in dat:
                    x = dat[:-1].strip()
                    if x and 'END' not in x: # remove blank lines
                        seg.append(x)
                        dat = fin.readline().lstrip()
                return seg
            else:
                while '#BASIS SET:' not in dat:
                    dat = fin.readline().lstrip()
            dat = fin.readline().lstrip()
    raise RuntimeError('Basis not found for  %s  in  %s' % (symb, basisfile))

def search_ecp(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline().lstrip()
        while not dat.startswith('ECP'):
            dat = fin.readline().lstrip()

        dat = fin.readline().lstrip()
        # searching
        while not dat.startswith(symb+' ') and not dat.startswith('END'):
            dat = fin.readline()

        seg = []
        while not dat.startswith('END'):
            if dat[0].isalpha() and symb+' ' not in dat:
                return seg
            if dat: # remove blank lines
                seg.append(dat)
            dat = fin.readline()[:-1].strip()
    raise RuntimeError('Basis not found for  %s  in  %s' % (symb, basisfile))

def _parse(raw_basis):
    basis_add = []
    for dat in raw_basis:
        if dat.startswith('#'):
            continue
        elif dat[0].isalpha():
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

def _parse_ecp(raw_ecp):
    ecp_add = []
    for dat in raw_ecp:
        if dat.startswith('#'): # comment line
            continue
        elif dat[0].isalpha():
            key = dat.split()[1].upper()
            if key == 'NELEC':
                nelec = int(dat.split()[2])
                continue
            elif key == 'UL':
                ecp_add.append([-1])
            else:
                ecp_add.append([MAPSPDF[key]])
            by_ang = [[], [], []]
            ecp_add[-1].append(by_ang)
        else:
            line = dat.replace('D','e').split()
            l = int(line[0])
            by_ang[l].append([float(x) for x in line[1:]])
    bsort = []
    for l in range(-1, MAXL):
        bsort.extend([b for b in ecp_add if b[0] == l])
    return [nelec, bsort]

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O', basis='6-31g')
    load_ecp('ecp_lanl2dz.dat', 'Na')
