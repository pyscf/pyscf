#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#
# parse CP2K format
#

MAXL = 8

def parse(string):
    '''Parse the basis text which is in CP2K format, return an internal
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

def _parse(blines):
    header_ln = blines.pop(0)
    nsets = int(blines.pop(0))
    basis = []
    for n in range(nsets):
        comp = [int(p) for p in blines.pop(0).split()]
        n, lmin, lmax, nexps, ncontractions = comp[0], comp[1], comp[2], comp[3], comp[4:]
        basis_n = [[l] for l in range(lmin,lmax+1)]
        for nexp in range(nexps):
            bfun = [float(x) for x in blines.pop(0).split()]
            exp = bfun.pop(0)
            for i,l in enumerate(range(lmin,lmax+1)):
                cl = [exp]
                for c in range(ncontractions[i]):
                    cl.append(bfun.pop(0))
                basis_n[i].append(tuple(cl))
        basis.extend(basis_n)
    bsort = []
    for l in range(MAXL):
        bsort.extend([b for b in basis if b[0] == l])
    return bsort
        
def search_seg(basisfile, symb):
    fin = open(basisfile, 'r')
    fdata = fin.read().split('#BASIS SET')
    fin.close()
    for dat in fdata[1:]:
        if symb+' ' in dat:
            # remove blank lines
            return [x.strip() for x in dat.split('\n')[1:]
                    if x.strip() and 'END' not in x]
    raise RuntimeError('Basis not found for  %s  in  %s' % (symb, basisfile))

