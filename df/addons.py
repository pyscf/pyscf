#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo

class load(ao2mo.load):
    '''load 3c2e integrals from hdf5 file
    Usage:
        with load(cderifile) as eri:
            print eri.shape
    '''
    pass


def aug_etb_for_weigend(mol, beta=2.3):
    '''augment weigend basis with even tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    '''
    uniq_atoms = set([a[0] for a in mol._atom])

    dfbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.mole._charge(symb)
        if nuc_charge <= 36:  # Kr
            dfbasis[symb] = 'weigend'
        #?elif symb in mol._ecp:
        else:
            conf = lib.parameters.ELEMENTS[nuc_charge][2]
            max_shells = 4 - conf.count(0)
            emin_by_l = [1e3] * 8
            emax_by_l = [0] * 8
            for b in mol._basis[symb]:
                l = b[0]
                if l >= max_shells+1:
                    continue

                if isinstance(b[1], int):
                    es = numpy.array(b[2:])[:,0]
                else:
                    es = numpy.array(b[1:])[:,0]
                emax_by_l[l] = max(es.max(), emax_by_l[l])
                emin_by_l[l] = min(es.min(), emin_by_l[l])

            l_max = 8 - emax_by_l.count(0)
            emin_by_l = numpy.array(emin_by_l[:l_max])
            emax_by_l = numpy.array(emax_by_l[:l_max])
# Estimate the exponents ranges by geometric average
            emax = numpy.sqrt(numpy.einsum('i,j->ij', emax_by_l, emax_by_l))
            emin = numpy.sqrt(numpy.einsum('i,j->ij', emin_by_l, emin_by_l))
            liljsum = numpy.arange(l_max)[:,None] + numpy.arange(l_max)
            emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max*2-1)]
            emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max*2-1)]
            # Tune emin and emax
            emin_by_l = numpy.array(emin_by_l) * 2  # *2 for same center basis
            emax_by_l = numpy.array(emax_by_l) / (numpy.arange(l_max*2-1)*.5+1)

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            etb = [(l, n, emin_by_l[l], beta)
                   for l, n in enumerate(ns.astype(int)) if n > 0]
            dfbasis[symb] = gto.expand_etbs(etb)

    return dfbasis
