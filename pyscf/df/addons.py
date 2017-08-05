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
    def __init__(self, eri, dataname='j3c'):
        ao2mo.load.__init__(self, eri, dataname)


def aug_etb_for_dfbasis(mol, dfbasis='weigend', beta=2.3, start_at='Rb'):
    '''augment weigend basis with even tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    '''
    nuc_start = gto.mole._charge(start_at)
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.mole._charge(symb)
        if nuc_charge < nuc_start:
            newbasis[symb] = dfbasis
        #?elif symb in mol._ecp:
        else:
            conf = lib.parameters.ELEMENTS[nuc_charge][2]
            max_shells = 4 - conf.count(0)
            emin_by_l = [1e99] * 8
            emax_by_l = [0] * 8
            for b in mol._basis[symb]:
                l = b[0]
                if l >= max_shells+1:
                    continue

                if isinstance(b[1], int):
                    e_c = numpy.array(b[2:])
                else:
                    e_c = numpy.array(b[1:])
                es = e_c[:,0]
                cs = e_c[:,1:]
                es = es[abs(cs).max(axis=1) > 1e-3]
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
            emin_by_l = numpy.array(emin_by_l) * 2  # *2 for alpha+alpha on same center
            emax_by_l = numpy.array(emax_by_l) * 2  #/ (numpy.arange(l_max*2-1)*.5+1)

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            etb = [(l, max(n,1), emin_by_l[l], beta)
                   for l, n in enumerate(numpy.ceil(ns).astype(int))]
            newbasis[symb] = gto.expand_etbs(etb)

    return newbasis

def aug_etb(mol, beta=2.3):
    return aug_etb_for_dfbasis(mol, beta=beta, start_at=0)
