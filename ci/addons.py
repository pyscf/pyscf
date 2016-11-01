#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy

def spatial2spinorb(cisdvec):
    '''Convert CISD vector spatial orbital representation to spin-orbital
    representation'''
    nocc = myci.nocc
    nmo  = myci.nmo
    nvir = nmo - nocc
    nocc2 = nocc * 2
    nvir2 = nvir * 2

    cisp = numpy.zeros((1+nocc2*nvir2+(nocc2*nvir2)**2))
    ci1sp = cisp[1:nocc2*nvir2+1].reshape(nocc2,nvir2)
    ci2sp = cisp[nocc2*nvir2+1: ].reshape(nocc2,nocc2,nvir2,nvir2)
    c1 = cisdvec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = cisdvec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)

    cisp[0] = cisdvec[0]
    ci1sp[ ::2, ::2] = c1
    ci1sp[1::2,1::2] = c1
    ci2sp[ ::2,1::2, ::2,1::2] = c2
    ci2sp[1::2, ::2,1::2, ::2] = c2
    ci2sp[ ::2,1::2,1::2, ::2] =-c2.transpose(0,1,3,2)
    ci2sp[1::2, ::2, ::2,1::2] =-c2.transpose(0,1,3,2)
    ci2sp[ ::2, ::2, ::2, ::2] = c2 - c2.transpose(0,1,3,2)
    ci2sp[1::2,1::2,1::2,1::2] = c2 - c2.transpose(0,1,3,2)
    return cisp
