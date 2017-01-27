#!/usr/bin/env python

import numpy

def spatial2spinorb(t1_or_t2):
    '''Convert T1/T2 of spatial orbital representation to T1/T2 of
    spin-orbital representation'''
    if t1_or_t2.ndim == 2:
        t1 = t1_or_t2
        nocc, nvir = t1.shape
        nocc2 = nocc * 2
        nvir2 = nvir * 2
        t1s = numpy.zeros((nocc2,nvir2))
        t1s[ ::2, ::2] = t1
        t1s[1::2,1::2] = t1
        return t1s
    else:
        t2 = t1_or_t2
        nocc, nvir = t2.shape[::2]
        nocc2 = nocc * 2
        nvir2 = nvir * 2
        t2s = numpy.zeros((nocc2,nocc2,nvir2,nvir2))
        t2s[ ::2,1::2, ::2,1::2] = t2
        t2s[1::2, ::2,1::2, ::2] = t2
        t2s[ ::2,1::2,1::2, ::2] =-t2.transpose(0,1,3,2)
        t2s[1::2, ::2, ::2,1::2] =-t2.transpose(0,1,3,2)
        t2s[ ::2, ::2, ::2, ::2] = t2 - t2.transpose(0,1,3,2)
        t2s[1::2,1::2,1::2,1::2] = t2 - t2.transpose(0,1,3,2)
        return t2s
