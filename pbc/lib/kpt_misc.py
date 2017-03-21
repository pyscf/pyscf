#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy

KPT_DIFF_TOL = 1e-6

def is_zero(kpt):
    return abs(numpy.asarray(kpt)).sum() < KPT_DIFF_TOL
gamma_point = is_zero

def member(kpt, kpts):
    kpts = numpy.reshape(kpts, (len(kpts),kpt.size))
    dk = numpy.einsum('ki->k', abs(kpts-kpt.ravel()))
    return numpy.where(dk < KPT_DIFF_TOL)[0]

def unique(kpts):
    kpts = numpy.asarray(kpts)
    nkpts = len(kpts)
    uniq_kpts = []
    uniq_index = []
    uniq_inverse = numpy.zeros(nkpts, dtype=int)
    seen = numpy.zeros(nkpts, dtype=bool)
    n = 0
    for i, kpt in enumerate(kpts):
        if not seen[i]:
            uniq_kpts.append(kpt)
            uniq_index.append(i)
            idx = abs(kpt-kpts).sum(axis=1) < KPT_DIFF_TOL
            uniq_inverse[idx] = n
            seen[idx] = True
            n += 1
    return numpy.asarray(uniq_kpts), numpy.asarray(uniq_index), uniq_inverse
