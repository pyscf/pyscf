#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf.pbc import gto as pbcgto


def project_mo_nr2nr(cell1, mo1, cell2, kpt=None):
    r''' Project orbital coefficients

    .. math::

        |\psi1> = |AO1> C1

        |\psi2> = P |\psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2

        C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = cell2.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpt)
    s21 = pbcgto.intor_cross('cint1e_ovlp_sph', cell2, cell1, kpts=kpt)
    mo2 = numpy.dot(s21, mo1)
    return scipy.linalg.cho_solve(scipy.linalg.cho_factor(s22), mo2)

