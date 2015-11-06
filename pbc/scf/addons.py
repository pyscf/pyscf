#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
from pyscf.pbc.scf import scfint


def project_mo_nr2nr(cell1, mo1, cell2):
    r''' Project orbital coefficients

    .. math::

        |\psi1> = |AO1> C1

        |\psi2> = P |\psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2

        C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = scfint.get_ovlp(cell2)
    s21 = scfint.get_int1e_cross('cint1e_ovlp_sph', cell2, cell1)
    mo2 = numpy.dot(s21, mo1)
    return scipy.linalg.cho_solve(scipy.linalg.cho_factor(s22), mo2)

