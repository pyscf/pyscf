#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

'''
Density fitting
===============

This module provides the fundamental functions to handle the 3-index tensors
(including the 3-center 2-electron AO and MO integrals, the Cholesky
decomposed integrals) required by the density fitting method or the RI
(resolution of identity) approximation.

Simple usage::

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = dft.RKS(mol).density_fit().run()
'''

from . import incore
from . import outcore
from . import addons
from .addons import load, aug_etb, DEFAULT_AUXBASIS, make_auxbasis, make_auxmol
from .df import DF, DF4C

from . import r_incore

def density_fit(obj, *args, **kwargs):
    '''Given object, apply density fitting to replace the default 2e integrals.'''
    return obj.density_fit(*args, **kwargs)

