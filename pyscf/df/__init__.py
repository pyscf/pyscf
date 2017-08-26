#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

'''
Density fitting
===============

Simple usage::

    >>> from pyscf import gto, scf, df
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = df.DF(mol).update(scf.RHF(mol)).run()
    >>> mf = df.density_fit(scf.RHF(mol)).run()
'''

from . import incore
from . import outcore
from . import addons
from .incore import format_aux_basis
from .addons import load, aug_etb, DEFAULT_AUXBASIS, make_auxbasis
from .df import DF, DF4C

from . import r_incore

def density_fit(obj):
    '''Given object, apply density fitting to replace the default 2e integrals.'''
    return obj.density_fit()
