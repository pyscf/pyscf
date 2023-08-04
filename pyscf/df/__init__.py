#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
from .df import DF, GDF, DF4C, GDF4C

from . import r_incore

def density_fit(obj, *args, **kwargs):
    '''Given SCF/MCSCF or post-HF object, use density fitting technique to
    approximate the 2e integrals.'''
    return obj.density_fit(*args, **kwargs)

