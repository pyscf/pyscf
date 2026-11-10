#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Xubo Wang <wangxubo0201@gmail.com>
#

'''
Kramers-restricted eigenvalue problems in the spinor AO basis.

The spinor AOs are reordered into a Kramers basis (phi, T phi) with
mol.time_reversal_map and the problem is solved by
:func:`pyscf.lib.kramers.eigh`.
'''

import numpy


def pair_sort_ao_idx(mol, four_component=True):
    '''Indices of the spinor AOs in the Kramers basis order
    (phi_1..phi_n, T phi_1..T phi_n), for the large and small components
    together if four_component.'''
    trmaps = mol.time_reversal_map()
    idxA = numpy.where(trmaps > 0)[0]
    idxB = trmaps[idxA] - 1
    if four_component:
        n = trmaps.size
        idx = numpy.hstack((idxA,idxA+n,idxB,idxB+n))
    else:
        idx = numpy.hstack((idxA,idxB))
    return idx


def eigh(mol, h, s=None, iop=0):
    '''HC = SCE (or HC = CE) for Kramers-symmetric matrices in the spinor AO
    basis, two-component (n2c x n2c) or four-component (2n2c x 2n2c).

    The eigenvectors come in Kramers pairs (u, Tu), ordered as given by iop
    (see :func:`pyscf.lib.kramers.eigh`); the rows are in the spinor AO
    order of the input.
    '''
    four_component = h.shape[0] != mol.nao_2c()
    idx = pair_sort_ao_idx(mol, four_component)
    if s is not None:
        s = s[idx[:,None],idx]
    from pyscf.lib import kramers as lib_kramers
    e, c = lib_kramers.eigh(h[idx[:,None],idx], s, iop=iop)
    mo = numpy.empty_like(c)
    mo[idx] = c
    return e, mo
