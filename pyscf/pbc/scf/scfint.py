#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF (Hartree-Fock and DFT) tools for periodic systems at a *single* k-point,
    using analytical GTO integrals instead of PWs.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.

'''

import numpy as np
import pyscf.pbc.scf
from pyscf.pbc import gto as pgto

print('This module is deporacted and will be removed in future release.  '
      'Please use cell.pbc_intor and pbc.hf.get_hcore.')

def get_hcore(cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.'''
    if kpt is None:
        kpt = np.zeros(3)

    # TODO: these are still on grid
    if cell.pseudo is None:
        hcore = pyscf.pbc.scf.hf.get_nuc(cell, kpt)
    else:
        hcore = pyscf.pbc.scf.hf.get_pp(cell, kpt)
    hcore += get_t(cell, kpt)

    return hcore

def get_int1e_cross(intor, cell1, cell2, kpt=None, comp=1):
    r'''1-electron integrals from two molecules like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    return pgto.intor_cross(intor, cell1, cell2, comp, 0, kpt)

def get_int1e(intor, cell, kpt=None):
    '''Get the one-electron integral defined by `intor` using lattice sums.'''
    return cell.pbc_intor(intor, kpts=kpt)

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.'''
    return cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpt)

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.'''
    return cell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=kpt)

