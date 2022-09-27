#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Authors: Shiv Upadhyay <shivnupadhyay@gmail.com>
#

'''
Valence Virtual Orbitals
ref. 10.1021/acs.jctc.7b00493
'''

import numpy
import scipy.linalg
from pyscf.lo import iao
from pyscf.lo import orth
from pyscf.lo import ibo
from pyscf import __config__


def vvo(mol, orbocc, orbvirt, iaos=None, s=None, verbose=None):
    '''Valence Virtual Orbitals ref. 10.1021/acs.jctc.7b00493

    Valence virtual orbitals can be formed from the singular value
    decomposition of the overlap between the canonical molecular orbitals
    and an accurate underlying atomic basis set. This implementation uses
    the intrinsic atomic orbital as this underlying set. VVOs can also be
    formed from the null space of the overlap of the canonical molecular
    orbitals and the underlying atomic basis sets (IAOs). This is not
    implemented here.

    Args:
        mol : the molecule or cell object

        orbocc : occupied molecular orbital coefficients

        orbvirt : virtual molecular orbital coefficients

    Kwargs:
        iaos : 2D array
            the array of IAOs

        s : 2D array
            the overlap array in the ao basis

    Returns:
        VVOs in the basis defined in mol object.
    '''

    if s is None:
        if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
            if isinstance(orbocc, numpy.ndarray) and orbocc.ndim == 2:
                s = mol.pbc_intor('int1e_ovlp', hermi=1)
            else:
                raise NotImplementedError('k-points crystal orbitals')
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    if iaos is None:
        iaos = iao.iao(mol, orbocc)

    nvvo = iaos.shape[1] - orbocc.shape[1]

    # Symmetrically orthogonalization of the IAO orbitals as Knizia's
    # implementation.  The IAO returned by iao.iao function is not orthogonal.
    iaos = orth.vec_lowdin(iaos, s)

    #S = reduce(np.dot, (orbvirt.T, s, iaos))
    S = numpy.einsum('ji,jk,kl->il', orbvirt.conj(), s, iaos, optimize=True)
    U, sigma, Vh = scipy.linalg.svd(S)
    U = U[:, 0:nvvo]
    vvo = numpy.einsum('ik,ji->jk', U, orbvirt, optimize=True)
    return vvo

def livvo(mol, orbocc, orbvirt, locmethod='IBO', iaos=None, s=None,
          exponent=4, grad_tol=1e-8, max_iter=200, verbose=None):
    '''Localized Intrinsic Valence Virtual Orbitals ref. 10.1021/acs.jctc.7b00493

    Localized Intrinsic valence virtual orbitals are formed when the valence
    virtual orbitals are localized using an IBO-type of localization. Here
    the VVOs are created in the IAO basis then the IBO localization functions
    are called to localize the VVOs.

    Args:
        mol : the molecule or cell object

        orbocc : occupied molecular orbital coefficients

        orbvirt : virtual molecular orbital coefficients

    Kwargs:
        locmethod : string
            the localization method 'PM' for Pipek Mezey localization or 'IBO'
            for the IBO localization

        iaos : 2D array
            the array of IAOs

        s : 2D array
            the overlap array in the ao basis

    Returns:
        LIVVOs in the basis defined in mol object.
    '''
    if s is None:
        if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
            if isinstance(orbocc, numpy.ndarray) and orbocc.ndim == 2:
                s = mol.pbc_intor('int1e_ovlp', hermi=1)
            else:
                raise NotImplementedError('k-points crystal orbitals')
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    if iaos is None:
        iaos = iao.iao(mol, orbocc)

    vvos = vvo(mol, orbocc, orbvirt, iaos=iaos, s=s)
    locmethod = locmethod.strip().upper()
    if locmethod == 'PM':
        EXPONENT = getattr(__config__, 'lo_ibo_PipekMezey_exponent', exponent)
        livvos = ibo.PipekMezey(mol, vvos, iaos, s, exponent=EXPONENT)
        del (EXPONENT)
    else:
        livvos = ibo.ibo_loc(mol, vvos, iaos, s, exponent=exponent,
                             grad_tol=grad_tol, max_iter=max_iter,
                             verbose=verbose)
    return livvos
