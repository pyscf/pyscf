#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Paul J. Robinson <pjrobinson@ucla.edu>
#

import warnings
warnings.warn('This is an incomplete version of CDFT method. See also another '
              'implementation of cDFT in '
              'pyscf/examples/1-advanced/033-constrained_dft.py',
              DeprecationWarning)

'''
This is a purpose built constrained dft implementation which allows the
shifting of an orbital (or a linear combination of orbitals) by an arbitrary
constant.  Allows the freedom to select thine own basis
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf import lo
from pyscf.pbc import gto, dft

def cdft(mf,cell,offset,orbital,basis=None):
    '''
    Input:
        mf -- a mean field object for DFT or (in principle) HF (doesn't really matter)
        shift -- float -- a semi arbitrary energy which displaces the selected orbitals by the diagonal
        orbital -- int -- indicating which orbital are shifted in the selected basis
        basis -- 2D numpy array -- the working basis in the basis of AOs from 'cell' (Defaults to AO basis)

    Returns:
        mf -- converged mean field object (with AO basis)

    '''
    if basis is not None:
        a = basis
    else:
        a = numpy.eye(cell._bas.shape[1])

    #
    # Here we run the calculation using each IAO as an offset parameter
    #

    iaoi = a.T[orbital,:]
    # gonna try normalizing to see if that makes life better
    # iaoi = iaoi / numpy.linalg.norm(iaoi)
    mf.shift_hamiltonian= numpy.diag(iaoi) * offset
    mf.constrained_dft = True

    def get_veff(*args, **kwargs):
        vxc = dft.rks.get_veff(mf, *args, **kwargs)
        # Make a shift to the Veff matrix, while ecoul and exc are kept unchanged.
        # The total energy is computed with the correct ecoul and exc.
        vxc = lib.tag_array(vxc+mf.shift_hamiltonian,
                            ecoul=vxc.ecoul, exc=vxc.exc, vj=None, vk=None)
        return vxc
    mf.get_veff = get_veff
    return mf

def fast_iao_mullikan_pop(mf,cell,a=None):
    '''
    Input: mf -- a preconverged mean field object
    Returns: mullikan population analysis in the basisIAO a
    '''

    #
    # here we convert the density matrix to the IAO basis
    #
    if a is None:
        a = numpy.eye(mf.make_rdm1().shape[1])
    #converts the occupied MOs to the IAO basis
    #ovlpS = mf.get_ovlp()
    #CIb = reduce(numpy.dot, (a.T, ovlpS , mf.make_rdm1()))

    #
    # This is the mullikan population below here
    #

    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    mo_occ = reduce(numpy.dot, (a.T, mf.get_ovlp(), mo_occ))
    dm = numpy.dot(mo_occ, mo_occ.T) * 2
    pmol = cell.copy()
    pmol.build(False, False, basis='minao')
    return mf.mulliken_pop(pmol, dm, s=numpy.eye(pmol.nao_nr()))
