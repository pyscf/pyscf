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
        shift -- float -- a semi aribitrary energy which displaces the selected orbitals by the diagonal
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




if __name__ == '__main__':

    cell = gto.Cell()
    # .a is a matrix for lattice vectors.
    cell.a = '''
            2.4560000896         0.0000000000         0.0000000000
           -1.2280000448         2.1269584693         0.0000000000
            0.0000000000         0.0000000000        10.0000000000
    '''
    cell.atom='''
        C  0.000000000         0.000000000         4.999999702
        C  1.227999862         0.708986051         4.999999702
       '''

    cell.ke_cutoff = 50
    cell.basis = 'gth-tzvp'
    cell.pseudo = 'gth-pbe'
    cell.verbose=0
    cell.charge=0
    cell.unit="Angstrom"
    cell.build()
    cell.rcut*=2

    print("running initial DFT calc to generate IAOs")
    mf = dft.RKS(cell)
    mf.chkfile = 'graphene.chk'
    mf.init_guess = 'chkfile'
    mf.xc = 'pbe,pbe'
    mf.kernel()

    #we need to makVe the IAOs out of a converged calculation
    print("generating IAOs")
    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    a = lo.iao.iao(cell, mo_occ)
    # Orthogonalize IAO
    a = lo.vec_lowdin(a, mf.get_ovlp())

    #arbitrary parameters
    offset = 0.0001
    orbital =4

    print("running constrained dft")

    mf  = cdft(mf,mf.cell,offset,orbital,basis=a)
    population = fast_iao_mullikan_pop(mf,a=a)
    result = numpy.zeros(3)

    result[0] = offset
    result[1] = mf.e_tot
    result[2] = population[0][4]

    print(result)
