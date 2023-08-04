#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

'''
Second order X2C-SCF solver
'''

import numpy
from pyscf.scf import hf
from pyscf.x2c import x2c
# To ensure .gen_response() methods are registered
from pyscf.x2c import _response_functions  # noqa
from pyscf.soscf import newton_ah

gen_g_hop_x2chf = newton_ah.gen_g_hop_dhf

def newton(mf):
    '''Co-iterative augmented hessian (CIAH) second order SCF solver

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> mf = x2c.RHF(mol).newton()
    >>> mf.kernel()
    -1.0811707843774987
    '''
    assert isinstance(mf, x2c.SCF)

    if isinstance(mf, newton_ah._CIAH_SOSCF):
        return mf

    if isinstance(mf, x2c.RHF):
        class SecondOrderX2CHF(newton_ah._CIAH_SOSCF, mf.__class__):
            gen_g_hop = gen_g_hop_x2chf

            def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
                nmo = mo_occ.size
                nocc = numpy.count_nonzero(mo_occ)
                nvir = nmo - nocc
                dx = dx.reshape(nvir, nocc)
                dx_aa = dx[::2,::2]
                dr_aa = hf.unpack_uniq_var(dx_aa.ravel(), mo_occ[::2])
                u = numpy.zeros((nmo, nmo), dtype=dr_aa.dtype)
                # Allows only the rotation within the up-up space and down-down space
                u[::2,::2] = u[1::2,1::2] = newton_ah.expmat(dr_aa)
                return numpy.dot(u0, u)

            def rotate_mo(self, mo_coeff, u, log=None):
                mo = numpy.dot(mo_coeff, u)
                return mo
    else:
        class SecondOrderX2CHF(newton_ah._CIAH_SOSCF, mf.__class__):
            gen_g_hop = gen_g_hop_x2chf

            def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
                dr = hf.unpack_uniq_var(dx, mo_occ)
                return numpy.dot(u0, newton_ah.expmat(dr))

            def rotate_mo(self, mo_coeff, u, log=None):
                mo = numpy.dot(mo_coeff, u)
                return mo

    return SecondOrderX2CHF(mf)
