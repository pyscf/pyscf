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
# Author: Peter Pinski <peter.pinski@quantumsimulations.de>
#

'''
Localized molecular orbitals via Cholesky factorization.
The orbitals are usually less well localized than with Boys, Pipek-Mezey, etc.
On the other hand, the procedure is non-iterative and the result unique,
except for degeneracies.

F. Aquilante, T.B. Pedersen, J. Chem. Phys. 125, 174101 (2006)
https://doi.org/10.1063/1.2360264
'''


import numpy as np
from pyscf.lib.scipy_helper import pivoted_cholesky

__all__ = ['cholesky_mos']


def cholesky_mos(mo_coeff):
    '''
    Calculates localized orbitals through a pivoted Cholesky factorization
    of the density matrix.

    Args:
        mo_coeff: block of MO coefficients to be localized

    Returns:
        the localized MOs
    '''
    assert (mo_coeff.ndim == 2)
    nao, nmo = mo_coeff.shape

    # Factorization of a density matrix-like quantity.
    D = np.dot(mo_coeff, mo_coeff.T.conj())
    L, piv, rank = pivoted_cholesky(D, lower=True)
    if rank < nmo:
        raise RuntimeError('rank of matrix lower than the number of orbitals')

    # Permute L back to the original order of the AOs.
    # Superfluous columns are cropped out.
    P = np.zeros((nao, nao), dtype=D.dtype)
    P[piv, np.arange(nao)] = 1
    mo_loc = np.dot(P, L[:, :nmo])

    return mo_loc
