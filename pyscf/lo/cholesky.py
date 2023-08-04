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
    D = np.dot(mo_coeff, mo_coeff.T)
    L, piv, rank = pivoted_cholesky(D, lower=True)
    if rank < nmo:
        raise RuntimeError('rank of matrix lower than the number of orbitals')

    # Permute L back to the original order of the AOs.
    # Superfluous columns are cropped out.
    P = np.zeros((nao, nao))
    P[piv, np.arange(nao)] = 1
    mo_loc = np.dot(P, L[:, :nmo])

    return mo_loc


if __name__ == "__main__":

    import numpy
    from pyscf.gto import Mole
    from pyscf.scf import RHF
    from pyscf.tools.mo_mapping import mo_comps

    mol = Mole()
    mol.atom = '''
    C        0.681068338      0.605116159      0.307300799
    C       -0.733665805      0.654940451     -0.299036438
    C       -1.523996730     -0.592207689      0.138683275
    H        0.609941801      0.564304456      1.384183068
    H        1.228991034      1.489024155      0.015946420
    H       -1.242251083      1.542928348      0.046243898
    H       -0.662968178      0.676527364     -1.376503770
    H       -0.838473936     -1.344174292      0.500629028
    H       -2.075136399     -0.983173387     -0.703807608
    H       -2.212637905     -0.323898759      0.926200671
    O        1.368219958     -0.565620846     -0.173113101
    H        2.250134219     -0.596689848      0.204857736
    '''
    mol.basis = 'STO-3G'
    mol.build()

    mf = RHF(mol)
    mf.kernel()

    nocc = numpy.count_nonzero(mf.mo_occ > 0)
    mo_loc = cholesky_mos(mf.mo_coeff[:, :nocc])
    print('LMO    Largest coefficients')
    numpy.set_printoptions(precision=3, suppress=True, sign=' ')
    for i in range(nocc):
        li = numpy.argsort(abs(mo_loc[:, i]))
        print('{0:3d}    {1}'. format(i, mo_loc[li[:-6:-1], i]))
