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
#

'''Analytic GTH-PP integrals for open boundary conditions.

See also pyscf/pbc/gto/pseudo/pp_int.py
'''

import numpy as np
from pyscf import lib
from pyscf.gto.mole import ATOM_OF, intor_cross

def get_gth_pp(mol):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore

    # Analytical integration for get_pp_loc_part1(cell).
    fakemol = pp_int.fake_cell_vloc(mol, 0)
    vpploc = 0
    if fakemol.nbas > 0:
        charges = fakemol.atom_charges()
        atmlst = fakemol._bas[:,ATOM_OF]
        v = incore.aux_e2(mol, fakemol, 'int3c2e', aosym='s2', comp=1)
        vpploc = np.einsum('...i,i->...', v, -charges[atmlst])

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    for cn in range(1, 5):
        fakemol = pp_int.fake_cell_vloc(mol, cn)
        if fakemol.nbas > 0:
            v = incore.aux_e2(mol, fakemol, intors[cn], aosym='s2', comp=1)
            vpploc += np.einsum('...i->...', v)

    if isinstance(vpploc, np.ndarray):
        vpploc = lib.unpack_tril(vpploc)

    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas
    ppnl_half = []
    intors = ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi')
    for i, intor in enumerate(intors):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(intor_cross(intor, fakemol, mol))
        else:
            ppnl_half.append(None)
    fakemol._bas = _bas

    nao = mol.nao
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao))
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        vpploc += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    return vpploc
