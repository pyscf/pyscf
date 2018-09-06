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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Analytic GTH-PP integrals for open boundary conditions.

See also pyscf/pbc/gto/pseudo/pp_int.py
'''

import numpy
from pyscf import lib

def get_gth_pp(mol):
    from pyscf.gto import ATOM_OF
    from pyscf.pbc.gto import Cell
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore

    # Analytical integration for get_pp_loc_part1(cell).
    fakemol = pp_int.fake_cell_vloc(mol, 0)
    vpploc = 0
    if fakemol.nbas > 0:
        charges = fakemol.atom_charges()
        atmlst = fakemol._bas[:,ATOM_OF]
        v = incore.aux_e2(mol, fakemol, 'int3c2e', aosym='s2', comp=1)
        v = numpy.einsum('...i,i->...', v, -charges[atmlst])
        vpploc += lib.unpack_tril(v)

    # To compute the rest part of GTH PP, mimic the mol with a 0D cell.
    cell_0D = mol.view(Cell)
    cell_0D.dimension = 0
    cell_0D.a = numpy.eye(3)
    vpploc += pp_int.get_pp_loc_part2(cell_0D).real
    vpploc += pp_int.get_pp_nl(cell_0D).real
    return vpploc


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df
    cell = pbcgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade',
                   'He': pbcgto.pseudo.parse('''He
    2
     0.40000000    3    -1.98934751    -0.75604821    0.95604821
    2
     0.29482550    3     1.23870466    .855         .3
                                       .71         -1.1
                                                    .9
     0.32235865    2     2.25670239    -0.39677748
                                        0.93894690
                                                 ''')}
    cell.a = numpy.eye(3)
    cell.dimension = 0
    cell.build()
    mf = pbcscf.RHF(cell)
    mf.with_df = df.AFTDF(cell)
    mf.run()

    mol = cell.to_mol()
    scf.RHF(mol).run()
