#!/usr/bin/env python

'''
PBC-SOC integrals
'''

from pyscf.pbc import gto

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'ccpvdz'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.build()

#
# 1-center approximation
#
def get_1c_pvxp(cell, kpts=None):
    import numpy
    atom_slices = cell.offset_nr_by_atom()
    nao = cell.nao_nr()
    mat_soc = numpy.zeros((3,nao,nao))
    for ia in range(cell.natm):
        ish0, ish1, p0, p1 = atom_slices[ia]
        shls_slice = (ish0, ish1, ish0, ish1)
        with cell.with_rinv_as_nucleus(ia):
            z = -cell.atom_charge(ia)
            # Apply Koseki effective charge on z?
            w = z * cell.intor('int1e_prinvxp', comp=3, shls_slice=shls_slice)
        mat_soc[:,p0:p1,p0:p1] = w
    return mat_soc

#
# SOC with lattice summation (G != 0)
#
from pyscf.pbc.x2c.x2c1e import get_pbc_pvxp
from pyscf.pbc.df import GDF
mydf = GDF(cell)
soc_pbc = get_pbc_pvxp(mydf)
soc_1c = get_1c_pvxp(cell)

print('PBC and 1-center SOC difference', abs(soc_pbc - soc_1c).max())
