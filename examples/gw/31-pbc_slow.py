from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf.krhf_slow import TDRHF
from pyscf.pbc.gw import KRGW

cell = Cell()
cell.atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.67   1.68   1.69
'''
cell.basis = {'C': [[0, (0.8, 1.0)],
                    [1, (1.0, 1.0)]]}
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 7
cell.build()

model = KRHF(cell, cell.make_kpts([2, 1, 1]))
model.kernel()

model_td = TDRHF(model)
model_td.kernel()

model_gw = KRGW(model_td)
model_gw.kernel()

print(model_gw.mo_energy)
