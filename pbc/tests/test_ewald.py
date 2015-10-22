import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import tools

def test_ewald():
    cell = pbcgto.Cell()

    cell.unit = 'B'
    Lx = Ly = Lz = 5.
    cell.h = np.diag([Lx,Ly,Lz])
    cell.gs = np.array([20,20,20])
    cell.nimgs = [1,1,1]

    cell.atom.extend([['He', (2, 0.5*Ly, 0.5*Lz)],
                      ['He', (3, 0.5*Ly, 0.5*Lz)]])
    # these are some exponents which are not hard to integrate
    cell.basis = {'He': [[0, (1.0, 1.0)]]}

    cell.verbose = 5
    cell.build()

    ew_eta0, ew_cut0 = cell.get_ewald_params(1.e-3)
    print ew_eta0, ew_cut0

    ew_cut = (20,20,20)
    for ew_eta in [0.05, 0.1, 0.2, 1]:
        print tools.ewald(cell, ew_eta, ew_cut) # -0.468640671931

    for precision in [1.e-3, 1.e-5, 1.e-7, 1.e-9]:
        ew_eta0, ew_cut0 = cell.get_ewald_params(precision)
        print "precision", precision, ew_eta0, ew_cut0
        print tools.ewald(cell, ew_eta0, ew_cut0) 
        # Ewald values
        # 1.e-3: -0.469112631739
        # 1.e-5: -0.468642153932
        # 1.e-7: -0.468640678042
        # 1.e-9: -0.46864067196

if __name__ == '__main__':
    test_ewald()
