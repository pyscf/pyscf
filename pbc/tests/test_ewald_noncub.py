import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf

def test_ewald():
    cell = pbcgto.Cell()

    cell.unit = 'B'
    Lx = 6.
    Ly = 9.
    Lz = 30.
    cell.h = np.array([[Lx,0.5,0.5],
                       [0.5,Ly,0.5],
                       [0.5,0.5,Lz]])
    #  Note here we use a much larger gs than in the earlier test
    cell.gs = np.array([10,10,10])
    cell.nimgs = [1,1,1]

    cell.atom.extend([['He', (1, 0.5*Ly, 0.5*Lz)],
                      ['He', (2, 0.5*Ly, 0.5*Lz)]])
    # these are some exponents which are not hard to integrate
    cell.basis = {'He': [[0, (1.0, 1.0)]]}

    cell.verbose = 5
    cell.build()

    ew_eta0, ew_cut0 = cell.get_ewald_params(1.e-3)
    print ew_eta0, ew_cut0

    ew_cut = (20,20,20)
    for ew_eta in [0.05, 0.07, 0.1, 0.3]:
        print pbchf.ewald(cell, ew_eta, ew_cut) # 4.53091146254

    for precision in [1.e-3, 1.e-5, 1.e-7, 1.e-9]:
        ew_eta0, ew_cut0 = cell.get_ewald_params(precision)
        print "precision", precision, ew_eta0, ew_cut0
        print pbchf.ewald(cell, ew_eta0, ew_cut0)

if __name__ == '__main__':
    test_ewald()
