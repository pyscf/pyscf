import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def main(): 
    cell = pbcgto.Cell()

    cell.unit = 'A'
    cell.atom = '''
      Si    0.000000000    0.000000000    0.000000000;
      Si    0.000000000    2.715348700    2.715348700;
      Si    2.715348700    2.715348700    0.000000000;
      Si    2.715348700    0.000000000    2.715348700;
      Si    4.073023100    1.357674400    4.073023100;
      Si    1.357674400    1.357674400    1.357674400;
      Si    1.357674400    4.073023100    4.073023100;
      Si    4.073023100    4.073023100    1.357674400
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'

    Lx = Ly = Lz = 5.430697500 
    cell.h = np.diag([Lx,Ly,Lz])
    cell.gs = np.array([8,8,8])

    cell.verbose = 4
    cell.build()

    print "Cell nimgs =", cell.nimgs
    print "Cell _basis =", cell._basis
    print "Cell _pseudo =", cell._pseudo
    print "Cell nelectron =", cell.nelectron

    kmf = pbcdft.RKS(cell)
    kmf.xc = 'b88, lyp'
    return kmf.scf()

if __name__ == '__main__':
    main()
