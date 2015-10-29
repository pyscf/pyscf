import sys
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def main(n): 
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

    L = 5.430697500 
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])

    #cell.verbose = 4
    cell.build()

    print "Cell nimgs =", cell.nimgs
    print "Cell _basis =", cell._basis
    print "Cell _pseudo =", cell._pseudo
    print "Cell nelectron =", cell.nelectron

    kmf = pbcdft.RKS(cell)
    kmf.xc = 'lda,vwn'
    return kmf.scf()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print 'usage: n' 
        sys.exit(1)
    n = int(args[0])
    
    main(n)
