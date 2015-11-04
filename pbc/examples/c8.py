import sys
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def main(n): 
    cell = pbcgto.Cell()

    cell.unit = 'A'
    cell.atom = '''
      C    0.           0.           0.        ;
      C    0.           1.78339997   1.78339997;
      C    1.78339997   1.78339997   0.        ;
      C    1.78339997   0.           1.78339997;
      C    2.67509998   0.89170002   2.67509998;
      C    0.89170002   0.89170002   0.89170002;
      C    0.89170002   2.67509998   2.67509998;
      C    2.67509998   2.67509998   0.89170002
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'

    L = 3.5668 
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

'''
>>> python c8.py 12
converged SCF energy = -44.8805127845637
'''
