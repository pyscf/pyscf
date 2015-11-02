import sys
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import hf

def main(n): 
    cell = pbcgto.Cell()

    cell.unit = 'A'
    cell.atom = '''
      H    0.0000    0.0000    0.0000;
      H    0.7414    0.0000    0.0000;
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'

    # Slightly greater than 2*(bond length)
    L = 1.5 
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
    print kmf.scf()
    """
    dm = kmf.make_rdm1()

    tao = hf.get_t(cell) 
    print "Kinetic energy ::", np.trace(np.dot(tao,dm))

    vpp = hf.get_pp(cell)
    print "Pseudopotential energy ::", np.trace(np.dot(vpp,dm))

    kmf.get_veff()
    print "XC energy ::", kmf._exc
    """

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print 'usage: n' 
        sys.exit(1)
    n = int(args[0])
    
    main(n)

'''
>>> python h2.py 8
converged SCF energy = -1.93426844243467
'''
