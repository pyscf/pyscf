import sys
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import hf

def main(L,n,pseudo=None):
    cell = pbcgto.Cell()

    cell.unit = 'A'
    cell.atom = 'He  0.0  0.0  0.0'
    cell.basis = 'gth-szv'
    cell.pseudo = pseudo 

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
    if len(args) != 2:
        print 'usage: L n' 
        sys.exit(1)
    L = float(args[0])
    n = int(args[1])
    
    main(L,n,'gth-lda')

'''
>>> python he.py 2.0 40
converged SCF energy = -3.05707668619348
'''
