import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import rks as pbcrks
import pyscf.pbc.dft.gen_grid
import pyscf.pbc
import pyscf.pbc.dft.numint

def test_ks(pseudo=None):
    # The molecular calculation
    mol = gto.Mole()
    mol.unit = 'B'
    L = 10
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    # these are some exponents which are not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    mol.build()


    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    #m.xc = 'B88,LYP'
    print "Molecular DFT energy"
    print (m.scf()) 
    # LDA,VWN_RPA
    # -2.64096172441 
    # BLYP
    # -2.66058401340308
    # The periodic calculation
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([80,80,80])
    cell.nimgs = [1,1,1]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()

    mf = pbcrks.RKS(cell)
    mf.xc = 'LDA,VWN_RPA'
    # mf.xc = 'B88,LYP'
    print (mf.scf()) 
    # LDA,VWN_RPA
    # gs    mf.scf()
    # 80    -2.64096172553
    # BLYP
    # 80    -2.66058220141

if __name__ == '__main__':
    test_ks()
    #test_ks('gth-lda')
