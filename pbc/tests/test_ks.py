import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import rks as pbcrks

def test_ks(pseudo=None):
    # The molecular calculation
    mol = gto.Mole()
    mol.unit = 'B'
    L = 60
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    # these are some exponents which are not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    mol.build()

    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print "Molecular DFT energy"
    print (m.scf()) # -2.64096172441

    print "coordinates"
    print np.array([mol.atom_coord(i) for i in range(mol.natm)])

    # The periodic calculation
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([80,80,80])
    cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()

    mf = pbcrks.RKS(cell)
    mf.xc = 'LDA,VWN_RPA'
    mf.kpt = np.reshape(np.array([1,1,1]), (3,1))
    print (mf.scf()) 
    # gs    mf.scf()
    # 80    -2.63907898485
    # 90    -2.64065784113
    # 100   -2.64086844062

if __name__ == '__main__':
    test_ks()
    #test_ks('gth-lda')
